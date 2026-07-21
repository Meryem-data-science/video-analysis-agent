"""VLM visuel via **Gemini** (google-genai) — remplace l'ancien runner Claude.

Analyse UN segment vidéo (prompt + images motion/frames) et renvoie un dict conforme
au contrat `vlm_schema.build_output_schema`. Indépendant de Claude.

Distinct de `scene_audio.py` (audio) : modèle propre `GEMINI_VLM_MODEL` (fallback
documenté `GEMINI_MODEL`), import google-genai différé, **aucun print** (logs via
`logging`/stderr). Une réponse inexploitable lève `VlmGeminiError` : l'appelant
(run_vlm_agent) journalise l'échec du segment et continue les autres.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vlm_schema import (
    VLM_PROMPT_DIRECTIVES,
    build_output_schema,
    to_gemini_schema,
)

# Modèle VLM par défaut (multimodal). NE PAS confondre avec le modèle AUDIO de
# scene_audio.py : on lit `GEMINI_VLM_MODEL`, et seulement en FALLBACK documenté
# `GEMINI_MODEL`, sinon ce défaut.
DEFAULT_VLM_MODEL = "gemini-2.5-flash"

# Timeout par requête Gemini (ms).
REQUEST_TIMEOUT_MS = 120_000

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Sommeil isolé (mockable / testable) pour le backoff.
_sleep = time.sleep


class VlmGeminiError(Exception):
    """Échec d'analyse VLM via Gemini (réponse absente/non parsable)."""


def resolve_vlm_model(model: Optional[str] = None) -> str:
    """Modèle VLM : argument explicite > $GEMINI_VLM_MODEL > $GEMINI_MODEL (fallback) > défaut."""
    if model and model.strip():
        return model.strip()
    env_vlm = os.environ.get("GEMINI_VLM_MODEL")
    if env_vlm and env_vlm.strip():
        return env_vlm.strip()
    # Fallback DOCUMENTÉ : si seul GEMINI_MODEL (servant aussi à l'audio) est défini.
    env_shared = os.environ.get("GEMINI_MODEL")
    if env_shared and env_shared.strip():
        logging.info("vlm_gemini: fallback sur $GEMINI_MODEL (partagé avec l'audio) pour le VLM.")
        return env_shared.strip()
    return DEFAULT_VLM_MODEL


def _init_gemini():
    """Import différé + création client. Renvoie (client, types) ou lève VlmGeminiError.

    Contrairement à l'audio (dégradation silencieuse), le VLM est l'étape principale :
    on lève une erreur claire si la clé/SDK manquent (l'appelant gère par segment).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise VlmGeminiError("GEMINI_API_KEY absente — impossible d'appeler le VLM Gemini.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - dépend de l'install
        raise VlmGeminiError(
            "'google-genai' non installé. Installez-le : `python -m pip install google-genai`."
        ) from exc
    try:
        http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        return genai.Client(api_key=api_key, http_options=http_options), types
    except Exception as exc:  # noqa: BLE001
        raise VlmGeminiError(f"Création du client Gemini échouée : {exc}") from exc


def _image_part(types, path: str):
    """Construit une Part image pour Gemini (bytes inline). Lève si introuvable/non supporté."""
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise VlmGeminiError(f"Image introuvable : {p}")
    media_type = _MEDIA_TYPES.get(p.suffix.lower())
    if media_type is None:
        raise VlmGeminiError(f"Type d'image non supporté : {p.suffix} ({p})")
    return types.Part.from_bytes(data=p.read_bytes(), mime_type=media_type)


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_markdown_fences(text: str) -> str:
    """Retire d'éventuels blocs Markdown ```json … ``` autour du JSON."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t)
        # Retire une éventuelle fence fermante restée en fin.
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def _parse_json_strict(raw_text: str) -> Dict[str, Any]:
    """Parse strict du JSON renvoyé par Gemini (après nettoyage des fences)."""
    cleaned = _strip_markdown_fences(raw_text)
    if not cleaned:
        raise VlmGeminiError("Réponse Gemini vide.")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Dernier recours : extraire le 1er objet {...} de la réponse.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                raise VlmGeminiError(f"JSON Gemini invalide : {exc}") from exc
        else:
            raise VlmGeminiError(f"JSON Gemini invalide : {exc}") from exc
    if not isinstance(data, dict):
        raise VlmGeminiError(f"JSON Gemini inattendu (type={type(data).__name__}).")
    return data


def _is_transient(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return "503" in s or "unavailable" in s or "overloaded" in s or "429" in s


def analyze_segment(
    prompt_text: str,
    image_paths: List[str],
    segment_id: int,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    combined_text: str = "",
    model: Optional[str] = None,
    timeout_s: float = 300.0,
    *,
    max_retries: int = 2,
    base_delay: float = 2.0,
) -> Dict[str, Any]:
    """Analyse un segment via Gemini et retourne un dict CONFORME au contrat VLM.

    - mêmes entrées que l'ancien runner (prompt texte + images motion/frames) ;
    - response_mime_type JSON + response_schema (si supporté) ;
    - parsing strict (nettoyage des fences) ;
    - aucune réparation silencieuse : la réponse brute est validée par l'appelant ;
    - retry backoff+jitter sur erreurs transitoires (503/UNAVAILABLE/429).
    Lève `VlmGeminiError` si aucune réponse exploitable (l'appelant gère par segment).
    """
    if not image_paths:
        raise VlmGeminiError("Aucune image à analyser pour ce segment.")

    client, types = _init_gemini()
    resolved_model = resolve_vlm_model(model)
    parts = [prompt_text + VLM_PROMPT_DIRECTIVES]
    parts += [_image_part(types, pth) for pth in image_paths]

    # Config : JSON strict + schéma de réponse compatible google-genai (sans const).
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=to_gemini_schema(
                build_output_schema(segment_id, start_s, end_s, combined_text)
            ),
        )
    except Exception:  # noqa: BLE001 - schéma refusé par la version du SDK
        logging.warning("vlm_gemini: response_schema non accepté, repli sur JSON simple.")
        config = types.GenerateContentConfig(response_mime_type="application/json")

    attempt = 0
    while True:
        try:
            response = client.models.generate_content(
                model=resolved_model, contents=parts, config=config
            )
            raw_text = getattr(response, "text", None)
            if not raw_text:
                raise VlmGeminiError("réponse vide de Gemini")
            parsed = _parse_json_strict(raw_text)
            return parsed
        except Exception as exc:  # noqa: BLE001
            if _is_transient(exc) and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                logging.warning(
                    "vlm_gemini: segment %d — erreur transitoire (tentative %d/%d), retry %.1fs : %s",
                    segment_id, attempt + 1, max_retries + 1, delay, exc,
                )
                if delay > 0:
                    _sleep(delay)
                attempt += 1
                continue
            if isinstance(exc, VlmGeminiError):
                raise
            raise VlmGeminiError(f"Analyse VLM Gemini échouée (segment {segment_id}) : {exc}") from exc


class GeminiVLMProvider:
    """Provider VLM visuel Gemini (interface `analyze_segment`)."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    def analyze_segment(
        self,
        prompt_text: str,
        image_paths: List[str],
        segment_id: int,
        start_s: Optional[float] = None,
        end_s: Optional[float] = None,
        combined_text: str = "",
        model: Optional[str] = None,
        timeout_s: float = 300.0,
    ) -> Dict[str, Any]:
        return analyze_segment(
            prompt_text, image_paths, segment_id,
            start_s=start_s, end_s=end_s, combined_text=combined_text,
            model=model or self.model, timeout_s=timeout_s,
        )
