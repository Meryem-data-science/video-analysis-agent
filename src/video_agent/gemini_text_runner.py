"""Génération de texte structuré via **Gemini** (google-genai) — remplace Claude texte.

Sert à l'arc narratif (`narrative_arc.py`) : UN appel texte seul (aucune image),
sortie JSON structurée conforme à un `json_schema` fourni. Indépendant de Claude.

Modèle : `GEMINI_TEXT_MODEL` puis FALLBACK `GEMINI_VLM_MODEL`, sinon défaut. Import
google-genai différé, **aucun print** (logs via `logging`). En cas d'erreur, lève une
exception : l'appelant (narrative_arc) la capture et renvoie un arc vide (jamais de
plantage de pipeline).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, Optional

from .vlm_schema import to_gemini_schema

DEFAULT_TEXT_MODEL = "gemini-2.5-flash"
REQUEST_TIMEOUT_MS = 120_000


class GeminiTextError(Exception):
    """Échec d'une génération texte structurée via Gemini."""


def resolve_text_model(model: Optional[str] = None) -> str:
    """Modèle texte : argument > $GEMINI_TEXT_MODEL > $GEMINI_VLM_MODEL (fallback) > défaut."""
    if model and model.strip():
        return model.strip()
    for env_key in ("GEMINI_TEXT_MODEL", "GEMINI_VLM_MODEL"):
        val = os.environ.get(env_key)
        if val and val.strip():
            return val.strip()
    return DEFAULT_TEXT_MODEL


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t)
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def _init_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiTextError("GEMINI_API_KEY absente — texte Gemini indisponible.")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise GeminiTextError("'google-genai' non installé.") from exc
    try:
        http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        return genai.Client(api_key=api_key, http_options=http_options), types
    except Exception as exc:  # noqa: BLE001
        raise GeminiTextError(f"Création du client Gemini échouée : {exc}") from exc


def analyze_text_with_gemini(
    prompt_text: str,
    schema: Dict[str, Any],
    model: Optional[str] = None,
    timeout_s: float = 180.0,
) -> Dict[str, Any]:
    """Envoie un prompt TEXTE SEUL à Gemini et renvoie un dict JSON structuré.

    `schema` : json_schema attendu (adapté pour google-genai via `to_gemini_schema`).
    Lève `GeminiTextError` si la réponse est absente/non parsable.
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise GeminiTextError("Prompt texte vide.")
    if not isinstance(schema, dict) or not schema:
        raise GeminiTextError("Schéma de sortie invalide.")

    client, types = _init_gemini()
    resolved_model = resolve_text_model(model)
    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=to_gemini_schema(schema),
        )
    except Exception:  # noqa: BLE001
        logging.warning("gemini_text: response_schema non accepté, repli sur JSON simple.")
        config = types.GenerateContentConfig(response_mime_type="application/json")

    try:
        response = client.models.generate_content(
            model=resolved_model, contents=[prompt_text], config=config
        )
        raw = getattr(response, "text", None)
        if not raw:
            raise GeminiTextError("réponse vide de Gemini")
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
    except GeminiTextError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GeminiTextError(f"Texte Gemini non exploitable : {exc}") from exc

    if not isinstance(data, dict):
        raise GeminiTextError(f"JSON texte inattendu (type={type(data).__name__}).")
    return data
