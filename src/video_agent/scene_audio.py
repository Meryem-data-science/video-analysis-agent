"""Sous-étape audio par scène — analyse de la piste sonore (musique + SFX) via Gemini.

Miroir de ``scene_ocr.py`` côté audio. Pour CHAQUE scène : extrait le segment audio
(ffmpeg ``-ss/-to``, mono), l'envoie à Gemini en sortie JSON structurée (prompt durci),
puis remappe le résultat vers la scène. Anti-hallucination héritée du prompt : confiance
par champ, incertitudes listées, « incertain » plutôt que deviner, jamais de titre/artiste
inventé.

Robustesse : dégradation gracieuse PAR SCÈNE (un échec sur une scène -> cette scène
``audio_status="échec_gracieux"``, on continue) ; **retry avec backoff exponentiel +
jitter** sur les erreurs transitoires (503 / UNAVAILABLE) — les autres erreurs ne sont
pas retentées ; timeout de requête ; import du SDK ``google-genai`` différé à l'exécution
(le module s'importe sans le SDK) ; **ne lève jamais** d'exception vers l'appelant.
N'appelle NI Claude NI le VLM.

Reprise : si ``cache_dir`` est fourni, chaque scène RÉUSSIE est mise en cache dans un
fichier ``scene_<id>.json`` (écriture atomique). Une relance saute les scènes déjà en
cache et ne refait QUE les scènes manquantes. Une scène en échec n'est JAMAIS mise en
cache, donc elle sera ré-essayée au prochain run.
"""
from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Modèle par défaut (gemini-2.5-flash : plus stable et plus généreux sur le tier
# gratuit → moins de 503). Surchargeable par argument explicite, sinon par
# $env:GEMINI_MODEL, sinon ce défaut.
DEFAULT_MODEL = "gemini-2.5-flash"

# Timeout par requête Gemini (en millisecondes). Étape audio NON-BLOQUANTE : un appel
# coincé (réseau/serveur) expire et déclenche le skip de la scène au lieu de figer le run.
REQUEST_TIMEOUT_MS = 45_000

# --- Prompt durci + schéma de sortie (repris à l'identique des prototypes) ----------

PROMPT = """Tu es un analyste audio rigoureux. On te fournit UNIQUEMENT la piste
audio d'une courte vidéo. Décris la musique et les effets sonores présents.

Règles STRICTES :
- Donne une confiance (0.0 à 1.0) pour CHAQUE affirmation.
- Liste explicitement tes doutes dans le champ "uncertainties".
- En cas de doute, dis "incertain" plutôt que de deviner.
- N'INVENTE JAMAIS le titre d'un morceau ni le nom d'un artiste/chanteur.
- Reste factuel et qualitatif : pas de BPM précis inventé (utilise tempo_feel).
- Réponds UNIQUEMENT par un JSON valide conforme au schéma imposé.

Champs attendus :
- music.present : y a-t-il de la musique ?
- music.mood : ambiance/humeur générale de la musique (qualitatif).
- music.energy : "faible", "moyenne" ou "élevée".
- music.tempo_feel : "lent", "modéré" ou "rapide".
- sound_effects : liste d'effets sonores identifiés, chacun avec un label et une confiance.
- speech_overlap : y a-t-il de la parole qui se superpose au reste ?
- evidence : ce que tu entends concrètement et qui justifie tes réponses.
- uncertainties : la liste de tes doutes.
"""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "music": {
            "type": "object",
            "properties": {
                "present": {"type": "boolean"},
                "confidence": {"type": "number"},
                "mood": {"type": "string"},
                "mood_confidence": {"type": "number"},
                "energy": {"type": "string", "enum": ["faible", "moyenne", "élevée"]},
                "tempo_feel": {"type": "string", "enum": ["lent", "modéré", "rapide"]},
            },
            "required": [
                "present",
                "confidence",
                "mood",
                "mood_confidence",
                "energy",
                "tempo_feel",
            ],
        },
        "sound_effects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "confidence"],
            },
        },
        "speech_overlap": {"type": "boolean"},
        "evidence": {"type": "string"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "music",
        "sound_effects",
        "speech_overlap",
        "evidence",
        "uncertainties",
    ],
}


# --- Helpers (seams mockables dans les tests) ---------------------------------------


def _resolve_model(model: Optional[str]) -> str:
    """Argument explicite > $env:GEMINI_MODEL > défaut (gemini-2.5-flash)."""
    return model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL


def _normalize_scenes(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accepte {index|scene_index, start|start_s, end|end_s} -> liste normalisée triée."""
    out: List[Dict[str, Any]] = []
    for i, sc in enumerate(scenes or [], start=1):
        if not isinstance(sc, dict):
            continue
        idx = sc.get("index", sc.get("scene_index", i))
        start = sc.get("start", sc.get("start_s"))
        end = sc.get("end", sc.get("end_s"))
        try:
            idx_i = int(idx)
            start_f = float(start)
            end_f = float(end)
        except (TypeError, ValueError):
            logging.warning("scene_audio: scène ignorée (bornes invalides) : %r", sc)
            continue
        out.append({"scene_index": idx_i, "start_s": start_f, "end_s": end_f})
    out.sort(key=lambda s: s["scene_index"])
    return out


def _init_gemini():
    """Import différé + création client. Renvoie (client, types) ou (None, None).

    Aucune clé / SDK absent -> (None, None) (l'appelant dégrade gracieusement).
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logging.warning("scene_audio: GEMINI_API_KEY absente -> aucune analyse possible.")
        return None, None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logging.warning("scene_audio: 'google-genai' non installé -> aucune analyse possible.")
        return None, None
    try:
        http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        return genai.Client(api_key=api_key, http_options=http_options), types
    except Exception as exc:  # noqa: BLE001
        logging.warning("scene_audio: création client Gemini échouée : %s", exc)
        return None, None


def _extract_scene_audio(video_path: str, start: float, end: float, tmp_dir: str) -> Optional[Path]:
    """Extrait [start, end] en .mp3 mono via ffmpeg. -> Path ou None (échec)."""
    fd, name = tempfile.mkstemp(suffix=".mp3", prefix="scene_audio_", dir=tmp_dir)
    os.close(fd)
    out = Path(name)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-vn", "-ac", "1",
        "-c:a", "libmp3lame", "-q:a", "4",
        str(out),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
        )
    except FileNotFoundError:
        logging.warning("scene_audio: ffmpeg introuvable sur le PATH.")
        out.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        logging.warning("scene_audio: extraction ffmpeg échouée [%.3f, %.3f].", start, end)
        out.unlink(missing_ok=True)
        return None
    return out


def _analyze_scene_call(client, types, audio_path: Path, model: str) -> Dict[str, Any]:
    """UN essai Gemini : upload, generate_content, delete, parse. -> dict (ou lève)."""
    uploaded = client.files.upload(file=str(audio_path))
    try:
        response = client.models.generate_content(
            model=model,
            contents=[PROMPT, uploaded],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RESPONSE_SCHEMA,
            ),
        )
        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError("réponse vide de Gemini")
        return json.loads(raw)
    finally:
        name = getattr(uploaded, "name", None)
        if name:
            try:
                client.files.delete(name=name)
            except Exception as exc:  # noqa: BLE001
                logging.warning("scene_audio: suppression distante échouée : %s", exc)


# Marqueur de sommeil isolé pour rester testable (les tests passent retry_pause_s=0,
# donc aucun sleep réel ; en prod, backoff exponentiel + jitter borné par base_delay).
_sleep = time.sleep


def _is_transient_error(exc: Exception) -> bool:
    """True pour les erreurs transitoires (serveur surchargé) qui méritent un retry."""
    s = f"{type(exc).__name__} {exc}".lower()
    return "503" in s or "unavailable" in s or "overloaded" in s


def _analyze_scene_with_retry(
    client, types, audio_path: Path, model: str, scene_index: int,
    *, max_retries: int, base_delay: float,
) -> Optional[Dict[str, Any]]:
    """Analyse Gemini avec retry + backoff exponentiel + jitter sur 503/UNAVAILABLE.

    ~``max_retries + 1`` tentatives au total. Délai avant le retry n° i :
    ``base_delay * 2**i + uniform(0, base_delay)``. Les erreurs NON transitoires
    (parse, réponse vide, etc.) ne sont pas retentées. Renvoie None après épuisement
    (l'appelant dégrade gracieusement). Ne lève jamais.
    """
    attempt = 0
    while True:
        try:
            return _analyze_scene_call(client, types, audio_path, model)
        except Exception as exc:  # noqa: BLE001 — étape non-bloquante
            transient = _is_transient_error(exc)
            if transient and attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                logging.warning(
                    "scene_audio: scène %d — erreur transitoire (tentative %d/%d), "
                    "nouvel essai dans %.1fs : %s",
                    scene_index, attempt + 1, max_retries + 1, delay, exc,
                )
                if delay > 0:
                    _sleep(delay)
                attempt += 1
                continue
            logging.warning(
                "scene_audio: scène %d — audio non analysé (%s, %d tentative(s)) : %s",
                scene_index, "transitoire" if transient else "non-retentable",
                attempt + 1, exc,
            )
            return None


def _scene_cache_path(cache_dir: Optional[Path], scene_index: int) -> Optional[Path]:
    """Chemin du fichier de cache d'une scène (ou None si pas de cache_dir)."""
    return None if cache_dir is None else cache_dir / f"scene_{scene_index:03d}.json"


def _load_scene_cache(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Charge un résultat de scène en cache (dict) ou None (absent/illisible)."""
    if path is None or not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_scene_cache_atomic(path: Path, record: Dict[str, Any]) -> None:
    """Écrit le résultat d'une scène RÉUSSIE de façon atomique (.tmp puis os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _global_status(ok: int, total: int) -> str:
    """ok / partiel / échec."""
    if total <= 0 or ok == 0:
        return "échec"
    if ok == total:
        return "ok"
    return "partiel"


# --- Fonction principale ------------------------------------------------------------


def build_scene_audio(
    video_path: str,
    scenes: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    model: Optional[str] = None,
    *,
    max_retries: int = 3,
    pause_s: float = 2.5,
    retry_pause_s: float = 2.0,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyse l'audio scène par scène et écrit ``audio_events_<stem>.json``.

    Args :
        video_path : vidéo source.
        scenes : bornes [{index, start, end}] (ou {scene_index, start_s, end_s}).
        output_path : sortie JSON (défaut ``outputs/audio_events_<stem>.json``).
        model : modèle Gemini (défaut $GEMINI_MODEL puis gemini-2.5-flash).
        max_retries : nombre de retries sur erreur transitoire 503/UNAVAILABLE
            (≈ max_retries + 1 tentatives au total).
        pause_s : pause entre appels réseau (débit) ; sautée pour une scène servie par le cache.
        retry_pause_s : délai de BASE du backoff exponentiel (s) ; 0 = pas de sleep.
        cache_dir : dossier de cache PAR SCÈNE. Si fourni, les scènes déjà réussies
            (``scene_<id>.json``) sont rechargées sans appel réseau ; une scène en échec
            n'est jamais mise en cache (donc ré-essayée au run suivant).

    Renvoie un dict ``{status, artifact_path, scenes_ok, scenes_total, model_used}``.
    Ne lève jamais : tout échec se traduit en statut/artefact dégradé.
    """
    try:
        resolved_model = _resolve_model(model)
        stem = Path(video_path).stem
        out_path = Path(output_path) if output_path else Path("outputs") / f"audio_events_{stem}.json"
        cache_path = Path(cache_dir) if cache_dir else None
        if cache_path is not None:
            cache_path.mkdir(parents=True, exist_ok=True)

        scenes_norm = _normalize_scenes(scenes)
        client = types = None  # client Gemini créé paresseusement (inutile si tout est en cache)

        scenes_out: List[Dict[str, Any]] = []
        ok_count = 0
        with tempfile.TemporaryDirectory(prefix="scene_audio_") as tmp:
            for i, sc in enumerate(scenes_norm):
                scene_id = sc["scene_index"]

                # 1) Cache par scène : déjà réussie -> on recharge, aucun appel réseau.
                scene_file = _scene_cache_path(cache_path, scene_id)
                cached = _load_scene_cache(scene_file)
                if cached is not None:
                    scenes_out.append(cached)
                    if cached.get("audio_status") == "ok":
                        ok_count += 1
                    logging.info("scene_audio: scène %d servie par le cache (skip).", scene_id)
                    continue

                # 2) Création paresseuse du client Gemini au 1er besoin réel.
                if client is None and types is None:
                    client, types = _init_gemini()

                events: Dict[str, Any] = {}
                status = "échec_gracieux"
                if client is not None:
                    audio_path = _extract_scene_audio(
                        video_path, sc["start_s"], sc["end_s"], tmp
                    )
                    if audio_path is not None:
                        parsed = _analyze_scene_with_retry(
                            client, types, audio_path, resolved_model, scene_id,
                            max_retries=max_retries, base_delay=retry_pause_s,
                        )
                        audio_path.unlink(missing_ok=True)
                        if parsed is not None:
                            events, status = parsed, "ok"
                            ok_count += 1

                scene_rec = {
                    "scene_index": scene_id,
                    "start_s": sc["start_s"],
                    "end_s": sc["end_s"],
                    "audio_events": events,
                    "audio_status": status,
                }
                scenes_out.append(scene_rec)

                # 3) Mise en cache UNIQUEMENT en cas de succès (jamais une scène en échec).
                if scene_file is not None and status == "ok":
                    _write_scene_cache_atomic(scene_file, scene_rec)

                if pause_s > 0 and i < len(scenes_norm) - 1:
                    time.sleep(pause_s)

        total = len(scenes_out)
        status = _global_status(ok_count, total)
        artifact = {
            "schema": "scene_audio/1.0",
            "video_path": str(video_path),
            "model_used": resolved_model,
            "status": status,
            "scenes": scenes_out,
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        return {
            "status": status,
            "artifact_path": str(out_path),
            "scenes_ok": ok_count,
            "scenes_total": total,
            "model_used": resolved_model,
        }
    except Exception as exc:  # noqa: BLE001 — garantie « ne lève jamais »
        logging.exception("scene_audio: erreur inattendue, échec global : %s", exc)
        return {
            "status": "échec",
            "artifact_path": str(output_path) if output_path else None,
            "scenes_ok": 0,
            "scenes_total": len(scenes or []),
            "model_used": _resolve_model(model),
        }
