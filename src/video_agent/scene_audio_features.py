"""Features audio MESURÉES par scène — calcul LOCAL et DÉTERMINISTE (librosa).

Miroir structurel de ``scene_audio.py`` mais **100% local** : aucun appel API /
réseau / quota, ni Gemini ni Claude ni VLM. Pour CHAQUE scène : extrait le segment
audio en .wav mono (ffmpeg, plus robuste que mp3 pour librosa) puis mesure des
features rythmiques/énergétiques via librosa.

Mesures (provenance ``"source": "librosa"``) :
- ``tempo_bpm`` (``librosa.beat.beat_track``) UNIQUEMENT si la scène dure ≥ 4 s ;
  sinon ``null`` + ``tempo_reason`` « scène trop courte ». Note honnête : le tempo
  est une ESTIMATION, ambiguïté possible d'un facteur 2 (demi/double tempo).
- ``energy_rms`` (moyenne de ``librosa.feature.rms``).
- ``beat_count`` / ``beat_density_per_s`` (depuis ``beat_track``).
- ``audio_activity`` : ``"actif"`` / ``"quasi_silence"`` via un seuil d'énergie.
  NB : il s'agit d'une activité ÉNERGÉTIQUE, **pas** d'une détection musique-vs-parole.

Robustesse : dégradation gracieuse PAR SCÈNE (audio illisible/vide -> ``audio_status``
``"features_échec"``, on continue) ; import de librosa différé à l'exécution (le module
s'importe sans le SDK) ; **ne lève jamais** vers l'appelant ; aucune invention.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Paramètres par défaut (déterministes).
DEFAULT_SR = 22050                  # ré-échantillonnage mono pour librosa.
DEFAULT_TEMPO_MIN_DURATION_S = 4.0  # tempo non calculé en deçà (trop court = peu fiable).
DEFAULT_ENERGY_ACTIVITY_THRESHOLD = 0.01  # RMS moyen en deçà -> quasi_silence.

_TEMPO_NOTE = (
    "Estimation rythmique automatique (librosa) ; ambiguïté possible d'un facteur 2 "
    "(demi/double tempo)."
)


def _eprint_warn(msg: str) -> None:
    logging.warning(msg)


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
            logging.warning("scene_audio_features: scène ignorée (bornes invalides) : %r", sc)
            continue
        out.append({"scene_index": idx_i, "start_s": start_f, "end_s": end_f})
    out.sort(key=lambda s: s["scene_index"])
    return out


def _extract_scene_wav(video_path: str, start: float, end: float, tmp_dir: str,
                       sr: int = DEFAULT_SR) -> Optional[Path]:
    """Extrait [start, end] en .wav mono PCM16 via ffmpeg. -> Path ou None (échec)."""
    fd, name = tempfile.mkstemp(suffix=".wav", prefix="scene_feat_", dir=tmp_dir)
    os.close(fd)
    out = Path(name)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-i", str(video_path),
        "-vn", "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
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
        _eprint_warn("scene_audio_features: ffmpeg introuvable sur le PATH.")
        out.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        _eprint_warn(f"scene_audio_features: extraction ffmpeg échouée [{start:.3f}, {end:.3f}].")
        out.unlink(missing_ok=True)
        return None
    return out


def _features_from_signal(
    y, sr: int, duration_s: Optional[float], *,
    tempo_min_duration_s: float = DEFAULT_TEMPO_MIN_DURATION_S,
    energy_threshold: float = DEFAULT_ENERGY_ACTIVITY_THRESHOLD,
) -> Dict[str, Any]:
    """Calcule les features depuis un signal numpy déjà chargé. Cœur déterministe.

    Séparé de l'IO fichier pour être testable hors-ligne avec des signaux synthétiques.
    Lève ValueError si le signal est vide (l'appelant marque alors features_échec).
    """
    import numpy as np
    import librosa

    if y is None or len(y) == 0:
        raise ValueError("signal audio vide")

    rms = librosa.feature.rms(y=y)
    energy_rms = round(float(np.mean(rms)), 6)
    activity = "actif" if energy_rms >= energy_threshold else "quasi_silence"

    tempo_bpm: Optional[float] = None
    tempo_reason: Optional[str] = None
    beat_count: Optional[int] = None
    beat_density: Optional[float] = None

    if duration_s is not None and duration_s >= tempo_min_duration_s:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_val = float(np.atleast_1d(tempo)[0]) if tempo is not None else 0.0
        tempo_bpm = round(tempo_val, 2) if tempo_val > 0 else None
        beat_count = int(len(beats))
        if duration_s and duration_s > 0:
            beat_density = round(beat_count / duration_s, 4)
    else:
        tempo_reason = f"scène trop courte (< {tempo_min_duration_s:g} s)"

    return {
        "source": "librosa",
        "audio_activity": activity,
        "energy_rms": energy_rms,
        "tempo_bpm": tempo_bpm,
        "tempo_reason": tempo_reason,
        "tempo_note": _TEMPO_NOTE,
        "beat_count": beat_count,
        "beat_density_per_s": beat_density,
    }


def _compute_features(
    wav_path: Path, duration_s: Optional[float], *,
    tempo_min_duration_s: float = DEFAULT_TEMPO_MIN_DURATION_S,
    energy_threshold: float = DEFAULT_ENERGY_ACTIVITY_THRESHOLD,
    sr: int = DEFAULT_SR,
) -> Dict[str, Any]:
    """Charge le .wav (librosa, import différé) puis délègue à _features_from_signal."""
    import librosa  # import différé : le module s'importe même sans librosa.

    y, sr_loaded = librosa.load(str(wav_path), sr=sr, mono=True)
    return _features_from_signal(
        y, sr_loaded, duration_s,
        tempo_min_duration_s=tempo_min_duration_s,
        energy_threshold=energy_threshold,
    )


def _global_status(ok: int, total: int) -> str:
    """ok / partiel / échec."""
    if total <= 0 or ok == 0:
        return "échec"
    if ok == total:
        return "ok"
    return "partiel"


def build_scene_audio_features(
    video_path: str,
    scenes: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    *,
    sr: int = DEFAULT_SR,
    tempo_min_duration_s: float = DEFAULT_TEMPO_MIN_DURATION_S,
    energy_activity_threshold: float = DEFAULT_ENERGY_ACTIVITY_THRESHOLD,
) -> Dict[str, Any]:
    """Mesure les features audio scène par scène et écrit ``audio_features_<stem>.json``.

    Args :
        video_path : vidéo source.
        scenes : bornes [{index, start, end}] (ou {scene_index, start_s, end_s}).
        output_path : sortie JSON (défaut ``outputs/audio_features_<stem>.json``).
        sr : ré-échantillonnage mono.
        tempo_min_duration_s : durée mini pour calculer le tempo.
        energy_activity_threshold : seuil RMS pour actif/quasi_silence.

    Renvoie ``{status, artifact_path, scenes_ok, scenes_total}``. Ne lève jamais.
    """
    try:
        stem = Path(video_path).stem
        out_path = (
            Path(output_path) if output_path
            else Path("outputs") / f"audio_features_{stem}.json"
        )
        scenes_norm = _normalize_scenes(scenes)

        scenes_out: List[Dict[str, Any]] = []
        ok_count = 0
        with tempfile.TemporaryDirectory(prefix="scene_feat_") as tmp:
            for sc in scenes_norm:
                start, end = sc["start_s"], sc["end_s"]
                duration = round(end - start, 3) if end > start else 0.0
                features: Dict[str, Any] = {}
                status = "features_échec"

                wav = _extract_scene_wav(video_path, start, end, tmp, sr=sr)
                if wav is not None:
                    try:
                        features = _compute_features(
                            wav, duration,
                            tempo_min_duration_s=tempo_min_duration_s,
                            energy_threshold=energy_activity_threshold,
                            sr=sr,
                        )
                        status = "ok"
                        ok_count += 1
                    except Exception as exc:  # noqa: BLE001 — dégradation gracieuse
                        logging.warning(
                            "scene_audio_features: features échouées scène %s : %s",
                            sc["scene_index"], exc,
                        )
                    finally:
                        wav.unlink(missing_ok=True)

                scenes_out.append({
                    "scene_index": sc["scene_index"],
                    "start_s": start,
                    "end_s": end,
                    "duration_s": duration,
                    "features": features,
                    "audio_status": status,
                })

        total = len(scenes_out)
        status = _global_status(ok_count, total)
        artifact = {
            "schema": "audio_features/1.0",
            "video_path": str(video_path),
            "status": status,
            "params": {
                "sr": sr,
                "tempo_min_duration_s": tempo_min_duration_s,
                "energy_activity_threshold": energy_activity_threshold,
            },
            "method_notes": {
                "tempo": _TEMPO_NOTE,
                "audio_activity": (
                    "Activité ÉNERGÉTIQUE (seuil RMS), PAS une détection musique-vs-parole."
                ),
            },
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
        }
    except Exception as exc:  # noqa: BLE001 — garantie « ne lève jamais »
        logging.exception("scene_audio_features: erreur inattendue, échec global : %s", exc)
        return {
            "status": "échec",
            "artifact_path": str(output_path) if output_path else None,
            "scenes_ok": 0,
            "scenes_total": len(scenes or []),
        }
