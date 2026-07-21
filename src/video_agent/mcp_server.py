import json
import importlib
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# Ensure the package parent is importable when running this module directly.
PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = (
    _SOURCE_ROOT
    if (_SOURCE_ROOT / "src" / "video_agent").is_dir()
    else Path.cwd().resolve()
)
# Alias historique utilisé par les validateurs de chemins internes.
ROOT_DIR = PROJECT_ROOT
_source_scripts = PROJECT_ROOT / "scripts"
SCRIPTS = (
    _source_scripts
    if _source_scripts.is_dir()
    else Path(importlib.import_module("scripts").__file__).resolve().parent
)

_env_video_root = os.getenv("VIDEO_ROOT")
if _env_video_root:
    candidate = Path(_env_video_root)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    VIDEO_ROOT = candidate.resolve()
elif (PROJECT_ROOT / "vidéos").exists():
    VIDEO_ROOT = (PROJECT_ROOT / "vidéos").resolve()
elif (PROJECT_ROOT / "videos").exists():
    VIDEO_ROOT = (PROJECT_ROOT / "videos").resolve()
else:
    VIDEO_ROOT = (PROJECT_ROOT / "vidéos").resolve()

from fastmcp.server import FastMCP

from video_agent.inspection import InspectError, inspect_video
from video_agent.segmentation import load_strategy, segment_video
from video_agent.strategy import decide_strategy

app = FastMCP(
    name="Agent Analyse Vidéo MCP",
    instructions=(
        "Expose inspect_video_file pour l'inspection technique des vidéos via FFprobe, "
        "decide_video_strategy pour produire une stratégie, segment_video_file pour détecter les scènes visuelles, "
        "extract_video_frames pour extraire des frames représentatives via FFmpeg, "
        "build_motion_representation pour produire grille temporelle et composite de mouvement via Pillow, "
        "extract_audio_track pour extraire la piste audio en WAV mono 16 kHz via FFmpeg, "
        "analyze_audio_light_file pour une analyse audio légère (énergie/silence) avant transcription, "
        "transcribe_audio_file pour transcrire l'audio en segments horodatés via faster-whisper, "
        "fuse_audio_visual_file pour fusionner visuel et audio en un manifest aligné par timestamps, "
        "build_vlm_pack_file pour préparer le pack d'entrée du VLM par segment, "
        "prepare_vlm_run_file pour générer un dossier de run VLM (prompts + plan) sans appeler de modèle, "
        "validate_vlm_results_file pour valider les réponses JSON du VLM vis-à-vis du pack, "
        "run_vlm_agent_analysis pour analyser automatiquement les segments via le VLM Gemini (google-genai), "
        "build_final_report_file pour produire le rapport vidéo final (synthèse globale), "
        "export_report_pdf_file pour exporter ce rapport en PDF (arabe rendu correctement), "
        "et run_full_pipeline_file pour exécuter tout le pipeline depuis le nom d'une vidéo."
    ),
    version="0.1",
)


def _is_within_project_root(path: Path) -> bool:
    try:
        path.relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _resolve_video_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] in {"vidéos", "videos"}:
            candidate = PROJECT_ROOT / candidate
        else:
            candidate = VIDEO_ROOT / candidate
    candidate = candidate.resolve()

    if not _is_within_project_root(candidate):
        raise InspectError(
            f"Accès refusé : la vidéo doit rester dans le projet ({PROJECT_ROOT})"
        )

    try:
        candidate.relative_to(VIDEO_ROOT)
    except ValueError:
        raise InspectError(
            f"Accès refusé : la vidéo doit être sous VIDEO_ROOT ({VIDEO_ROOT})"
        )

    if not candidate.exists() or not candidate.is_file():
        raise InspectError(f"Fichier vidéo introuvable : {candidate}")

    return candidate


def _resolve_strategy_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    if not _is_within_project_root(candidate):
        raise InspectError(
            f"Accès refusé : le fichier de stratégie doit être dans le projet ({PROJECT_ROOT})"
        )

    if not candidate.exists() or not candidate.is_file():
        raise InspectError(f"Fichier de stratégie introuvable : {candidate}")

    return candidate


def _resolve_inspection_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(ROOT_DIR)
    except ValueError:
        raise InspectError(
            f"Accès refusé : le fichier d'inspection doit être dans le projet ({ROOT_DIR})"
        )

    if not candidate.exists() or not candidate.is_file():
        raise InspectError(f"Fichier d'inspection introuvable : {candidate}")

    return candidate


def _load_inspection(inspection: Any) -> dict:
    if isinstance(inspection, dict):
        return inspection
    if isinstance(inspection, str):
        inspection_path = _resolve_inspection_path(inspection)
        try:
            return json.loads(inspection_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise InspectError(f"Impossible de lire le JSON d'inspection : {exc}") from exc
    raise InspectError("L'argument doit être un dictionnaire d'inspection ou un chemin vers un fichier JSON.")


@app.tool(
    name="inspect_video_file",
    description="Inspecte une vidéo technique et retourne les métadonnées normalisées.",
)
def inspect_video_file(path: str) -> dict:
    """Inspecte une vidéo depuis un chemin dans VIDEO_ROOT."""
    resolved = _resolve_video_path(path)
    return inspect_video(resolved)


@app.tool(
    name="decide_video_strategy",
    description=(
        "Décide une stratégie vidéo à partir d'un JSON d'inspection ou d'un chemin vers un fichier d'inspection."
    ),
)
def decide_video_strategy(inspection: Any) -> dict:
    """Retourne le JSON de stratégie pour une inspection vidéo donnée."""
    inspection_data = _load_inspection(inspection)
    return decide_strategy(inspection_data)

def _run_segmentation_subprocess(
    video_path: Path, strategy_file: Path | None, timeout: float = 180.0
) -> dict:
    """Exécute la segmentation dans un sous-processus isolé et retourne son JSON.

    Pourquoi un sous-processus plutôt qu'un appel in-process ?
    Sous le serveur MCP stdio (Windows), le thread de lecture stdin monopolise le
    GIL et affame les autres threads Python du worker : aussi bien le thread de
    décodage interne de PySceneDetect que les threads lecteurs de pipe utilisés par
    ``subprocess`` avec ``capture_output``. La détection ne progresse alors jamais
    (blocage indéfini observé).

    On isole donc le travail dans un PROCESSUS séparé (GIL distinct) dont stdout et
    stderr sont redirigés vers des fichiers : le parent ne crée aucun thread lecteur
    et se contente d'une attente noyau bornée par ``timeout`` (qui libère le GIL).
    """
    script = SCRIPTS / "segment_video.py"
    fd, out_path = tempfile.mkstemp(suffix=".segments.json")
    os.close(fd)
    err_path = out_path + ".err"
    cmd = [sys.executable, str(script), str(video_path), "--output", out_path]
    if strategy_file is not None:
        cmd += ["--strategy", str(strategy_file)]
    logging.info("segment_video_file: lancement sous-processus %s", cmd)
    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Segmentation échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(Path(out_path).read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Segmentation interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        for tmp in (out_path, err_path):
            try:
                os.remove(tmp)
            except OSError:
                pass


@app.tool(
    name="segment_video_file",
    description=(
        "Segmenter une vidéo en scènes visuelles à l'aide de PySceneDetect et d'une stratégie vidéo facultative."
    ),
    run_in_thread=True,
)
def segment_video_file(path: str, strategy_path: str | None = "outputs/strategy_demo.json") -> dict:
    """Retourne les segments détectés pour une vidéo située sous VIDEO_ROOT."""
    logging.info("segment_video_file called path=%s strategy_path=%s", path, strategy_path)
    resolved = _resolve_video_path(path)
    logging.info("resolved video_path=%s", resolved)

    strategy_file: Path | None = None
    tmp_strategy: str | None = None
    if strategy_path is not None:
        if isinstance(strategy_path, dict):
            # Stratégie fournie en ligne : on la matérialise pour le sous-processus.
            fd, tmp_strategy = tempfile.mkstemp(suffix=".strategy.json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(strategy_path, f, ensure_ascii=False)
            strategy_file = Path(tmp_strategy)
        elif isinstance(strategy_path, str):
            strategy_file = _resolve_strategy_path(strategy_path)
            logging.info("resolved strategy_path=%s", strategy_file)
        else:
            raise InspectError(
                "Le paramètre strategy_path doit être un dictionnaire JSON ou un chemin de fichier JSON."
            )

    try:
        result = _run_segmentation_subprocess(resolved, strategy_file)
        logging.info(
            "segment_video_file completed segments=%s", len(result.get("segments", []))
        )
        return result
    except Exception:
        logging.exception("segment_video_file failed")
        raise
    finally:
        if tmp_strategy is not None:
            try:
                os.remove(tmp_strategy)
            except OSError:
                pass


def _resolve_existing_project_file(path: str, label: str) -> Path:
    """Résout un chemin de fichier devant exister et rester dans le projet."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not _is_within_project_root(candidate):
        raise InspectError(
            f"Accès refusé : {label} doit être dans le projet ({PROJECT_ROOT})"
        )
    if not candidate.exists() or not candidate.is_file():
        raise InspectError(f"{label} introuvable : {candidate}")
    return candidate


def _resolve_output_path(path: str, label: str) -> Path:
    """Résout un chemin de sortie (peut ne pas encore exister) restant dans le projet."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not _is_within_project_root(candidate):
        raise InspectError(
            f"Accès refusé : {label} doit être dans le projet ({PROJECT_ROOT})"
        )
    return candidate


def _run_extraction_subprocess(
    video_path: Path,
    segments_file: Path,
    strategy_file: Path,
    output_dir: str,
    manifest_out: Path,
    overwrite: bool,
    max_width: int | None,
    max_height: int | None,
    timeout: float = 300.0,
) -> dict:
    """Exécute l'extraction de frames dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que ``_run_segmentation_subprocess`` : on délègue le
    travail lourd (FFmpeg) à ``scripts/extract_frames.py`` dans un PROCESSUS séparé
    dont stdout/stderr vont vers des fichiers — le parent ne crée aucun thread
    lecteur et ne fait qu'une attente noyau bornée par ``timeout``. Cela évite le
    blocage GIL/stdio observé sous le serveur MCP stdio sur Windows.
    """
    script = SCRIPTS / "extract_frames.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".extract.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        str(video_path),
        "--segments", str(segments_file),
        "--strategy", str(strategy_file),
        "--output-dir", output_dir,
        "--manifest", str(manifest_out),
    ]
    if not overwrite:
        cmd.append("--no-overwrite")
    if max_width is not None:
        cmd += ["--max-width", str(max_width)]
    if max_height is not None:
        cmd += ["--max-height", str(max_height)]
    logging.info("extract_video_frames: lancement sous-processus %s", cmd)
    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Extraction des frames échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(manifest_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Extraction des frames interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="extract_video_frames",
    description=(
        "Extraire des frames représentatives d'une vidéo via FFmpeg, segment par segment, "
        "selon la stratégie (frame_strategy.mode). Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def extract_video_frames(
    path: str,
    segments_path: str = "outputs/segments_demo.json",
    strategy_path: str = "outputs/strategy_demo.json",
    output_dir: str = "outputs/frames",
    manifest_path: str | None = None,
    overwrite: bool = True,
    max_width: int | None = None,
    max_height: int | None = None,
) -> dict:
    """Retourne le manifest d'extraction de frames (dict JSON sérialisable)."""
    logging.info(
        "extract_video_frames called path=%s segments=%s strategy=%s output_dir=%s",
        path, segments_path, strategy_path, output_dir,
    )
    resolved_video = _resolve_video_path(path)
    segments_file = _resolve_existing_project_file(segments_path, "Le fichier de segments")
    strategy_file = _resolve_existing_project_file(strategy_path, "Le fichier de stratégie")

    # Le répertoire de sortie peut ne pas encore exister : on le valide puis on le crée.
    output_resolved = _resolve_output_path(output_dir, "Le répertoire de sortie")
    output_resolved.mkdir(parents=True, exist_ok=True)

    tmp_manifest: str | None = None
    if manifest_path is not None:
        manifest_out = _resolve_output_path(manifest_path, "Le manifest")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_manifest = tempfile.mkstemp(suffix=".frames_manifest.json")
        os.close(fd)
        manifest_out = Path(tmp_manifest)

    try:
        result = _run_extraction_subprocess(
            resolved_video,
            segments_file,
            strategy_file,
            output_dir,
            manifest_out,
            overwrite,
            max_width,
            max_height,
        )
        logging.info(
            "extract_video_frames completed frames=%s",
            result.get("summary", {}).get("frame_count"),
        )
        return result
    except Exception:
        logging.exception("extract_video_frames failed")
        raise
    finally:
        if tmp_manifest is not None:
            try:
                os.remove(tmp_manifest)
            except OSError:
                pass


def _run_motion_subprocess(
    frames_manifest_file: Path,
    output_dir: str,
    manifest_out: Path,
    thumbnail_width: int,
    composite_long_side: int,
    overwrite: bool,
    timeout: float = 300.0,
) -> dict:
    """Exécute la construction des représentations de mouvement dans un sous-processus isolé.

    Même pattern robuste que segmentation/extraction : on délègue le travail lourd
    (Pillow) à ``scripts/build_motion.py`` dans un PROCESSUS séparé dont stdout/stderr
    vont vers des fichiers — le parent ne crée aucun thread lecteur et ne fait qu'une
    attente noyau bornée par ``timeout`` (évite le blocage GIL/stdio sous Windows).
    """
    script = SCRIPTS / "build_motion.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".motion.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--frames-manifest", str(frames_manifest_file),
        "--output-dir", output_dir,
        "--manifest", str(manifest_out),
        "--thumbnail-width", str(thumbnail_width),
        "--composite-long-side", str(composite_long_side),
    ]
    if not overwrite:
        cmd.append("--no-overwrite")
    logging.info("build_motion_representation: lancement sous-processus %s", cmd)
    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Construction du mouvement échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(manifest_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Construction du mouvement interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="build_motion_representation",
    description=(
        "Construire les représentations de mouvement (grille temporelle + composite transparent) "
        "via Pillow, à partir d'un manifest de frames. Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def build_motion_representation(
    frames_manifest_path: str = "outputs/frames_manifest_demo.json",
    output_dir: str = "outputs/motion",
    manifest_path: str | None = None,
    thumbnail_width: int = 360,
    composite_long_side: int = 1080,
    overwrite: bool = True,
) -> dict:
    """Retourne le manifest des représentations de mouvement (dict JSON sérialisable)."""
    logging.info(
        "build_motion_representation called frames_manifest=%s output_dir=%s",
        frames_manifest_path, output_dir,
    )
    frames_manifest_file = _resolve_existing_project_file(
        frames_manifest_path, "Le manifest de frames"
    )

    # Le répertoire de sortie peut ne pas encore exister : on le valide puis on le crée.
    output_resolved = _resolve_output_path(output_dir, "Le répertoire de sortie")
    output_resolved.mkdir(parents=True, exist_ok=True)

    tmp_manifest: str | None = None
    if manifest_path is not None:
        manifest_out = _resolve_output_path(manifest_path, "Le manifest")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_manifest = tempfile.mkstemp(suffix=".motion_manifest.json")
        os.close(fd)
        manifest_out = Path(tmp_manifest)

    try:
        result = _run_motion_subprocess(
            frames_manifest_file,
            output_dir,
            manifest_out,
            thumbnail_width,
            composite_long_side,
            overwrite,
        )
        logging.info(
            "build_motion_representation completed images=%s",
            result.get("summary", {}).get("image_count"),
        )
        return result
    except Exception:
        logging.exception("build_motion_representation failed")
        raise
    finally:
        if tmp_manifest is not None:
            try:
                os.remove(tmp_manifest)
            except OSError:
                pass


def _run_audio_subprocess(
    video_path: Path,
    output_dir: str,
    manifest_out: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
    timeout: float = 300.0,
) -> dict:
    """Exécute l'extraction audio dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que segmentation/extraction/motion : on délègue le travail
    (FFmpeg) à ``scripts/extract_audio.py`` dans un PROCESSUS séparé dont stdout/stderr
    vont vers des fichiers — le parent ne crée aucun thread lecteur et ne fait qu'une
    attente noyau bornée par ``timeout`` (évite le blocage GIL/stdio sous Windows).
    """
    script = SCRIPTS / "extract_audio.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".audio.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        str(video_path),
        "--output-dir", output_dir,
        "--manifest", str(manifest_out),
        "--sample-rate", str(sample_rate),
        "--channels", str(channels),
    ]
    if not overwrite:
        cmd.append("--no-overwrite")
    logging.info("extract_audio_track: lancement sous-processus %s", cmd)
    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Extraction audio échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(manifest_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Extraction audio interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="extract_audio_track",
    description=(
        "Extraire la piste audio d'une vidéo en WAV mono 16 kHz PCM 16-bit via FFmpeg "
        "(format adapté à Whisper). Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def extract_audio_track(
    path: str,
    output_dir: str = "outputs/audio",
    manifest_path: str | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    overwrite: bool = True,
) -> dict:
    """Retourne le manifest d'extraction audio (dict JSON sérialisable)."""
    logging.info(
        "extract_audio_track called path=%s output_dir=%s sr=%d ac=%d",
        path, output_dir, sample_rate, channels,
    )
    resolved_video = _resolve_video_path(path)

    # Le répertoire de sortie peut ne pas encore exister : on le valide puis on le crée.
    output_resolved = _resolve_output_path(output_dir, "Le répertoire de sortie")
    output_resolved.mkdir(parents=True, exist_ok=True)

    tmp_manifest: str | None = None
    if manifest_path is not None:
        manifest_out = _resolve_output_path(manifest_path, "Le manifest")
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_manifest = tempfile.mkstemp(suffix=".audio_manifest.json")
        os.close(fd)
        manifest_out = Path(tmp_manifest)

    try:
        result = _run_audio_subprocess(
            resolved_video,
            output_dir,
            manifest_out,
            sample_rate,
            channels,
            overwrite,
        )
        logging.info(
            "extract_audio_track completed audio_path=%s", result.get("audio_path")
        )
        return result
    except Exception:
        logging.exception("extract_audio_track failed")
        raise
    finally:
        if tmp_manifest is not None:
            try:
                os.remove(tmp_manifest)
            except OSError:
                pass


def _run_audio_analysis_subprocess(
    audio_file: Path,
    audio_manifest_file: Path | None,
    output_out: Path,
    window_ms: int,
    timeout: float = 180.0,
) -> dict:
    """Exécute l'analyse audio légère dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que les autres tools : délégation à
    ``scripts/analyze_audio_light.py`` dans un PROCESSUS séparé dont stdout/stderr
    vont vers des fichiers — attente noyau bornée par ``timeout`` (évite le blocage
    GIL/stdio sous Windows). On lit le résultat depuis le fichier ``--output``.
    """
    script = SCRIPTS / "analyze_audio_light.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".audio_light.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        str(audio_file),
        "--output", str(output_out),
        "--window-ms", str(window_ms),
    ]
    if audio_manifest_file is not None:
        cmd += ["--audio-manifest", str(audio_manifest_file)]
    logging.info("analyze_audio_light_file: lancement sous-processus %s", cmd)
    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Analyse audio légère échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Analyse audio légère interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="analyze_audio_light_file",
    description=(
        "Analyse audio légère (énergie RMS / silence) d'un WAV PCM pour estimer s'il "
        "faut lancer Whisper. Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def analyze_audio_light_file(
    audio_path: str,
    audio_manifest_path: str | None = None,
    output_path: str | None = None,
    window_ms: int = 500,
) -> dict:
    """Retourne le manifest d'analyse audio légère (dict JSON sérialisable)."""
    logging.info(
        "analyze_audio_light_file called audio_path=%s window_ms=%d", audio_path, window_ms
    )
    audio_file = _resolve_existing_project_file(audio_path, "Le fichier audio")

    audio_manifest_file: Path | None = None
    if audio_manifest_path is not None:
        audio_manifest_file = _resolve_existing_project_file(
            audio_manifest_path, "Le manifest audio"
        )

    tmp_output: str | None = None
    if output_path is not None:
        output_out = _resolve_output_path(output_path, "Le fichier de sortie")
        output_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_output = tempfile.mkstemp(suffix=".audio_light.json")
        os.close(fd)
        output_out = Path(tmp_output)

    try:
        result = _run_audio_analysis_subprocess(
            audio_file, audio_manifest_file, output_out, window_ms
        )
        logging.info(
            "analyze_audio_light_file completed mode=%s",
            result.get("classification", {}).get("mode"),
        )
        return result
    except Exception:
        logging.exception("analyze_audio_light_file failed")
        raise
    finally:
        if tmp_output is not None:
            try:
                os.remove(tmp_output)
            except OSError:
                pass


def _run_transcription_subprocess(
    audio_file: Path,
    audio_manifest_file: Path | None,
    audio_light_file: Path | None,
    output_out: Path,
    model: str,
    device: str,
    compute_type: str,
    language: str | None,
    task: str,
    beam_size: int,
    timeout: float = 1800.0,
) -> dict:
    """Exécute la transcription dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que les autres tools, avec un ``timeout`` **généreux**
    (la transcription faster-whisper peut être longue, surtout au premier appel
    qui télécharge le modèle). On délègue à ``scripts/transcribe_audio.py`` dans un
    PROCESSUS séparé (stdout/stderr vers fichiers, attente noyau) pour éviter le
    blocage GIL/stdio sous Windows.
    """
    script = SCRIPTS / "transcribe_audio.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".transcribe.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        str(audio_file),
        "--output", str(output_out),
        "--model", str(model),
        "--device", str(device),
        "--compute-type", str(compute_type),
        "--task", str(task),
        "--beam-size", str(beam_size),
    ]
    if audio_manifest_file is not None:
        cmd += ["--audio-manifest", str(audio_manifest_file)]
    if audio_light_file is not None:
        cmd += ["--audio-light-analysis", str(audio_light_file)]
    if language is not None:
        cmd += ["--language", str(language)]
    logging.info("transcribe_audio_file: lancement sous-processus %s", cmd)

    # Forcer l'UTF-8 dans le sous-processus : le transcript contient du texte non
    # latin (ex. arabe). Sans cela, l'encodage console par défaut (cp1252 sous
    # Windows) ferait planter l'affichage stdout du CLI enfant.
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Transcription échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Transcription interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="transcribe_audio_file",
    description=(
        "Transcrire un audio WAV avec faster-whisper (timestamps par segment, pas de "
        "traduction). Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def transcribe_audio_file(
    audio_path: str,
    audio_manifest_path: str | None = None,
    audio_light_analysis_path: str | None = None,
    output_path: str | None = None,
    model: str = "small",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    task: str = "transcribe",
    beam_size: int = 5,
) -> dict:
    """Retourne le manifest de transcription (dict JSON sérialisable)."""
    logging.info(
        "transcribe_audio_file called audio_path=%s model=%s task=%s", audio_path, model, task
    )
    audio_file = _resolve_existing_project_file(audio_path, "Le fichier audio")

    audio_manifest_file: Path | None = None
    if audio_manifest_path is not None:
        audio_manifest_file = _resolve_existing_project_file(
            audio_manifest_path, "Le manifest audio"
        )
    audio_light_file: Path | None = None
    if audio_light_analysis_path is not None:
        audio_light_file = _resolve_existing_project_file(
            audio_light_analysis_path, "L'analyse audio légère"
        )

    tmp_output: str | None = None
    if output_path is not None:
        output_out = _resolve_output_path(output_path, "Le fichier de sortie")
        output_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_output = tempfile.mkstemp(suffix=".transcript.json")
        os.close(fd)
        output_out = Path(tmp_output)

    try:
        result = _run_transcription_subprocess(
            audio_file,
            audio_manifest_file,
            audio_light_file,
            output_out,
            model,
            device,
            compute_type,
            language,
            task,
            beam_size,
        )
        logging.info(
            "transcribe_audio_file completed segments=%s lang=%s",
            len(result.get("segments", [])),
            result.get("summary", {}).get("detected_language"),
        )
        return result
    except Exception:
        logging.exception("transcribe_audio_file failed")
        raise
    finally:
        if tmp_output is not None:
            try:
                os.remove(tmp_output)
            except OSError:
                pass


def _run_fusion_subprocess(
    segments_file: Path,
    frames_file: Path,
    motion_file: Path,
    audio_light_file: Path | None,
    transcript_file: Path | None,
    output_out: Path,
    timeout: float = 180.0,
) -> dict:
    """Exécute la fusion audio-visuelle dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que les autres tools. Comme la fusion contient du texte non
    latin (arabe/Darija), on force l'UTF-8 du sous-processus (PYTHONUTF8 /
    PYTHONIOENCODING) et on lit le stderr avec errors="replace".
    """
    script = SCRIPTS / "fuse_audio_visual.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".fusion.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--segments", str(segments_file),
        "--frames-manifest", str(frames_file),
        "--motion-manifest", str(motion_file),
        "--output", str(output_out),
    ]
    if audio_light_file is not None:
        cmd += ["--audio-light", str(audio_light_file)]
    if transcript_file is not None:
        cmd += ["--transcript", str(transcript_file)]
    logging.info("fuse_audio_visual_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Fusion audio-visuelle échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Fusion audio-visuelle interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="fuse_audio_visual_file",
    description=(
        "Fusionner les artefacts visuels (segments/frames/motion) et audio "
        "(analyse légère/transcript) en un manifest aligné par timestamps, prêt pour le VLM."
    ),
    run_in_thread=True,
)
def fuse_audio_visual_file(
    segments_manifest_path: str,
    frames_manifest_path: str,
    motion_manifest_path: str,
    audio_light_path: str | None = None,
    transcript_path: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Retourne le manifest de fusion audio-visuelle (dict JSON sérialisable)."""
    logging.info(
        "fuse_audio_visual_file called segments=%s frames=%s motion=%s",
        segments_manifest_path, frames_manifest_path, motion_manifest_path,
    )
    segments_file = _resolve_existing_project_file(segments_manifest_path, "Le manifest de segments")
    frames_file = _resolve_existing_project_file(frames_manifest_path, "Le manifest de frames")
    motion_file = _resolve_existing_project_file(motion_manifest_path, "Le manifest motion")

    audio_light_file: Path | None = None
    if audio_light_path is not None:
        audio_light_file = _resolve_existing_project_file(audio_light_path, "L'analyse audio légère")
    transcript_file: Path | None = None
    if transcript_path is not None:
        transcript_file = _resolve_existing_project_file(transcript_path, "Le transcript")

    tmp_output: str | None = None
    if output_path is not None:
        output_out = _resolve_output_path(output_path, "Le fichier de sortie")
        output_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_output = tempfile.mkstemp(suffix=".fusion.json")
        os.close(fd)
        output_out = Path(tmp_output)

    try:
        result = _run_fusion_subprocess(
            segments_file,
            frames_file,
            motion_file,
            audio_light_file,
            transcript_file,
            output_out,
        )
        logging.info(
            "fuse_audio_visual_file completed segments=%s ready=%s",
            result.get("summary", {}).get("segment_count"),
            result.get("summary", {}).get("ready_for_vlm_count"),
        )
        return result
    except Exception:
        logging.exception("fuse_audio_visual_file failed")
        raise
    finally:
        if tmp_output is not None:
            try:
                os.remove(tmp_output)
            except OSError:
                pass


def _run_vlm_pack_subprocess(
    fusion_file: Path,
    output_out: Path,
    max_frames_per_segment: int,
    skip_not_ready: bool,
    timeout: float = 180.0,
) -> dict:
    """Exécute la construction du pack VLM dans un sous-processus isolé et retourne son JSON.

    Même pattern robuste que les autres tools. Le pack contient du texte non latin
    (arabe/Darija) : on force l'UTF-8 du sous-processus (PYTHONUTF8 / PYTHONIOENCODING)
    et on lit le stderr avec errors="replace".
    """
    script = SCRIPTS / "build_vlm_pack.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".vlmpack.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--fusion", str(fusion_file),
        "--output", str(output_out),
        "--max-frames-per-segment", str(max_frames_per_segment),
    ]
    if skip_not_ready:
        cmd.append("--skip-not-ready")
    logging.info("build_vlm_pack_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Construction du pack VLM échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Construction du pack VLM interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="build_vlm_pack_file",
    description=(
        "Construire le pack VLM (par segment : frames sélectionnées + motion + parole + "
        "instructions/questions) à partir d'un manifest de fusion. Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def build_vlm_pack_file(
    fusion_manifest_path: str,
    output_path: str | None = None,
    max_frames_per_segment: int = 6,
    skip_not_ready: bool = False,
) -> dict:
    """Retourne le pack VLM (dict JSON sérialisable)."""
    logging.info(
        "build_vlm_pack_file called fusion=%s max_frames=%s skip_not_ready=%s",
        fusion_manifest_path, max_frames_per_segment, skip_not_ready,
    )
    fusion_file = _resolve_existing_project_file(fusion_manifest_path, "Le manifest de fusion")

    tmp_output: str | None = None
    if output_path is not None:
        output_out = _resolve_output_path(output_path, "Le fichier de sortie")
        output_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_output = tempfile.mkstemp(suffix=".vlm_pack.json")
        os.close(fd)
        output_out = Path(tmp_output)

    try:
        result = _run_vlm_pack_subprocess(
            fusion_file, output_out, max_frames_per_segment, skip_not_ready
        )
        logging.info(
            "build_vlm_pack_file completed segments=%s selected_frames=%s",
            result.get("summary", {}).get("segment_count"),
            result.get("summary", {}).get("total_selected_frames"),
        )
        return result
    except Exception:
        logging.exception("build_vlm_pack_file failed")
        raise
    finally:
        if tmp_output is not None:
            try:
                os.remove(tmp_output)
            except OSError:
                pass


def _run_prepare_vlm_subprocess(
    pack_file: Path,
    config_file: Path,
    output_dir: Path,
    timeout: float = 180.0,
) -> dict:
    """Prépare le run VLM dans un sous-processus isolé et retourne un dict assemblé.

    Délègue à ``scripts/prepare_vlm_run.py`` (PROCESSUS séparé, stdout/stderr vers
    fichier, attente noyau bornée par timeout, UTF-8 forcé car les prompts
    contiennent du texte non latin). Après exécution, on lit run_plan.json,
    selected_inputs.json et la liste des prompts pour construire la réponse.
    """
    script = SCRIPTS / "prepare_vlm_run.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".vlmrun.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--pack", str(pack_file),
        "--config", str(config_file),
        "--output-dir", str(output_dir),
    ]
    logging.info("prepare_vlm_run_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0:
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Préparation du run VLM échouée (code {proc.returncode}) : {err_tail}"
            )
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Préparation du run VLM interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass

    run_plan_path = output_dir / "run_plan.json"
    selected_inputs_path = output_dir / "selected_inputs.json"
    if not run_plan_path.exists() or not selected_inputs_path.exists():
        raise InspectError(
            "Préparation du run VLM : fichiers de sortie manquants (run_plan/selected_inputs)."
        )
    run_plan = json.loads(run_plan_path.read_text(encoding="utf-8"))
    selected_inputs = json.loads(selected_inputs_path.read_text(encoding="utf-8"))
    prompt_paths = sorted(p.as_posix() for p in output_dir.glob("prompt_segment_*.md"))

    return {
        "run_plan": run_plan,
        "selected_inputs": selected_inputs,
        "output_dir": output_dir.as_posix(),
        "prompt_paths": prompt_paths,
        "segment_count": run_plan.get("segment_count"),
        "ready_for_vlm_count": run_plan.get("ready_for_vlm_count"),
        "next_step": run_plan.get("next_step"),
    }


@app.tool(
    name="prepare_vlm_run_file",
    description=(
        "Préparer un dossier de run VLM (prompts Markdown par segment + plan) à partir "
        "d'un pack VLM et d'une config, sans appeler de modèle. Retourne un manifest JSON."
    ),
    run_in_thread=True,
)
def prepare_vlm_run_file(
    pack_path: str,
    config_path: str = "config/vlm_config.example.json",
    output_dir: str | None = None,
) -> dict:
    """Retourne le plan + inputs + chemins de prompts du run VLM (dict sérialisable)."""
    logging.info(
        "prepare_vlm_run_file called pack=%s config=%s output_dir=%s",
        pack_path, config_path, output_dir,
    )
    pack_file = _resolve_existing_project_file(pack_path, "Le pack VLM")
    config_file = _resolve_existing_project_file(config_path, "La config VLM")

    if output_dir is not None:
        out_resolved = _resolve_output_path(output_dir, "Le dossier de sortie")
    else:
        # Par défaut : outputs/vlm_runs/<video_stem> dérivé du pack.
        pack = json.loads(pack_file.read_text(encoding="utf-8"))
        stem = Path(pack.get("video_path", "video")).stem or "video"
        out_resolved = _resolve_output_path(f"outputs/vlm_runs/{stem}", "Le dossier de sortie")
    out_resolved.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_prepare_vlm_subprocess(pack_file, config_file, out_resolved)
        logging.info(
            "prepare_vlm_run_file completed segments=%s ready=%s",
            result.get("segment_count"), result.get("ready_for_vlm_count"),
        )
        return result
    except Exception:
        logging.exception("prepare_vlm_run_file failed")
        raise


def _resolve_existing_project_dir(path: str, label: str) -> Path:
    """Résout un chemin de DOSSIER devant exister et rester dans le projet."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not _is_within_project_root(candidate):
        raise InspectError(
            f"Accès refusé : {label} doit être dans le projet ({PROJECT_ROOT})"
        )
    if not candidate.exists() or not candidate.is_dir():
        raise InspectError(f"{label} introuvable : {candidate}")
    return candidate


def _run_validate_results_subprocess(
    results_dir: Path,
    pack_file: Path,
    output_out: Path,
    timeout: float = 120.0,
) -> dict:
    """Valide les résultats VLM dans un sous-processus isolé et retourne le résumé JSON.

    Même pattern robuste que les autres tools : délégation à
    ``scripts/validate_vlm_results.py`` (PROCESSUS séparé, stdout/stderr vers fichier,
    attente noyau bornée par timeout, UTF-8 forcé car les résultats peuvent contenir
    du texte non latin). Le résumé est lu depuis le fichier ``--output``.
    """
    script = SCRIPTS / "validate_vlm_results.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".validate.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--results-dir", str(results_dir),
        "--pack", str(pack_file),
        "--output", str(output_out),
    ]
    logging.info("validate_vlm_results_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode not in (0, 2) or not output_out.exists():
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Validation des résultats VLM échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Validation des résultats VLM interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="validate_vlm_results_file",
    description=(
        "Valider les réponses JSON d'un VLM (mode manual) d'un dossier vis-à-vis du "
        "pack VLM (segments attendus, timestamps, types, doublons, confidence, evidence). "
        "Retourne un résumé de validation JSON."
    ),
    run_in_thread=True,
)
def validate_vlm_results_file(
    results_dir: str,
    pack_path: str,
    output_path: str | None = None,
) -> dict:
    """Retourne le résumé de validation des résultats VLM (dict JSON sérialisable)."""
    logging.info(
        "validate_vlm_results_file called results_dir=%s pack=%s", results_dir, pack_path
    )
    results_resolved = _resolve_existing_project_dir(results_dir, "Le dossier de résultats")
    pack_file = _resolve_existing_project_file(pack_path, "Le pack VLM")

    tmp_output: str | None = None
    if output_path is not None:
        output_out = _resolve_output_path(output_path, "Le fichier de sortie")
        output_out.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, tmp_output = tempfile.mkstemp(suffix=".vlm_results_summary.json")
        os.close(fd)
        output_out = Path(tmp_output)

    try:
        result = _run_validate_results_subprocess(results_resolved, pack_file, output_out)
        logging.info(
            "validate_vlm_results_file completed valid=%s invalid=%s",
            result.get("valid_count"), result.get("invalid_count"),
        )
        return result
    except Exception:
        logging.exception("validate_vlm_results_file failed")
        raise
    finally:
        if tmp_output is not None:
            try:
                os.remove(tmp_output)
            except OSError:
                pass


def _run_vlm_agent_subprocess(
    run_dir: Path,
    pack_file: Path,
    config_file: Path,
    results_dir: Path,
    model: str | None,
    overwrite: bool,
    only: int | None,
    timeout: float = 1800.0,
) -> dict:
    """Exécute l'analyse VLM automatisée en sous-processus isolé.

    Délègue à ``scripts/run_vlm_agent.py`` (qui appelle le provider VLM Gemini via
    google-genai, valide par segment, puis produit le résumé). PROCESSUS séparé,
    stdout/stderr vers fichiers, attente noyau bornée par un ``timeout`` **généreux**
    (plusieurs appels modèle), UTF-8 forcé (texte arabe/Darija). Résumé relu du dossier.
    """
    script = SCRIPTS / "run_vlm_agent.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".vlm_agent.err")
    os.close(err_fd)
    summary_out = results_dir / "vlm_results_summary.json"
    cmd = [
        sys.executable,
        str(script),
        "--run-dir", str(run_dir),
        "--pack", str(pack_file),
        "--config", str(config_file),
        "--results-dir", str(results_dir),
        "--summary-output", str(summary_out),
    ]
    if model is not None:
        cmd += ["--model", str(model)]
    if only is not None:
        cmd += ["--only", str(only)]
    if overwrite:
        cmd.append("--overwrite")
    logging.info("run_vlm_agent_analysis: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        # rc != 0 peut signifier "au moins un segment invalide" (code 2) : on lit
        # quand même le résumé s'il existe, sinon on remonte l'erreur.
        if proc.returncode not in (0, 2) or not summary_out.exists():
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Analyse VLM (Gemini) échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(summary_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Analyse VLM (Gemini) interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="run_vlm_agent_analysis",
    description=(
        "Analyser tous les segments en mode automatic avec le provider gemini explicitement "
        "configuré (google-genai, clé GEMINI_API_KEY), valider chaque résultat et produire "
        "un résumé. Retourne le résumé JSON."
    ),
    run_in_thread=True,
)
def run_vlm_agent_analysis(
    run_dir: str = "outputs/vlm_runs/demo_darija",
    pack_path: str = "outputs/vlm_pack_demo_darija.json",
    config_path: str = "config/vlm_config.example.json",
    results_dir: str = "outputs/vlm_results/demo_darija",
    model: str | None = None,
    overwrite: bool = False,
    only: int | None = None,
) -> dict:
    """Retourne le résumé de validation des résultats VLM (dict JSON sérialisable)."""
    logging.info(
        "run_vlm_agent_analysis called run_dir=%s results_dir=%s model=%s overwrite=%s only=%s",
        run_dir, results_dir, model, overwrite, only,
    )
    run_resolved = _resolve_existing_project_dir(run_dir, "Le dossier de run VLM")
    pack_file = _resolve_existing_project_file(pack_path, "Le pack VLM")
    config_file = _resolve_existing_project_file(config_path, "La config VLM")
    results_resolved = _resolve_output_path(results_dir, "Le dossier de résultats")
    results_resolved.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_vlm_agent_subprocess(
            run_resolved, pack_file, config_file, results_resolved,
            model, overwrite, only,
        )
        logging.info(
            "run_vlm_agent_analysis completed valid=%s invalid=%s",
            result.get("valid_count"), result.get("invalid_count"),
        )
        return result
    except Exception:
        logging.exception("run_vlm_agent_analysis failed")
        raise


def _run_final_report_subprocess(
    results_dir: Path,
    fusion_file: Path,
    output_out: Path,
    text_out: Path,
    summary_file: Path | None,
    include_arc: bool = True,
    inspection_file: Path | None = None,
    timeout: float = 240.0,
) -> dict:
    """Construit le rapport vidéo final dans un sous-processus isolé et retourne le JSON.

    Délègue à ``scripts/build_final_report.py`` (agrégation déterministe). PROCESSUS
    séparé, stdout/stderr vers fichiers, attente noyau bornée, UTF-8 forcé (texte
    arabe/Darija). Le rapport est relu depuis le fichier ``--output``.
    """
    script = SCRIPTS / "build_final_report.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".final_report.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--results-dir", str(results_dir),
        "--fusion", str(fusion_file),
        "--output", str(output_out),
        "--text-output", str(text_out),
    ]
    if summary_file is not None:
        cmd += ["--summary", str(summary_file)]
    if not include_arc:
        cmd += ["--no-arc"]
    if inspection_file is not None:
        cmd += ["--inspection", str(inspection_file)]
    logging.info("build_final_report_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0 or not output_out.exists():
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Construction du rapport final échouée (code {proc.returncode}) : {err_tail}"
            )
        return json.loads(output_out.read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Construction du rapport final interrompue : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="build_final_report_file",
    description=(
        "Construire le rapport vidéo final : synthèse globale (scènes horodatées, "
        "personnages/objets, actions, ton, langue, incertitudes, résumé lisible, arc "
        "narratif) à partir des analyses VLM validées et de la fusion. Si include_arc "
        "(défaut True), un arc narratif est généré par UN appel LLM texte (sans image, "
        "sans ré-analyse vidéo) ; mettre include_arc=False pour un rapport 100% "
        "hors-ligne. Retourne le rapport JSON."
    ),
    run_in_thread=True,
)
def build_final_report_file(
    results_dir: str = "outputs/vlm_results/demo_darija",
    fusion_manifest_path: str = "outputs/audio_visual_fusion_demo_darija.json",
    summary_path: str | None = None,
    output_path: str = "outputs/final_report_demo_darija.json",
    text_output_path: str = "outputs/final_report_demo_darija.txt",
    include_arc: bool = True,
    inspection_path: str | None = None,
) -> dict:
    """Retourne le rapport vidéo final (dict JSON sérialisable)."""
    logging.info(
        "build_final_report_file called results_dir=%s fusion=%s", results_dir, fusion_manifest_path
    )
    results_resolved = _resolve_existing_project_dir(results_dir, "Le dossier de résultats")
    fusion_file = _resolve_existing_project_file(fusion_manifest_path, "Le manifest de fusion")

    summary_file: Path | None = None
    if summary_path is not None:
        summary_file = _resolve_existing_project_file(summary_path, "Le résumé de validation")

    inspection_file: Path | None = None
    if inspection_path is not None:
        inspection_file = _resolve_existing_project_file(inspection_path, "L'inspection technique")

    output_out = _resolve_output_path(output_path, "Le rapport de sortie")
    output_out.parent.mkdir(parents=True, exist_ok=True)
    text_out = _resolve_output_path(text_output_path, "Le résumé texte de sortie")
    text_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_final_report_subprocess(
            results_resolved, fusion_file, output_out, text_out, summary_file,
            include_arc=include_arc, inspection_file=inspection_file,
        )
        logging.info(
            "build_final_report_file completed scenes=%s", result.get("scene_count")
        )
        return result
    except Exception:
        logging.exception("build_final_report_file failed")
        raise


def _run_export_pdf_subprocess(
    report_file: Path,
    output_out: Path,
    font_path: str | None,
    timeout: float = 120.0,
) -> dict:
    """Exporte le rapport en PDF dans un sous-processus isolé et retourne les métadonnées.

    Délègue à ``scripts/export_report_pdf.py``. PROCESSUS séparé, stdout/stderr vers
    fichiers, attente noyau bornée, UTF-8 forcé. Le PDF étant binaire, on retourne
    son chemin et sa taille (pas son contenu).
    """
    script = SCRIPTS / "export_report_pdf.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".export_pdf.err")
    os.close(err_fd)
    cmd = [
        sys.executable,
        str(script),
        "--report", str(report_file),
        "--output", str(output_out),
    ]
    if font_path is not None:
        cmd += ["--font", str(font_path)]
    logging.info("export_report_pdf_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        if proc.returncode != 0 or not output_out.exists():
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Export PDF échoué (code {proc.returncode}) : {err_tail}"
            )
        return {
            "pdf_path": output_out.as_posix(),
            "size_bytes": output_out.stat().st_size,
            "report_path": report_file.as_posix(),
        }
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Export PDF interrompu : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass


@app.tool(
    name="export_report_pdf_file",
    description=(
        "Exporter le rapport vidéo final (JSON) en PDF mis en forme, avec rendu correct "
        "de l'arabe (police Unicode + reshaping + RTL). Retourne le chemin et la taille du PDF."
    ),
    run_in_thread=True,
)
def export_report_pdf_file(
    report_path: str = "outputs/final_report_demo_darija.json",
    output_path: str | None = None,
    font_path: str | None = None,
) -> dict:
    """Retourne les métadonnées du PDF produit (dict JSON sérialisable)."""
    logging.info("export_report_pdf_file called report=%s output=%s", report_path, output_path)
    report_file = _resolve_existing_project_file(report_path, "Le rapport JSON")

    if output_path is None:
        report = json.loads(report_file.read_text(encoding="utf-8"))
        stem = Path(report.get("video_path", "video")).stem or "video"
        output_out = _resolve_output_path(f"outputs/final_report_{stem}.pdf", "Le PDF de sortie")
    else:
        output_out = _resolve_output_path(output_path, "Le PDF de sortie")
    output_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = _run_export_pdf_subprocess(report_file, output_out, font_path)
        logging.info("export_report_pdf_file completed size=%s", result.get("size_bytes"))
        return result
    except Exception:
        logging.exception("export_report_pdf_file failed")
        raise


def _run_full_pipeline_subprocess(
    video_name: str,
    config_path: str,
    model: str | None,
    overwrite_analysis: bool,
    timeout: float = 3600.0,
) -> dict:
    """Exécute le pipeline complet dans un sous-processus isolé et retourne le statut.

    Délègue à ``scripts/run_full_pipeline.py`` (qui enchaîne tous les niveaux).
    PROCESSUS séparé, stdout/stderr vers fichiers, attente noyau bornée par un
    ``timeout`` **généreux** (analyse complète), UTF-8 forcé. Le statut est relu
    depuis le JSON stdout écrit dans un fichier temporaire (jamais via un pipe).
    """
    script = SCRIPTS / "run_full_pipeline.py"
    err_fd, err_path = tempfile.mkstemp(suffix=".pipeline.err")
    os.close(err_fd)
    out_fd, out_path = tempfile.mkstemp(suffix=".pipeline.json")
    os.close(out_fd)
    cmd = [py_executable(), str(script), video_name, "--config", config_path]
    if model is not None:
        cmd += ["--model", str(model)]
    if overwrite_analysis:
        cmd.append("--overwrite-analysis")
    logging.info("run_full_pipeline_file: lancement sous-processus %s", cmd)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        with open(err_path, "w", encoding="utf-8") as errf, open(out_path, "w", encoding="utf-8") as outf:
            proc = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=outf,
                stderr=errf,
                timeout=timeout,
                cwd=str(PROJECT_ROOT),
                env=env,
            )
        # rc 0 = tout OK ; rc 2 = pipeline interrompu mais statut écrit -> on le lit.
        if proc.returncode not in (0, 2):
            err_tail = ""
            try:
                err_tail = Path(err_path).read_text(encoding="utf-8", errors="replace")[-800:]
            except OSError:
                pass
            raise InspectError(
                f"Pipeline complet échoué (code {proc.returncode}) : {err_tail}"
            )
        result = json.loads(Path(out_path).read_text(encoding="utf-8"))
        if not isinstance(result, dict) or not result.get("run_id") or not result.get("run_dir"):
            raise InspectError("Le pipeline n'a pas retourné une identité de run valide.")
        return result
    except subprocess.TimeoutExpired as exc:
        raise InspectError(
            f"Pipeline complet interrompu : délai dépassé ({timeout}s)."
        ) from exc
    finally:
        try:
            os.remove(err_path)
        except OSError:
            pass
        try:
            os.remove(out_path)
        except OSError:
            pass


def py_executable() -> str:
    return sys.executable


@app.tool(
    name="run_full_pipeline_file",
    description=(
        "Exécuter le pipeline complet depuis le NOM d'une vidéo sous VIDEO_ROOT : "
        "inspection → segmentation → frames → motion → (audio →) fusion → pack VLM → "
        "suite selon le mode disabled/manual/automatic (Gemini seulement si explicitement "
        "configuré). Retourne run_id, run_dir, statut et chemins d'artefacts."
    ),
    run_in_thread=True,
)
def run_full_pipeline_file(
    video_name: str,
    config_path: str = "config/vlm_config.example.json",
    model: str | None = None,
    overwrite_analysis: bool = False,
) -> dict:
    """Retourne le statut du pipeline complet (dict JSON sérialisable)."""
    logging.info("run_full_pipeline_file called video=%s", video_name)
    # Vérifie l'existence/sécurité de la vidéo (sous VIDEO_ROOT/projet).
    resolved = _resolve_video_path(video_name)
    try:
        result = _run_full_pipeline_subprocess(
            video_name, config_path, model, overwrite_analysis
        )
        logging.info(
            "run_full_pipeline_file completed run_id=%s ok=%s",
            result.get("run_id"), result.get("ok"),
        )
        return result
    except Exception:
        logging.exception("run_full_pipeline_file failed")
        raise


def main() -> int:
    """Point d'entrée du serveur MCP stdio installable."""
    import asyncio

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logging.info(f"Démarrage MCP stdio. VIDEO_ROOT={VIDEO_ROOT}")
    asyncio.run(app.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
