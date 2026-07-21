"""Refuse les secrets, chemins personnels et artefacts lourds suivis par Git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml",
}
BANNED_SUFFIXES = {
    ".aac", ".avi", ".ckpt", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4",
    ".onnx", ".pth", ".pt", ".safetensors", ".wav", ".webm",
}
BANNED_PREFIXES = (
    ".venv/", "outputs/", "runs/", "videos/", "vidéos/", "models/",
)
IGNORED_FALLBACK_DIRS = {
    ".agents", ".codex", ".git", ".pytest_cache", ".ruff_cache", ".venv",
    ".venv-ocr", "Flow_regeneration", "outputs", "sap_fiori_dashboard", "videos",
    "vidéos", "_ocr_test", "__pycache__",
}
IGNORED_FALLBACK_FILES = {"AGENTS.md", "CLAUDE.md", "inspection.py"}

PERSONAL_PATH = re.compile(r"(?i)\b[A-Z]:[\\/]Users[\\/][^\\/\s\"']+")
SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])(?:sk-|ghp_|github_pat_)[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _git_files() -> list[Path] | None:
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return [ROOT / item.decode("utf-8") for item in proc.stdout.split(b"\0") if item]


def _fallback_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(
            part in IGNORED_FALLBACK_DIRS
            or part.startswith(".venv-")
            or part.startswith(".test_tmp")
            for part in rel.parts
        ):
            continue
        if rel.name in IGNORED_FALLBACK_FILES:
            continue
        if len(rel.parts) >= 2 and rel.parts[0] == "scripts" and rel.name.startswith("_"):
            continue
        files.append(path)
    return files


def main() -> int:
    tracked = _git_files()
    mode = "fichiers suivis par Git"
    if tracked is None:
        tracked = _fallback_files()
        mode = "fichiers publiables (fallback : aucun index Git disponible)"

    errors: list[str] = []
    for path in tracked:
        rel = path.relative_to(ROOT).as_posix()
        rel_lower = rel.lower()
        if rel_lower.startswith(BANNED_PREFIXES) or path.suffix.lower() in BANNED_SUFFIXES:
            errors.append(f"artefact interdit : {rel}")
            continue
        if rel_lower == ".env" or (rel_lower.startswith(".env.") and rel_lower != ".env.example"):
            errors.append(f"configuration secrète interdite : {rel}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        if PERSONAL_PATH.search(text):
            errors.append(f"chemin Windows personnel : {rel}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"secret potentiel ({pattern.pattern}) : {rel}")

    if errors:
        print("Contrôle d'hygiène échoué :", file=sys.stderr)
        for error in sorted(set(errors)):
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Contrôle d'hygiène OK : {len(tracked)} {mode} inspectés.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
