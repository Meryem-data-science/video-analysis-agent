"""Points d'entrée installables, délégués aux CLI historiques."""

from __future__ import annotations

import sys
from typing import Callable


def run_main() -> int:
    """Exécute le pipeline complet via le script existant."""
    from scripts.run_full_pipeline import main

    return main()


def validate_main() -> int:
    """Valide les résultats VLM via le script existant."""
    from scripts.validate_vlm_results import main

    return main()


def mcp_main() -> int:
    """Démarre le serveur MCP local."""
    from .mcp_server import main

    return main()


def web_main() -> int:
    """Démarre l'API HTTP locale (extra ``web`` requis)."""
    from .web_cli import main

    return main()


def generate_demo_main() -> int:
    """Génère la vidéo synthétique locale."""
    from scripts.generate_demo_video import main

    return main()


def evaluate_main() -> int:
    """Évalue un dataset ou un run existant."""
    from scripts.evaluate import main

    return main()


def benchmark_main() -> int:
    """Benchmarke le pipeline local sur un dataset."""
    from scripts.benchmark import main

    return main()


def _dispatch(command: str) -> Callable[[], int]:
    commands = {
        "run": run_main,
        "validate": validate_main,
        "mcp": mcp_main,
        "web": web_main,
        "generate-demo": generate_demo_main,
        "evaluate": evaluate_main,
        "benchmark": benchmark_main,
    }
    try:
        return commands[command]
    except KeyError as exc:
        raise SystemExit(f"Commande inconnue : {command!r}. Utilisez --help.") from exc


def main() -> int:
    """Route vers ``run``, ``validate`` ou ``mcp`` sans dupliquer leur logique."""
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(
            "usage: video-agent {run|validate|mcp|web|generate-demo|evaluate|benchmark} "
            "[options]\n\n"
            "  run       exécuter le pipeline complet\n"
            "  validate  valider des résultats VLM\n"
            "  mcp       démarrer le serveur MCP stdio\n"
            "  web       démarrer l'API FastAPI locale\n"
            "  generate-demo  générer la vidéo synthétique locale\n"
            "  evaluate  comparer un run à des annotations\n"
            "  benchmark répéter le pipeline et mesurer ses temps\n\n"
            "Utilisez 'video-agent <commande> --help' pour l'aide détaillée."
        )
        return 0
    command = sys.argv.pop(1)
    return _dispatch(command)()
