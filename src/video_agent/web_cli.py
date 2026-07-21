"""Commande de lancement de l'API FastAPI locale."""

from __future__ import annotations

import argparse
from pathlib import Path


def _find_frontend_index() -> Path | None:
    """Trouve le build UI dans le projet courant ou dans un checkout source."""
    candidates = (
        Path.cwd() / "frontend" / "dist" / "index.html",
        Path(__file__).resolve().parents[2] / "frontend" / "dist" / "index.html",
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Démarrer l'API locale Video Analysis Agent.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port HTTP (défaut: 8000)")
    parser.add_argument("--reload", action="store_true", help="Recharger en développement")
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Servir l'interface compilée depuis frontend/dist",
    )
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "L'extra web est requis : python -m pip install \".[web]\""
        ) from exc

    if args.ui:
        if _find_frontend_index() is None:
            raise SystemExit(
                "Interface compilée introuvable. Exécutez "
                "'cd frontend' puis 'npm run build', ou démarrez sans --ui."
            )

    if args.reload:
        uvicorn.run(
            "video_agent.web_api:app", host=args.host, port=args.port, reload=True
        )
    else:
        from .web_api import create_app

        uvicorn.run(
            create_app(serve_ui=True if args.ui else None),
            host=args.host,
            port=args.port,
            reload=False,
        )
    return 0
