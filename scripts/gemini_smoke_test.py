r"""Test de connectivité MINIMAL pour l'API Gemini (google-genai).

Isolé du pipeline : ne touche à aucun autre module/output. N'appelle ni Claude
ni le VLM. Lit la clé depuis la variable d'environnement GEMINI_API_KEY.

Usage :
    $env:PYTHONUTF8=1; .\.venv\Scripts\python.exe scripts\gemini_smoke_test.py
"""

from __future__ import annotations

import os
import sys


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _pick_flash_model(model_names: list[str]) -> str | None:
    """Choisit un modèle Flash de génération 3 dans la liste.

    Préférence : gemini-3-flash-preview > gemini-3.5-flash > Flash le plus
    récent listé (heuristique : numéro de version le plus élevé).
    """
    # Normalise : on retire un éventuel préfixe "models/".
    def short(name: str) -> str:
        return name.split("/", 1)[1] if name.startswith("models/") else name

    shorts = {short(n): n for n in model_names}

    # Préférences explicites, dans l'ordre.
    for preferred in ("gemini-3-flash-preview", "gemini-3.5-flash"):
        if preferred in shorts:
            return shorts[preferred]

    # Sinon, le Flash le plus récent : on garde les modèles "flash" puis on
    # trie par numéro de version détecté dans le nom (3.5 > 3 > 2.5 > 2 > 1.5).
    import re

    def version_key(name: str) -> tuple[float, str]:
        m = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        ver = float(m.group(1)) if m else -1.0
        return (ver, name)

    flash = [s for s in shorts if "flash" in s.lower()]
    if not flash:
        return None
    best = max(flash, key=version_key)
    return shorts[best]


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        _eprint(
            "ERREUR : variable d'environnement GEMINI_API_KEY absente.\n"
            "Définissez-la d'abord, par ex. :\n"
            '    $env:GEMINI_API_KEY = "votre_cle"\n'
            "puis relancez ce script. (Aucune clé n'est stockée en dur.)"
        )
        return 1

    try:
        from google import genai
    except ImportError:
        _eprint(
            "ERREUR : le paquet 'google-genai' n'est pas installé dans ce venv.\n"
            "Installez-le : .\\.venv\\Scripts\\python.exe -m pip install google-genai"
        )
        return 1

    client = genai.Client(api_key=api_key)

    # 1) Liste des modèles disponibles.
    print("=== Modèles disponibles ===")
    model_names: list[str] = []
    try:
        for m in client.models.list():
            name = getattr(m, "name", None) or str(m)
            model_names.append(name)
            print(f"  - {name}")
    except Exception as exc:  # noqa: BLE001 — message clair, pas de traceback
        _eprint(f"ERREUR lors de client.models.list() : {exc}")
        return 1

    if not model_names:
        _eprint("ERREUR : aucun modèle retourné par l'API.")
        return 1

    # 2) Choix d'un modèle Flash gén. 3.
    chosen = _pick_flash_model(model_names)
    if not chosen:
        _eprint(
            "ERREUR : aucun modèle Flash trouvé dans la liste ci-dessus ; "
            "impossible de faire l'appel de test."
        )
        return 1

    print(f"\n=== Modèle choisi pour l'appel de test ===\n  {chosen}")

    # 3) Un seul appel minimal.
    prompt = "Réponds uniquement par le mot OK."
    try:
        response = client.models.generate_content(model=chosen, contents=prompt)
    except Exception as exc:  # noqa: BLE001
        _eprint(f"ERREUR lors de l'appel generate_content : {exc}")
        return 1

    text = getattr(response, "text", None)
    print("\n=== Réponse ===")
    print(f"  modèle utilisé : {chosen}")
    print(f"  réponse        : {text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
