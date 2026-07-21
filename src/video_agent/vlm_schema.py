"""Schéma VLM visuel — INDÉPENDANT de tout fournisseur (ni Claude, ni Gemini).

Centralise le contrat de sortie du VLM visuel (par segment) afin que les providers
(Gemini), le validateur et les tests s'y réfèrent **sans dépendre** d'un module lié à
un fournisseur. Contient :
- `build_output_schema()` : le json_schema canonique (avec `const` sur segment_id) ;
- `to_gemini_schema()` : variante compatible google-genai (sans `const`/`additionalProperties`) ;
- `VLM_PROMPT_DIRECTIVES` : consignes de prompt (langue/mood/texte_ecran) communes ;
- `normalize_vlm_result()` : compatibilité historique, sans réparation des données.

Aucun appel modèle/API ici : pure logique de données.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

# Champs obligatoires du contrat VLM (alignés sur vlm_result_validation.py).
REQUIRED_FIELDS: List[str] = [
    "segment_id", "start_s", "end_s", "combined_text",
    "scene_description", "people_objects", "actions",
    "motion_description", "audio_visual_relation", "uses_speech",
    "mood", "texte_ecran", "confidence", "uncertainties", "evidence",
]


def build_output_schema(
    segment_id: int,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    combined_text: Optional[str] = None,
) -> Dict[str, Any]:
    """json_schema CANONIQUE de sortie pour un segment donné.

    `segment_id` est fixé via `const` (l'identité du segment nous appartient, pas au
    modèle). Utilisé tel quel par les providers qui supportent `const` ; sinon, passer
    par `to_gemini_schema()` puis réinjecter `segment_id` après parsing.
    """
    str_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(REQUIRED_FIELDS),
        "properties": {
            "segment_id": {"type": "integer", "const": int(segment_id)},
            "start_s": {"type": "number", **({"const": float(start_s)} if start_s is not None else {})},
            "end_s": {"type": "number", **({"const": float(end_s)} if end_s is not None else {})},
            "combined_text": {"type": "string", **({"const": combined_text} if combined_text is not None else {})},
            "scene_description": {"type": "string"},
            "people_objects": str_array,
            "actions": str_array,
            "motion_description": {"type": "string"},
            "audio_visual_relation": {"type": "string"},
            "uses_speech": {"type": "boolean"},
            "mood": str_array,
            "texte_ecran": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "lang", "location"],
                    "properties": {
                        "text": {"type": "string"},
                        "lang": {"type": "string"},
                        "location": {"type": "string"},
                    },
                },
            },
            "confidence": {"type": "number"},
            "uncertainties": str_array,
            "evidence": {
                "type": "object",
                "additionalProperties": False,
                "required": ["visual", "audio"],
                "properties": {"visual": str_array, "audio": str_array},
            },
        },
    }


def to_gemini_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Adapte un json_schema au sous-ensemble accepté par google-genai `response_schema`.

    Retire récursivement les clés non supportées (`additionalProperties`, `const`).
    Ne modifie pas l'entrée (copie profonde).
    """
    cleaned = copy.deepcopy(schema)

    def _strip(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("additionalProperties", None)
            node.pop("const", None)
            for v in node.values():
                _strip(v)
        elif isinstance(node, list):
            for v in node:
                _strip(v)

    _strip(cleaned)
    return cleaned


# Consignes de prompt communes (langue FR, mood, texte à l'écran). Ajoutées au prompt
# du segment par n'importe quel provider — sans modifier le schéma de sortie.
_FRENCH_DIRECTIVE = (
    "\n\n--- CONSIGNE DE LANGUE (PRIORITAIRE) ---\n"
    "Rédige TOUS les champs descriptifs du JSON de sortie EN FRANÇAIS, quelle que soit "
    "la langue de la vidéo, de l'audio ou des images : scene_description, people_objects, "
    "actions, motion_description, audio_visual_relation et uncertainties.\n"
    "NE TRADUIS PAS et conserve dans leur langue d'origine : les paroles transcrites "
    "(le texte fourni comme « Parole » / combined_text), tout texte à l'écran que tu "
    "cites verbatim, et les noms propres (personnes, marques, clubs, équipes, lieux). "
    "Si tu cites un de ces éléments, garde-le tel quel (une glose française entre "
    "parenthèses est permise si utile).\n"
    "Le schéma JSON, les noms de champs et les types restent strictement identiques."
)

_MOOD_DIRECTIVE = (
    "\n\n--- CHAMP mood (ÉMOTION) ---\n"
    "Renseigne le champ `mood` : une LISTE COURTE (1 à 3 entrées) en français décrivant "
    "l'émotion ou le ton DOMINANT de la scène, d'après ce que tu perçois déjà "
    "(expressions des visages, ambiance, relation audio-visuel). "
    "Exemples : [\"joie\", \"tendresse\"], [\"tension\", \"colère\"], [\"surprise\"]. "
    "Si aucune émotion n'est identifiable, mets une liste vide []."
)

_SCREEN_TEXT_DIRECTIVE = (
    "\n\n--- CHAMP texte_ecran (TEXTE À L'ÉCRAN) ---\n"
    "Lis TOUT le texte visible à l'écran : sous-titres, menus, panneaux, pancartes, "
    "logos/marques, watermarks/pseudos, incrustations, génériques. Pour CHAQUE mention, "
    "donne un objet { text, lang, location } :\n"
    "- text : le texte EXACT, mot pour mot, sans traduction ni correction (conserve la "
    "langue/écriture d'origine, y compris l'arabe) ;\n"
    "- lang : code langue ISO court (« ar », « fr », « en »…) ou « inconnue » ;\n"
    "- location : emplacement bref (« haut », « bas », « centre », « watermark coin »…).\n"
    "Ne lis QUE ce qui est réellement lisible ; aucune invention. Aucun texte visible -> []."
)

# Rappel JSON strict (utile pour les modèles sans `response_schema` garanti).
_JSON_DIRECTIVE = (
    "\n\n--- FORMAT DE SORTIE (OBLIGATOIRE) ---\n"
    "Réponds UNIQUEMENT par un objet JSON valide conforme au schéma imposé "
    "(champs : " + ", ".join(REQUIRED_FIELDS) + "). "
    "Aucun texte hors du JSON, aucun bloc Markdown ```json."
)

VLM_PROMPT_DIRECTIVES = _FRENCH_DIRECTIVE + _MOOD_DIRECTIVE + _SCREEN_TEXT_DIRECTIVE + _JSON_DIRECTIVE


# --- Normalisation robuste vers le contrat -----------------------------------------


def _as_str(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _as_str_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    return [i for i in x if isinstance(i, str)]


def _clamp_confidence(x: Any) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def _normalize_texte_ecran(x: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(x, list):
        return out
    for it in x:
        if not isinstance(it, dict):
            continue
        text = it.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        out.append({
            "text": text,
            "lang": _as_str(it.get("lang")) or "inconnue",
            "location": _as_str(it.get("location")),
        })
    return out


def normalize_vlm_result(raw: Any, segment_id: int) -> Dict[str, Any]:
    """Retourne une copie de compatibilité sans corriger la réponse.

    ``segment_id`` est volontairement ignoré : l'identité fournie par le modèle doit
    être validée, jamais remplacée silencieusement par l'identité attendue.
    """
    return copy.deepcopy(raw) if isinstance(raw, dict) else {}
