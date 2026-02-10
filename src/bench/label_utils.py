from __future__ import annotations

import re
from typing import Dict, List

# All 905 base Pokemon names (lowercase) for fuzzy matching
_POKEMON_NAMES: set[str] | None = None


def _load_pokemon_names() -> set[str]:
    """Lazy-load Pokemon names from metadata if available."""
    global _POKEMON_NAMES
    if _POKEMON_NAMES is not None:
        return _POKEMON_NAMES
    try:
        import pandas as pd
        from pathlib import Path
        # Try multiple possible locations
        for csv_path in [
            Path("data/metadata.csv"),
            Path(__file__).parent.parent.parent / "data" / "metadata.csv",
        ]:
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                _POKEMON_NAMES = set(df["label"].str.lower().unique())
                return _POKEMON_NAMES
    except Exception:
        pass
    _POKEMON_NAMES = set()
    return _POKEMON_NAMES


_DEFAULT_ALIASES: Dict[str, str] = {
    "farfetchd": "Farfetch'd",
    "farfetch'd": "Farfetch'd",
    "mr mime": "Mr. Mime",
    "mr. mime": "Mr. Mime",
    "mime jr": "Mime Jr.",
    "mime jr.": "Mime Jr.",
    "nidoran f": "Nidoran♀",
    "nidoran female": "Nidoran♀",
    "nidoran m": "Nidoran♂",
    "nidoran male": "Nidoran♂",
    "type null": "Type: Null",
    "type: null": "Type: Null",
    "ho oh": "Ho-Oh",
    "ho-oh": "Ho-Oh",
    "porygon z": "Porygon-Z",
    "porygon-z": "Porygon-Z",
    "porygon2": "Porygon2",
    "porygon 2": "Porygon2",
}


def normalize_label(name: str) -> str:
    if name is None:
        return ""
    text = name.strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("’", "'")
    return text


def canonicalize_label(name: str, aliases: Dict[str, str] | None = None) -> str:
    aliases = aliases or _DEFAULT_ALIASES
    norm = normalize_label(name)
    if norm in aliases:
        return aliases[norm]
    # Title-case words while preserving tokens like 'Jr.' and apostrophes.
    parts = [p for p in norm.split(" ") if p]
    return " ".join(p.capitalize() for p in parts)


def extract_pokemon_name(response: str) -> str:
    """
    Extract Pokemon name from VLM response using multiple strategies.

    Strategies (in order):
    1. Explicit format: "NAME: <name>" or "Answer: <name>"
    2. Pattern: "It's <name>!" or "This is <name>"
    3. Pattern: "The Pokemon is <name>"
    4. First Pokemon name found in response (fuzzy match against known names)
    5. First capitalized word that looks like a name
    """
    if not response or not isinstance(response, str):
        return ""

    text = response.strip()

    # Strategy 1: Explicit format patterns
    explicit_patterns = [
        r"(?:NAME|ANSWER|Pokemon|Pokémon)\s*[:=]\s*([A-Za-z][A-Za-z0-9\-'\. ]{1,30})",
        r"^([A-Za-z][A-Za-z0-9\-'\.]+)$",  # Single word response
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip().rstrip(".,!?")
            if name and len(name) > 1:
                return name

    # Strategy 2: "It's X" or "This is X" patterns
    its_patterns = [
        r"(?:It'?s|This is|That'?s|That is)\s+(?:a\s+)?([A-Z][a-z]+(?:[-'][A-Za-z]+)?)",
        r"(?:It'?s|This is|That'?s|That is)\s+([A-Z][a-z]+(?:[-'][A-Za-z]+)?)",
    ]
    for pattern in its_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip().rstrip(".,!?")
            if name and len(name) > 2:
                return name

    # Strategy 3: "The Pokemon is X" pattern
    pokemon_is = re.search(
        r"(?:The\s+)?(?:Pokemon|Pokémon)\s+(?:is|appears to be|looks like|seems to be)\s+([A-Z][a-z]+(?:[-'][A-Za-z]+)?)",
        text, re.IGNORECASE
    )
    if pokemon_is:
        name = pokemon_is.group(1).strip().rstrip(".,!?")
        if name and len(name) > 2:
            return name

    # Strategy 4: Match against known Pokemon names
    known_names = _load_pokemon_names()
    if known_names:
        text_lower = text.lower()
        # Look for exact matches first (longer names first to match "Mr. Mime" before "Mime")
        for name in sorted(known_names, key=len, reverse=True):
            # Word boundary match
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, text_lower):
                return name

    # Strategy 5: First capitalized word that's 3+ chars (likely a name)
    cap_word = re.search(r'\b([A-Z][a-z]{2,}(?:[-][A-Z][a-z]+)?)\b', text)
    if cap_word:
        name = cap_word.group(1)
        # Filter out common non-Pokemon words
        skip_words = {"The", "This", "That", "Pokemon", "Pokémon", "Image", "Picture",
                      "Silhouette", "Answer", "Name", "Yes", "Its", "Here"}
        if name not in skip_words:
            return name

    # Fallback: return first word if short response
    words = text.split()
    if words and len(words) <= 3:
        return words[0].strip(".,!?")

    return ""


def extract_pokemon_names_topk(response: str, k: int = 5) -> List[str]:
    """
    Extract up to k Pokemon names from response.
    Handles pipe-delimited format and comma-separated lists.
    """
    if not response or not isinstance(response, str):
        return []

    names: List[str] = []

    # Check for pipe-delimited format first
    if "|" in response:
        parts = [p.strip() for p in response.split("|") if p.strip()]
        for part in parts[:k]:
            name = extract_pokemon_name(part)
            if name:
                names.append(name)
        if names:
            return names

    # Check for numbered list (1. X, 2. Y, ...)
    numbered = re.findall(r'\d+\.\s*([A-Z][a-z]+(?:[-\'][A-Za-z]+)?)', response)
    if numbered:
        return numbered[:k]

    # Check for comma-separated list
    if "," in response:
        parts = [p.strip() for p in response.split(",")]
        for part in parts[:k]:
            name = extract_pokemon_name(part)
            if name:
                names.append(name)
        if names:
            return names

    # Single extraction
    name = extract_pokemon_name(response)
    return [name] if name else []
