"""Parser-independent ordinal semantics shared by Structural IR schemas and extraction."""

import re
import unicodedata
from enum import StrEnum


class OrdinalSystem(StrEnum):
    """Controlled syntax used to derive a structural ordinal key."""

    NONE = "none"
    FORMAL_CONTAINER = "formal_container"
    ARTICLE = "article"
    CLAUSE = "clause"
    POINT = "point"
    GENERIC_ROMAN = "generic_roman"
    GENERIC_DECIMAL = "generic_decimal"
    GENERIC_LETTER = "generic_letter"


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
_STRICT_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")


def normalize_detection_text(value: str) -> str:
    """Normalize a detection copy without changing stored evidence or offsets."""
    normalized = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return re.sub(r"[\t\f\v ]+", " ", normalized).strip()


def _normalized_ordinal(value: str) -> str:
    return normalize_detection_text(value).strip(".:-").casefold()


def roman_ordinal_key(value: str) -> str | None:
    """Convert only a syntactically valid canonical Roman numeral (1..3999)."""
    upper = _normalized_ordinal(value).upper()
    if not upper or _STRICT_ROMAN.fullmatch(upper) is None:
        return None
    total = 0
    previous = 0
    for character in reversed(upper):
        current = _ROMAN_VALUES[character]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return str(total)


def article_ordinal_key(value: str) -> str | None:
    """Preserve a decimal Vietnamese Article ordinal and lowercase suffix."""
    normalized = _normalized_ordinal(value)
    return normalized if re.fullmatch(r"\d+[a-zđ]?", normalized) else None


def clause_ordinal_key(value: str) -> str | None:
    """Preserve a decimal Vietnamese Clause ordinal and lowercase suffix."""
    return article_ordinal_key(value)


def point_ordinal_key(value: str) -> str | None:
    """Preserve a Vietnamese Point letter; never interpret it as Roman."""
    normalized = _normalized_ordinal(value)
    return normalized if re.fullmatch(r"[a-zđ]", normalized) else None


def formal_container_ordinal_key(value: str) -> str | None:
    """Normalize a formal container's decimal/suffix or strict Roman ordinal."""
    return article_ordinal_key(value) or roman_ordinal_key(value)


def generic_decimal_ordinal_key(value: str) -> str | None:
    """Preserve a dotted decimal outline key."""
    normalized = _normalized_ordinal(value)
    return normalized if re.fullmatch(r"\d+(?:\.\d+)*", normalized) else None


def generic_letter_ordinal_key(value: str) -> str | None:
    """Preserve an evidence-backed letter used by a non-legal generic outline."""
    return point_ordinal_key(value)


def ordinal_key_for_system(system: OrdinalSystem, raw: str | None) -> str | None:
    """Return the sole canonical key permitted for ``raw`` under ``system``."""
    if raw is None:
        return None
    normalizers = {
        OrdinalSystem.FORMAL_CONTAINER: formal_container_ordinal_key,
        OrdinalSystem.ARTICLE: article_ordinal_key,
        OrdinalSystem.CLAUSE: clause_ordinal_key,
        OrdinalSystem.POINT: point_ordinal_key,
        OrdinalSystem.GENERIC_ROMAN: roman_ordinal_key,
        OrdinalSystem.GENERIC_DECIMAL: generic_decimal_ordinal_key,
        OrdinalSystem.GENERIC_LETTER: generic_letter_ordinal_key,
    }
    normalizer = normalizers.get(system)
    return normalizer(raw) if normalizer is not None else None


__all__ = [
    "OrdinalSystem",
    "article_ordinal_key",
    "clause_ordinal_key",
    "formal_container_ordinal_key",
    "generic_decimal_ordinal_key",
    "generic_letter_ordinal_key",
    "normalize_detection_text",
    "ordinal_key_for_system",
    "point_ordinal_key",
    "roman_ordinal_key",
]
