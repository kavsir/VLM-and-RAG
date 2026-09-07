"""High-precision deterministic Vietnamese semantic mention extraction."""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from vlm_rag.semantic_ir.models import (
    LegalReferenceComponents,
    LegalReferenceScope,
    QuantityComponents,
    SemanticMentionKind,
)

_DOCUMENT_CODE = r"[A-ZĐ][A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*"
_DOCUMENT_IDENTIFIER = re.compile(
    rf"(?<![\w/])(?P<identifier>\d{{1,4}}/(?:\d{{4}}/)?{_DOCUMENT_CODE})(?![\w/])"
)
_NUMERIC_DATE = re.compile(
    r"(?<!\d)(?P<day>0?[1-9]|[12]\d|3[01])/(?P<month>0?[1-9]|1[0-2])/(?P<year>\d{4})(?!\d)"
)
_WORD_DATE = re.compile(
    r"\bngày\s+(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})\s+năm\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    r"(?<![\w/])(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|ha|km²|km2|km|m²|m2|m|cm|mm|người|đồng|tỷ đồng|triệu đồng)\b",
    re.IGNORECASE,
)
_LEGAL_REFERENCE = re.compile(
    r"\b(?:(?P<point>điểm\s+[a-zđ])\s+)?(?:(?P<clause>khoản\s+\d+[a-z]?)\s+)?"
    r"(?P<article>Điều\s+\d+[a-z]?)"
    r"(?:\s+(?:của\s+)?(?P<instrument>Luật|Nghị định|Thông tư|Quyết định)"
    rf"(?:\s+số)?\s+(?P<number>\d{{1,4}}/(?:\d{{4}}/)?{_DOCUMENT_CODE}))?",
    re.IGNORECASE,
)
_AUTHORITY = re.compile(
    r"\b(?:Quốc hội|Thủ tướng Chính phủ|Chính phủ|Ủy ban nhân dân|UBND|HĐND|"
    r"Bộ\s+(?:Xây dựng|Kế hoạch và Đầu tư|Tài chính|Tài nguyên và Môi trường)|"
    r"Sở\s+(?:Xây dựng|Kế hoạch và Đầu tư|Tài chính|Tài nguyên và Môi trường|"
    r"Quy hoạch - Kiến trúc)|"
    r"Viện\s+(?:Quy hoạch xây dựng Hà Nội|Quy hoạch đô thị và nông thôn quốc gia))\b"
)


@dataclass(frozen=True, slots=True)
class MentionCandidate:
    kind: SemanticMentionKind
    start: int
    end: int
    raw_text: str
    normalized_value: str | None
    legal_reference: LegalReferenceComponents | None = None
    quantity: QuantityComponents | None = None


def _normalize_identifier(value: str) -> str:
    return unicodedata.normalize("NFKC", value).upper().replace(" ", "")


def _safe_date(match: re.Match[str]) -> str | None:
    try:
        return date(
            int(match.group("year")), int(match.group("month")), int(match.group("day"))
        ).isoformat()
    except ValueError:
        return None


def _number_key(value: str | None, prefix: str) -> str | None:
    if value is None:
        return None
    return re.sub(rf"(?i)^{prefix}\s+", "", value).lower()


def _safe_number(value: str) -> str | None:
    if value.isdigit():
        return value
    if value.count(",") == 1 and "." not in value:
        whole, fraction = value.split(",")
        if whole.isdigit() and fraction.isdigit():
            return f"{whole}.{fraction}"
    return None


def extract_mention_candidates(text: str) -> tuple[MentionCandidate, ...]:
    """Return stable, exact-span, high-precision mention candidates."""
    values: list[MentionCandidate] = []
    occupied: set[tuple[int, int, SemanticMentionKind]] = set()
    identifier_spans: list[tuple[int, int]] = []

    def add(candidate: MentionCandidate) -> None:
        key = (candidate.start, candidate.end, candidate.kind)
        if key not in occupied:
            occupied.add(key)
            values.append(candidate)

    for match in _LEGAL_REFERENCE.finditer(text):
        instrument = match.group("instrument")
        number = match.group("number")
        add(
            MentionCandidate(
                kind=SemanticMentionKind.LEGAL_REFERENCE,
                start=match.start(),
                end=match.end(),
                raw_text=match.group(0),
                normalized_value=None,
                legal_reference=LegalReferenceComponents(
                    instrument_type=instrument.casefold() if instrument else None,
                    instrument_number_raw=number,
                    instrument_number_normalized=(
                        _normalize_identifier(number) if number else None
                    ),
                    article=_number_key(match.group("article"), "Điều"),
                    clause=_number_key(match.group("clause"), "khoản"),
                    point=_number_key(match.group("point"), "điểm"),
                    scope=(
                        LegalReferenceScope.EXTERNAL_DOCUMENT
                        if instrument or number
                        else LegalReferenceScope.SELF_DOCUMENT
                    ),
                ),
            )
        )
    for match in _DOCUMENT_IDENTIFIER.finditer(text):
        raw = match.group("identifier")
        identifier_spans.append((match.start(), match.end()))
        add(
            MentionCandidate(
                SemanticMentionKind.DOCUMENT_IDENTIFIER,
                match.start(),
                match.end(),
                raw,
                _normalize_identifier(raw),
            )
        )
    for pattern in (_NUMERIC_DATE, _WORD_DATE):
        for match in pattern.finditer(text):
            normalized = _safe_date(match)
            if normalized is not None:
                add(
                    MentionCandidate(
                        SemanticMentionKind.TEMPORAL_EXPRESSION,
                        match.start(),
                        match.end(),
                        match.group(0),
                        normalized,
                    )
                )
    unit_map = {"m2": "m²", "km2": "km²"}
    for match in _QUANTITY.finditer(text):
        raw_value = match.group("value")
        raw_unit = match.group("unit")
        normalized_number = _safe_number(raw_value)
        normalized_unit = unit_map.get(raw_unit.casefold(), raw_unit.casefold())
        add(
            MentionCandidate(
                SemanticMentionKind.QUANTITY,
                match.start(),
                match.end(),
                match.group(0),
                (
                    f"{normalized_number} {normalized_unit}"
                    if normalized_number is not None
                    else None
                ),
                quantity=QuantityComponents(
                    raw_value=raw_value,
                    normalized_numeric_value=normalized_number,
                    raw_unit=raw_unit,
                    normalized_unit=normalized_unit,
                ),
            )
        )
    for match in _AUTHORITY.finditer(text):
        if any(start <= match.start() and match.end() <= end for start, end in identifier_spans):
            continue
        raw = match.group(0)
        add(
            MentionCandidate(
                SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
                match.start(),
                match.end(),
                raw,
                unicodedata.normalize("NFKC", raw).strip(),
            )
        )
    values.sort(key=lambda item: (item.start, item.end, item.kind.value))
    return tuple(values)


__all__ = ["MentionCandidate", "extract_mention_candidates"]
