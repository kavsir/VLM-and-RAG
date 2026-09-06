"""Deterministic Vietnamese legal/planning structure extraction from Physical IR v1."""

import hashlib
import re
from dataclasses import dataclass, field

from vlm_rag.physical_ir.models import BlockDisposition
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.v1 import BlockKindV1, PhysicalDocumentV1
from vlm_rag.structural_ir.anchors import PhysicalLineEvent, iter_block_lines, physical_text_events
from vlm_rag.structural_ir.models import (
    AnchorRole,
    PhysicalAnchor,
    RecognitionEvidence,
    RecognitionMethod,
    StructuralDocument,
    StructuralNode,
    StructuralNodeKind,
    StructuralProfile,
)
from vlm_rag.structural_ir.ordinals import (
    article_ordinal_key,
    clause_ordinal_key,
    formal_container_ordinal_key,
    generic_decimal_ordinal_key,
    generic_letter_ordinal_key,
    normalize_detection_text,
    point_ordinal_key,
    roman_ordinal_key,
)

_FORMAL_PATTERNS: tuple[tuple[StructuralNodeKind, re.Pattern[str], str], ...] = (
    (
        StructuralNodeKind.SUBSECTION,
        re.compile(
            r"^\s*(?:tiểu\s+mục|tieu\s+muc)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_SUBSECTION_V1",
    ),
    (
        StructuralNodeKind.APPENDIX,
        re.compile(
            r"^\s*(?:phụ\s+lục|phu\s+luc)(?:\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?))?(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_APPENDIX_V1",
    ),
    (
        StructuralNodeKind.CHAPTER,
        re.compile(
            r"^\s*(?:chương|chuong)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_CHAPTER_V1",
    ),
    (
        StructuralNodeKind.ARTICLE,
        re.compile(
            r"^\s*(?:điều|dieu)\s+(?P<ordinal>\d+[a-zđ]?)(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_ARTICLE_V1",
    ),
    (
        StructuralNodeKind.PART,
        re.compile(
            r"^\s*(?:phần|phan)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_PART_V1",
    ),
    (
        StructuralNodeKind.SECTION,
        re.compile(
            r"^\s*(?:mục|muc)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?:(?P<separator>[.\-:])(?=\s|$)|(?=\s|$))\s*",
            re.IGNORECASE,
        ),
        "VI_FORMAL_SECTION_V1",
    ),
)
_CLAUSE = re.compile(r"^\s*(?P<ordinal>\d+[a-zđ]?)\.\s+", re.IGNORECASE)
_POINT = re.compile(r"^\s*(?P<ordinal>[a-zđ])\)\s+", re.IGNORECASE)
_GENERIC_LETTER = _POINT
_GENERIC_ROMAN = re.compile(r"^\s*(?P<ordinal>[IVXLCDM]+)\.\s+", re.IGNORECASE)
_GENERIC_DECIMAL = re.compile(r"^\s*(?P<ordinal>\d+(?:\.\d+)*)\.(?:\s+|$)", re.IGNORECASE)
_TOC_HEADING = re.compile(r"^\s*(?:mục\s+lục|muc\s+luc)\s*$", re.IGNORECASE)
_TOC_LEADER_ENTRY = re.compile(r"(?:\.{3,}|…{2,})\s*(?:trang\s*)?\d+\s*$", re.IGNORECASE)
_PROSE_CONTINUATION = re.compile(
    r"^(?:của|cua|nêu\s+trên|neu\s+tren|kèm\s+theo|kem\s+theo)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _Detection:
    kind: StructuralNodeKind
    ordinal_raw: str | None
    ordinal_key: str | None
    marker_end: int
    method: RecognitionMethod
    rule_id: str
    generic_level: int | None = None


@dataclass(frozen=True, slots=True)
class StructuralDiagnostic:
    """Deterministic trace for a marker-shaped event rejected by the extractor."""

    category: str
    page_index: int
    block_id: str
    excerpt: str
    candidate_kind: StructuralNodeKind | None = None
    candidate_ordinal: str | None = None
    canonical_key: str | None = None


@dataclass(slots=True)
class _NodeDraft:
    id: str
    kind: StructuralNodeKind
    parent_id: str | None
    depth: int
    ordinal_raw: str | None
    ordinal_key: str | None
    marker_text: str | None
    title: str | None
    recognition_evidence: RecognitionEvidence
    canonical_path: str
    heading_anchors: list[PhysicalAnchor] = field(default_factory=list)
    direct_content_anchors: list[PhysicalAnchor] = field(default_factory=list)

    def freeze(self) -> StructuralNode:
        return StructuralNode(
            id=self.id,
            kind=self.kind,
            parent_id=self.parent_id,
            depth=self.depth,
            ordinal_raw=self.ordinal_raw,
            ordinal_key=self.ordinal_key,
            marker_text=self.marker_text,
            title=self.title,
            heading_anchors=tuple(self.heading_anchors),
            direct_content_anchors=tuple(self.direct_content_anchors),
            recognition_evidence=self.recognition_evidence,
            canonical_path=self.canonical_path,
        )


def _heading_like(text: str, kind: BlockKindV1, *, allow_short: bool = False) -> bool:
    candidate = normalize_detection_text(text)
    if not candidate or len(candidate) > 160 or len(candidate.split()) > 18:
        return False
    if kind == BlockKindV1.TITLE:
        return True
    letters = [character for character in candidate if character.isalpha()]
    if letters and candidate == candidate.upper():
        return True
    return allow_short and len(candidate) <= 100 and not candidate.endswith((";", ","))


def _formal_remainder_is_heading(
    kind: StructuralNodeKind, remainder: str, separator: str | None
) -> bool:
    """Reject prose continuations independently of a parser's TITLE classification."""
    normalized = normalize_detection_text(remainder)
    if not normalized:
        return True
    if _PROSE_CONTINUATION.match(normalized) is not None:
        return False
    if separator is not None:
        return True
    if kind == StructuralNodeKind.ARTICLE:
        return False
    title_part = normalized.split("(", maxsplit=1)[0].strip()
    letters = [character for character in title_part if character.isalpha()]
    return bool(letters) and title_part == title_part.upper()


def _starts_with_uppercase_word(value: str) -> bool:
    """Require visible sentence-initial heading evidence for contextual outline children."""
    normalized = normalize_detection_text(value)
    return bool(normalized) and normalized[0].isalpha() and normalized[0].isupper()


class VietnameseStructuralExtractor:
    """Conservative deterministic extractor for the Vietnamese legal/planning profile."""

    profile = StructuralProfile.VI_LEGAL_PLANNING_V1

    def extract(
        self,
        physical: PhysicalDocumentV1,
        *,
        diagnostics: list[StructuralDiagnostic] | None = None,
    ) -> StructuralDocument:
        """Extract hierarchy and complete content anchors from Physical IR wire version 2."""
        diagnostic_sink = diagnostics if diagnostics is not None else []
        toc_events = self._toc_events(physical)
        physical_payload = physical_document_v1_to_json(physical).encode("utf-8")
        physical_sha = hashlib.sha256(physical_payload).hexdigest()
        root = _NodeDraft(
            id="node-000000",
            kind=StructuralNodeKind.DOCUMENT,
            parent_id=None,
            depth=0,
            ordinal_raw=None,
            ordinal_key=None,
            marker_text=None,
            title=None,
            recognition_evidence=RecognitionEvidence(
                rule_id="DOCUMENT_ROOT_V1",
                recognition_method=RecognitionMethod.DOCUMENT_ROOT,
                matched_text="DOCUMENT",
            ),
            canonical_path="document",
        )
        drafts = [root]
        by_id = {root.id: root}
        active: dict[StructuralNodeKind, str] = {}
        generic_stack: list[tuple[int, str, str, RecognitionMethod, bool]] = []
        used_paths = {root.canonical_path}
        current_id = root.id
        pending_title_id: str | None = None
        document_order = 0

        for page in physical.pages:
            for block in page.blocks:
                if block.disposition != BlockDisposition.CONTENT:
                    document_order += 1
                    continue
                if not block.text:
                    by_id[current_id].direct_content_anchors.append(
                        PhysicalAnchor(
                            block_id=block.id,
                            page_index=block.page_index,
                            role=AnchorRole.OBJECT,
                        )
                    )
                    document_order += 1
                    continue
                for event in iter_block_lines(block, document_order):
                    event_key = (event.block_id, event.char_start, event.char_end)
                    if event_key in toc_events:
                        candidate_kind, candidate_ordinal = self._candidate_shape(event.text)
                        if candidate_kind is not None:
                            diagnostic_sink.append(
                                self._diagnostic(
                                    "table_of_contents_entry_rejected",
                                    event,
                                    candidate_kind,
                                    candidate_ordinal,
                                )
                            )
                        by_id[current_id].direct_content_anchors.append(
                            self._text_anchor(event, AnchorRole.BODY)
                        )
                        pending_title_id = None
                        continue
                    detection = self._detect(event, active, generic_stack)
                    if detection is None and pending_title_id is not None:
                        if _heading_like(event.text, event.block_kind):
                            title_node = by_id[pending_title_id]
                            title_node.title = normalize_detection_text(event.text)
                            title_node.heading_anchors.append(
                                self._text_anchor(event, AnchorRole.TITLE)
                            )
                            current_id = title_node.id
                            pending_title_id = None
                            continue
                        pending_title_id = None
                    if detection is None:
                        rejected = self._rejected_candidate(event)
                        if rejected is not None:
                            category, candidate_kind, candidate_ordinal = rejected
                            diagnostic_sink.append(
                                self._diagnostic(
                                    category,
                                    event,
                                    candidate_kind,
                                    candidate_ordinal,
                                )
                            )
                        by_id[current_id].direct_content_anchors.append(
                            self._text_anchor(event, AnchorRole.BODY)
                        )
                        continue

                    parent_id, generic_level = self._select_parent(
                        detection, root.id, active, generic_stack
                    )
                    parent = by_id[parent_id]
                    segment_name = (
                        "generic"
                        if detection.kind == StructuralNodeKind.GENERIC_SECTION
                        else detection.kind.value
                    )
                    segment_value = detection.ordinal_key or "unnumbered"
                    base_path = f"{segment_name}:{segment_value}"
                    if parent.kind != StructuralNodeKind.DOCUMENT:
                        base_path = f"{parent.canonical_path}/{base_path}"
                    if base_path in used_paths:
                        diagnostic_sink.append(
                            self._diagnostic(
                                "duplicate_structural_key_rejected",
                                event,
                                detection.kind,
                                detection.ordinal_raw,
                                canonical_key=base_path,
                            )
                        )
                        by_id[current_id].direct_content_anchors.append(
                            self._text_anchor(event, AnchorRole.BODY)
                        )
                        pending_title_id = None
                        continue
                    canonical_path = base_path
                    used_paths.add(canonical_path)
                    parent_is_legal_context = parent.kind in {
                        StructuralNodeKind.ARTICLE,
                        StructuralNodeKind.CLAUSE,
                    } or any(
                        segment.startswith(("article:", "clause:"))
                        for segment in parent.canonical_path.split("/")
                    )
                    self._reset_state(
                        detection,
                        active,
                        generic_stack,
                        generic_parent_is_legal=parent_is_legal_context,
                    )
                    node_id = f"node-{len(drafts):06d}"
                    marker_end = event.char_start + detection.marker_end
                    remainder = event.text[detection.marker_end :]
                    has_remainder = bool(normalize_detection_text(remainder))
                    has_inline_title = has_remainder and detection.kind not in {
                        StructuralNodeKind.CLAUSE,
                        StructuralNodeKind.POINT,
                    }
                    marker_anchor_end = marker_end if has_remainder else event.char_end
                    marker_text = event.text[: detection.marker_end].strip()
                    node = _NodeDraft(
                        id=node_id,
                        kind=detection.kind,
                        parent_id=parent_id,
                        depth=parent.depth + 1,
                        ordinal_raw=detection.ordinal_raw,
                        ordinal_key=detection.ordinal_key,
                        marker_text=marker_text,
                        title=normalize_detection_text(remainder) if has_inline_title else None,
                        recognition_evidence=RecognitionEvidence(
                            rule_id=detection.rule_id,
                            recognition_method=detection.method,
                            matched_text=marker_text,
                        ),
                        canonical_path=canonical_path,
                        heading_anchors=[
                            PhysicalAnchor(
                                block_id=event.block_id,
                                page_index=event.page_index,
                                role=AnchorRole.MARKER,
                                char_start=event.char_start,
                                char_end=marker_anchor_end,
                            )
                        ],
                    )
                    if has_inline_title:
                        node.heading_anchors.append(
                            PhysicalAnchor(
                                block_id=event.block_id,
                                page_index=event.page_index,
                                role=AnchorRole.TITLE,
                                char_start=marker_end,
                                char_end=event.char_end,
                            )
                        )
                    elif has_remainder:
                        node.direct_content_anchors.append(
                            PhysicalAnchor(
                                block_id=event.block_id,
                                page_index=event.page_index,
                                role=AnchorRole.BODY,
                                char_start=marker_end,
                                char_end=event.char_end,
                            )
                        )
                    drafts.append(node)
                    by_id[node.id] = node
                    active[node.kind] = node.id
                    if node.kind == StructuralNodeKind.GENERIC_SECTION:
                        assert generic_level is not None
                        generic_stack.append(
                            (
                                generic_level,
                                node.id,
                                detection.ordinal_key or "",
                                detection.method,
                                parent_is_legal_context,
                            )
                        )
                    current_id = node.id
                    pending_title_id = (
                        node.id
                        if not has_inline_title
                        and node.kind
                        in {
                            StructuralNodeKind.PART,
                            StructuralNodeKind.CHAPTER,
                            StructuralNodeKind.SECTION,
                            StructuralNodeKind.SUBSECTION,
                            StructuralNodeKind.APPENDIX,
                        }
                        else None
                    )
                document_order += 1

        return StructuralDocument(
            structural_ir_version=1,
            profile=self.profile,
            source_physical_ir_version=2,
            source_physical_ir_sha256=physical_sha,
            document_id=physical.document_id,
            version_id=physical.version_id,
            source_artifact_sha256=physical.source_artifact_sha256,
            nodes=tuple(draft.freeze() for draft in drafts),
        )

    @staticmethod
    def _text_anchor(event: PhysicalLineEvent, role: AnchorRole) -> PhysicalAnchor:
        return PhysicalAnchor(
            block_id=event.block_id,
            page_index=event.page_index,
            role=role,
            char_start=event.char_start,
            char_end=event.char_end,
        )

    @staticmethod
    def _diagnostic(
        category: str,
        event: PhysicalLineEvent,
        candidate_kind: StructuralNodeKind | None = None,
        candidate_ordinal: str | None = None,
        *,
        canonical_key: str | None = None,
    ) -> StructuralDiagnostic:
        return StructuralDiagnostic(
            category=category,
            page_index=event.page_index,
            block_id=event.block_id,
            excerpt=normalize_detection_text(event.text)[:240],
            candidate_kind=candidate_kind,
            candidate_ordinal=candidate_ordinal,
            canonical_key=canonical_key,
        )

    @staticmethod
    def _candidate_shape(text: str) -> tuple[StructuralNodeKind | None, str | None]:
        candidate = text.replace("\u00a0", " ").rstrip("\r\n")
        for kind, pattern, _ in _FORMAL_PATTERNS:
            match = pattern.match(candidate)
            if match is not None:
                return kind, match.groupdict().get("ordinal")
        for kind, pattern in (
            (StructuralNodeKind.CLAUSE, _CLAUSE),
            (StructuralNodeKind.POINT, _POINT),
            (StructuralNodeKind.GENERIC_SECTION, _GENERIC_ROMAN),
            (StructuralNodeKind.GENERIC_SECTION, _GENERIC_DECIMAL),
        ):
            match = pattern.match(candidate)
            if match is not None:
                return kind, match.group("ordinal")
        return None, None

    @staticmethod
    def _toc_events(physical: PhysicalDocumentV1) -> frozenset[tuple[str, int, int]]:
        """Identify a bounded explicit TOC range or isolated strong leader entries."""
        events_by_page: dict[int, list[PhysicalLineEvent]] = {}
        for event in physical_text_events(physical):
            events_by_page.setdefault(event.page_index, []).append(event)
        result: set[tuple[str, int, int]] = set()
        for events in events_by_page.values():
            normalized = [normalize_detection_text(event.text) for event in events]
            heading_indices = [
                index for index, text in enumerate(normalized) if _TOC_HEADING.fullmatch(text)
            ]
            leader_indices = [
                index
                for index, text in enumerate(normalized)
                if _TOC_LEADER_ENTRY.search(text) is not None
            ]
            if heading_indices:
                start = heading_indices[0]
                later_leaders = [index for index in leader_indices if index >= start]
                if later_leaders:
                    for event in events[start : max(later_leaders) + 1]:
                        result.add((event.block_id, event.char_start, event.char_end))
                    continue
            for index in leader_indices:
                event = events[index]
                candidate_kind, _ = VietnameseStructuralExtractor._candidate_shape(event.text)
                if candidate_kind is not None:
                    result.add((event.block_id, event.char_start, event.char_end))
        return frozenset(result)

    @staticmethod
    def _formal_key(kind: StructuralNodeKind, raw: str | None) -> str | None:
        if raw is None:
            return None
        if kind == StructuralNodeKind.ARTICLE:
            return article_ordinal_key(raw)
        return formal_container_ordinal_key(raw)

    @classmethod
    def _rejected_candidate(
        cls, event: PhysicalLineEvent
    ) -> tuple[str, StructuralNodeKind, str | None] | None:
        candidate = event.text.replace("\u00a0", " ").rstrip("\r\n")
        for kind, pattern, _ in _FORMAL_PATTERNS:
            match = pattern.match(candidate)
            if match is None:
                continue
            raw = match.groupdict().get("ordinal")
            if raw is not None and cls._formal_key(kind, raw) is None:
                return "invalid_roman_ordinal_rejected", kind, raw
            remainder = candidate[match.end() :]
            if not _formal_remainder_is_heading(
                kind, remainder, match.groupdict().get("separator")
            ):
                return "line_start_prose_reference_rejected", kind, raw
        decimal = _GENERIC_DECIMAL.match(candidate)
        if decimal is not None:
            return (
                "generic_heading_evidence_rejected",
                StructuralNodeKind.GENERIC_SECTION,
                decimal.group("ordinal"),
            )
        roman = _GENERIC_ROMAN.match(candidate)
        if roman is not None and roman_ordinal_key(roman.group("ordinal")) is None:
            return (
                "invalid_roman_ordinal_rejected",
                StructuralNodeKind.GENERIC_SECTION,
                roman.group("ordinal"),
            )
        return None

    def _detect(
        self,
        event: PhysicalLineEvent,
        active: dict[StructuralNodeKind, str],
        generic_stack: list[tuple[int, str, str, RecognitionMethod, bool]],
    ) -> _Detection | None:
        candidate = event.text.replace("\u00a0", " ").rstrip("\r\n")
        nonlegal_generic_context = bool(generic_stack and not generic_stack[-1][4])
        for kind, pattern, rule_id in _FORMAL_PATTERNS:
            match = pattern.match(candidate)
            if match is not None:
                raw = match.groupdict().get("ordinal")
                key = self._formal_key(kind, raw)
                if raw is not None and key is None:
                    continue
                remainder = candidate[match.end() :]
                if not _formal_remainder_is_heading(
                    kind, remainder, match.groupdict().get("separator")
                ):
                    continue
                if (
                    nonlegal_generic_context
                    and event.block_kind != BlockKindV1.TITLE
                    and kind in {StructuralNodeKind.PART, StructuralNodeKind.CHAPTER}
                ):
                    continue
                return _Detection(
                    kind=kind,
                    ordinal_raw=raw,
                    ordinal_key=key,
                    marker_end=match.end(),
                    method=RecognitionMethod.EXPLICIT_LEGAL_MARKER,
                    rule_id=rule_id,
                )

        if StructuralNodeKind.CLAUSE in active and not nonlegal_generic_context:
            point_match = _POINT.match(candidate)
            if point_match is not None:
                raw = point_match.group("ordinal")
                return _Detection(
                    kind=StructuralNodeKind.POINT,
                    ordinal_raw=raw,
                    ordinal_key=point_ordinal_key(raw),
                    marker_end=point_match.end(),
                    method=RecognitionMethod.CONTEXTUAL_POINT,
                    rule_id="VI_CONTEXTUAL_POINT_V1",
                )
        if StructuralNodeKind.ARTICLE in active and not nonlegal_generic_context:
            clause_match = _CLAUSE.match(candidate)
            if clause_match is not None:
                raw = clause_match.group("ordinal")
                return _Detection(
                    kind=StructuralNodeKind.CLAUSE,
                    ordinal_raw=raw,
                    ordinal_key=clause_ordinal_key(raw),
                    marker_end=clause_match.end(),
                    method=RecognitionMethod.CONTEXTUAL_CLAUSE,
                    rule_id="VI_CONTEXTUAL_CLAUSE_V1",
                )

        roman_match = _GENERIC_ROMAN.match(candidate)
        if roman_match is not None and _heading_like(candidate, event.block_kind):
            raw = roman_match.group("ordinal")
            key = roman_ordinal_key(raw)
            if key is None:
                return None
            return _Detection(
                kind=StructuralNodeKind.GENERIC_SECTION,
                ordinal_raw=raw,
                ordinal_key=key,
                marker_end=roman_match.end(),
                method=RecognitionMethod.GENERIC_ROMAN_HEADING,
                rule_id="VI_GENERIC_ROMAN_HEADING_V1",
                generic_level=1,
            )

        letter_match = _GENERIC_LETTER.match(candidate)
        if (
            letter_match is not None
            and nonlegal_generic_context
            and _starts_with_uppercase_word(candidate[letter_match.end() :])
        ):
            raw = letter_match.group("ordinal")
            parent_level = next(
                level
                for level, _, _, method, _ in reversed(generic_stack)
                if method != RecognitionMethod.GENERIC_LETTER_HEADING
            )
            return _Detection(
                kind=StructuralNodeKind.GENERIC_SECTION,
                ordinal_raw=raw,
                ordinal_key=generic_letter_ordinal_key(raw),
                marker_end=letter_match.end(),
                method=RecognitionMethod.GENERIC_LETTER_HEADING,
                rule_id="VI_GENERIC_LETTER_HEADING_V1",
                generic_level=parent_level + 1,
            )

        decimal_match = _GENERIC_DECIMAL.match(candidate)
        if decimal_match is None:
            return None
        raw = decimal_match.group("ordinal")
        components = raw.count(".") + 1
        has_roman_parent = any(
            method == RecognitionMethod.GENERIC_ROMAN_HEADING
            for _, _, _, method, _ in generic_stack
        )
        contextual_outline_supported = nonlegal_generic_context and _starts_with_uppercase_word(
            candidate[decimal_match.end() :]
        )
        simple_supported = contextual_outline_supported or _heading_like(
            candidate, event.block_kind
        )
        prefix = raw.rsplit(".", maxsplit=1)[0] if components > 1 else None
        parent_prefix_active = prefix is not None and any(
            key == prefix for _, _, key, _, _ in generic_stack
        )
        multi_supported = components > 1 and (
            _heading_like(candidate, event.block_kind)
            or (
                parent_prefix_active
                and _heading_like(candidate, event.block_kind, allow_short=True)
            )
        )
        if not simple_supported and not multi_supported:
            return None
        return _Detection(
            kind=StructuralNodeKind.GENERIC_SECTION,
            ordinal_raw=raw,
            ordinal_key=generic_decimal_ordinal_key(raw),
            marker_end=decimal_match.end(),
            method=RecognitionMethod.GENERIC_DECIMAL_HEADING,
            rule_id="VI_GENERIC_DECIMAL_HEADING_V1",
            generic_level=components + (1 if has_roman_parent else 0),
        )

    @staticmethod
    def _select_parent(
        detection: _Detection,
        root_id: str,
        active: dict[StructuralNodeKind, str],
        generic_stack: list[tuple[int, str, str, RecognitionMethod, bool]],
    ) -> tuple[str, int | None]:
        kind = detection.kind
        if kind == StructuralNodeKind.APPENDIX:
            return root_id, None
        if kind == StructuralNodeKind.PART:
            return active.get(StructuralNodeKind.APPENDIX, root_id), None
        if kind == StructuralNodeKind.CHAPTER:
            for parent_kind in (StructuralNodeKind.PART, StructuralNodeKind.APPENDIX):
                if parent_kind in active:
                    return active[parent_kind], None
            return root_id, None
        if kind == StructuralNodeKind.SECTION:
            for parent_kind in (
                StructuralNodeKind.CHAPTER,
                StructuralNodeKind.PART,
                StructuralNodeKind.APPENDIX,
            ):
                if parent_kind in active:
                    return active[parent_kind], None
            return root_id, None
        if kind == StructuralNodeKind.SUBSECTION:
            for parent_kind in (
                StructuralNodeKind.SECTION,
                StructuralNodeKind.CHAPTER,
                StructuralNodeKind.PART,
                StructuralNodeKind.APPENDIX,
            ):
                if parent_kind in active:
                    return active[parent_kind], None
            return root_id, None
        if kind == StructuralNodeKind.ARTICLE:
            for parent_kind in (
                StructuralNodeKind.SUBSECTION,
                StructuralNodeKind.SECTION,
                StructuralNodeKind.CHAPTER,
                StructuralNodeKind.PART,
                StructuralNodeKind.APPENDIX,
            ):
                if parent_kind in active:
                    return active[parent_kind], None
            return root_id, None
        if kind == StructuralNodeKind.CLAUSE:
            return active[StructuralNodeKind.ARTICLE], None
        if kind == StructuralNodeKind.POINT:
            return active[StructuralNodeKind.CLAUSE], None

        level = detection.generic_level or 1
        if detection.ordinal_key and "." in detection.ordinal_key:
            prefix = detection.ordinal_key.rsplit(".", maxsplit=1)[0]
            for _, node_id, key, _, _ in reversed(generic_stack):
                if key == prefix:
                    return node_id, level
        for candidate_level, node_id, _, _, _ in reversed(generic_stack):
            if candidate_level < level:
                return node_id, level
        clause_id = active.get(StructuralNodeKind.CLAUSE)
        if clause_id is not None:
            return clause_id, level
        if StructuralNodeKind.ARTICLE in active:
            return active[StructuralNodeKind.ARTICLE], level
        return active.get(StructuralNodeKind.APPENDIX, root_id), level

    @staticmethod
    def _reset_state(
        detection: _Detection,
        active: dict[StructuralNodeKind, str],
        generic_stack: list[tuple[int, str, str, RecognitionMethod, bool]],
        *,
        generic_parent_is_legal: bool,
    ) -> None:
        kind = detection.kind
        order = [
            StructuralNodeKind.PART,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.SECTION,
            StructuralNodeKind.SUBSECTION,
            StructuralNodeKind.ARTICLE,
            StructuralNodeKind.CLAUSE,
            StructuralNodeKind.POINT,
        ]
        if kind == StructuralNodeKind.APPENDIX:
            active.clear()
            generic_stack.clear()
            return
        if kind == StructuralNodeKind.GENERIC_SECTION:
            level = detection.generic_level or 1
            generic_stack[:] = [item for item in generic_stack if item[0] < level]
            if not generic_parent_is_legal:
                for legal_kind in (
                    StructuralNodeKind.ARTICLE,
                    StructuralNodeKind.CLAUSE,
                    StructuralNodeKind.POINT,
                ):
                    active.pop(legal_kind, None)
            return
        generic_stack.clear()
        start = order.index(kind)
        for descendant in order[start:]:
            active.pop(descendant, None)


__all__ = [
    "StructuralDiagnostic",
    "VietnameseStructuralExtractor",
    "article_ordinal_key",
    "clause_ordinal_key",
    "formal_container_ordinal_key",
    "generic_decimal_ordinal_key",
    "generic_letter_ordinal_key",
    "normalize_detection_text",
    "point_ordinal_key",
    "roman_ordinal_key",
]
