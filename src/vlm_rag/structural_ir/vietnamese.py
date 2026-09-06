"""Deterministic Vietnamese legal/planning structure extraction from Physical IR v1."""

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

from vlm_rag.physical_ir.models import BlockDisposition
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.v1 import BlockKindV1, PhysicalDocumentV1
from vlm_rag.structural_ir.anchors import PhysicalLineEvent, iter_block_lines
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

_FORMAL_PATTERNS: tuple[tuple[StructuralNodeKind, re.Pattern[str], str], ...] = (
    (
        StructuralNodeKind.SUBSECTION,
        re.compile(
            r"^\s*(?:tiểu\s+mục|tieu\s+muc)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?=\s|[.\-:]|$)\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_SUBSECTION_V1",
    ),
    (
        StructuralNodeKind.APPENDIX,
        re.compile(
            r"^\s*(?:phụ\s+lục|phu\s+luc)(?:\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?=\s|[.\-:]|$))?\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_APPENDIX_V1",
    ),
    (
        StructuralNodeKind.CHAPTER,
        re.compile(
            r"^\s*(?:chương|chuong)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?=\s|[.\-:]|$)\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_CHAPTER_V1",
    ),
    (
        StructuralNodeKind.ARTICLE,
        re.compile(
            r"^\s*(?:điều|dieu)\s+(?P<ordinal>\d+[a-zđ]?)(?=\s|[.\-:]|$)\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_ARTICLE_V1",
    ),
    (
        StructuralNodeKind.PART,
        re.compile(
            r"^\s*(?:phần|phan)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?=\s|[.\-:]|$)\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_PART_V1",
    ),
    (
        StructuralNodeKind.SECTION,
        re.compile(
            r"^\s*(?:mục|muc)\s+(?P<ordinal>[IVXLCDM]+|\d+[a-zđ]?)(?=\s|[.\-:]|$)\s*[.\-:]?",
            re.IGNORECASE,
        ),
        "VI_FORMAL_SECTION_V1",
    ),
)
_CLAUSE = re.compile(r"^\s*(?P<ordinal>\d+[a-zđ]?)\.\s+", re.IGNORECASE)
_POINT = re.compile(r"^\s*(?P<ordinal>[a-zđ])\)\s+", re.IGNORECASE)
_GENERIC_ROMAN = re.compile(r"^\s*(?P<ordinal>[IVXLCDM]+)\.\s+", re.IGNORECASE)
_GENERIC_DECIMAL = re.compile(r"^\s*(?P<ordinal>\d+(?:\.\d+)*)\.(?:\s+|$)", re.IGNORECASE)
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass(frozen=True, slots=True)
class _Detection:
    kind: StructuralNodeKind
    ordinal_raw: str | None
    ordinal_key: str | None
    marker_end: int
    method: RecognitionMethod
    rule_id: str
    generic_level: int | None = None


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


def normalize_detection_text(value: str) -> str:
    """Normalize only a detection copy; stored text and offsets remain untouched."""
    normalized = unicodedata.normalize("NFKC", value).replace("\u00a0", " ")
    return re.sub(r"[\t\f\v ]+", " ", normalized).strip()


def ordinal_key(value: str) -> str:
    """Return a deterministic ordinal key without losing letter suffixes."""
    normalized = normalize_detection_text(value).strip(".:-").casefold()
    upper = normalized.upper()
    if upper and all(character in _ROMAN_VALUES for character in upper):
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
    return normalized


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


class VietnameseStructuralExtractor:
    """Conservative deterministic extractor for the Vietnamese legal/planning profile."""

    profile = StructuralProfile.VI_LEGAL_PLANNING_V1

    def extract(self, physical: PhysicalDocumentV1) -> StructuralDocument:
        """Extract hierarchy and complete content anchors from Physical IR wire version 2."""
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
        generic_stack: list[tuple[int, str, str, RecognitionMethod]] = []
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
                        by_id[current_id].direct_content_anchors.append(
                            self._text_anchor(event, AnchorRole.BODY)
                        )
                        continue

                    parent_id, generic_level = self._select_parent(
                        detection, root.id, active, generic_stack
                    )
                    self._reset_state(detection, active, generic_stack)
                    parent = by_id[parent_id]
                    segment_name = (
                        "generic"
                        if detection.kind == StructuralNodeKind.GENERIC_SECTION
                        else detection.kind.value
                    )
                    segment_value = detection.ordinal_key or str(
                        sum(draft.kind == detection.kind for draft in drafts) + 1
                    )
                    base_path = f"{segment_name}:{segment_value}"
                    if parent.kind != StructuralNodeKind.DOCUMENT:
                        base_path = f"{parent.canonical_path}/{base_path}"
                    canonical_path = self._unique_path(base_path, used_paths)
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

    def _detect(
        self,
        event: PhysicalLineEvent,
        active: dict[StructuralNodeKind, str],
        generic_stack: list[tuple[int, str, str, RecognitionMethod]],
    ) -> _Detection | None:
        candidate = event.text.replace("\u00a0", " ").rstrip("\r\n")
        for kind, pattern, rule_id in _FORMAL_PATTERNS:
            match = pattern.match(candidate)
            if match is not None:
                raw = match.groupdict().get("ordinal")
                return _Detection(
                    kind=kind,
                    ordinal_raw=raw,
                    ordinal_key=ordinal_key(raw) if raw else None,
                    marker_end=match.end(),
                    method=RecognitionMethod.EXPLICIT_LEGAL_MARKER,
                    rule_id=rule_id,
                )

        if StructuralNodeKind.CLAUSE in active:
            point_match = _POINT.match(candidate)
            if point_match is not None:
                raw = point_match.group("ordinal")
                return _Detection(
                    kind=StructuralNodeKind.POINT,
                    ordinal_raw=raw,
                    ordinal_key=ordinal_key(raw),
                    marker_end=point_match.end(),
                    method=RecognitionMethod.CONTEXTUAL_POINT,
                    rule_id="VI_CONTEXTUAL_POINT_V1",
                )
        if StructuralNodeKind.ARTICLE in active:
            clause_match = _CLAUSE.match(candidate)
            if clause_match is not None:
                raw = clause_match.group("ordinal")
                return _Detection(
                    kind=StructuralNodeKind.CLAUSE,
                    ordinal_raw=raw,
                    ordinal_key=ordinal_key(raw),
                    marker_end=clause_match.end(),
                    method=RecognitionMethod.CONTEXTUAL_CLAUSE,
                    rule_id="VI_CONTEXTUAL_CLAUSE_V1",
                )

        roman_match = _GENERIC_ROMAN.match(candidate)
        if roman_match is not None and _heading_like(candidate, event.block_kind):
            raw = roman_match.group("ordinal")
            return _Detection(
                kind=StructuralNodeKind.GENERIC_SECTION,
                ordinal_raw=raw,
                ordinal_key=ordinal_key(raw),
                marker_end=roman_match.end(),
                method=RecognitionMethod.GENERIC_ROMAN_HEADING,
                rule_id="VI_GENERIC_ROMAN_HEADING_V1",
                generic_level=1,
            )

        decimal_match = _GENERIC_DECIMAL.match(candidate)
        if decimal_match is None:
            return None
        raw = decimal_match.group("ordinal")
        components = raw.count(".") + 1
        has_roman_parent = any(
            method == RecognitionMethod.GENERIC_ROMAN_HEADING for _, _, _, method in generic_stack
        )
        simple_supported = event.block_kind == BlockKindV1.TITLE or _heading_like(
            candidate, event.block_kind
        )
        if components == 1 and has_roman_parent:
            simple_supported = simple_supported or _heading_like(
                candidate, event.block_kind, allow_short=True
            )
        multi_supported = components > 1 and _heading_like(
            candidate, event.block_kind, allow_short=True
        )
        if not simple_supported and not multi_supported:
            return None
        return _Detection(
            kind=StructuralNodeKind.GENERIC_SECTION,
            ordinal_raw=raw,
            ordinal_key=ordinal_key(raw),
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
        generic_stack: list[tuple[int, str, str, RecognitionMethod]],
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
            for _, node_id, key, _ in reversed(generic_stack):
                if key == prefix:
                    return node_id, level
        for candidate_level, node_id, _, _ in reversed(generic_stack):
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
        generic_stack: list[tuple[int, str, str, RecognitionMethod]],
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
            return
        generic_stack.clear()
        start = order.index(kind)
        for descendant in order[start:]:
            active.pop(descendant, None)

    @staticmethod
    def _unique_path(base_path: str, used_paths: set[str]) -> str:
        candidate = base_path
        occurrence = 2
        while candidate in used_paths:
            candidate = f"{base_path}~{occurrence}"
            occurrence += 1
        used_paths.add(candidate)
        return candidate


__all__ = ["VietnameseStructuralExtractor", "normalize_detection_text", "ordinal_key"]
