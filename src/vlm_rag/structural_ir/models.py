"""Strict parser-independent models for Vietnamese Structural IR v1."""

import re
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator, model_validator

from vlm_rag.registry.models import Identifier, Sha256Digest, VersionIdentifier
from vlm_rag.structural_ir.ordinals import OrdinalSystem, ordinal_key_for_system


class StructuralModel(BaseModel):
    """Shared strict and immutable model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


class StructuralProfile(StrEnum):
    """Controlled domain profile implemented by the deterministic extractor."""

    VI_LEGAL_PLANNING_V1 = "vi_legal_planning_v1"


class StructuralNodeKind(StrEnum):
    """Physical-content organization kinds, without semantic legal meaning."""

    DOCUMENT = "document"
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    ARTICLE = "article"
    CLAUSE = "clause"
    POINT = "point"
    APPENDIX = "appendix"
    GENERIC_SECTION = "generic_section"


class AnchorRole(StrEnum):
    """Role played by an exact span or whole object within a structural node."""

    MARKER = "marker"
    TITLE = "title"
    BODY = "body"
    OBJECT = "object"


class RecognitionMethod(StrEnum):
    """Deterministic rule families that may create a structural node."""

    DOCUMENT_ROOT = "document_root"
    EXPLICIT_LEGAL_MARKER = "explicit_legal_marker"
    CONTEXTUAL_CLAUSE = "contextual_clause"
    CONTEXTUAL_POINT = "contextual_point"
    GENERIC_ROMAN_HEADING = "generic_roman_heading"
    GENERIC_DECIMAL_HEADING = "generic_decimal_heading"
    GENERIC_LETTER_HEADING = "generic_letter_heading"


class PhysicalAnchor(StructuralModel):
    """Exact link from Structural IR back to a Physical IR v1 block."""

    block_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    role: AnchorRole
    char_start: NonNegativeInt | None = None
    char_end: NonNegativeInt | None = None

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return AnchorRole(value)
            except ValueError:
                return value
        return value

    @model_validator(mode="after")
    def validate_span_shape(self) -> Self:
        """Require text spans for text roles and whole-block anchors for objects."""
        if self.role == AnchorRole.OBJECT:
            if self.char_start is not None or self.char_end is not None:
                raise ValueError("OBJECT anchors must have null character offsets")
        elif self.char_start is None or self.char_end is None:
            raise ValueError("text anchors require char_start and char_end")
        elif self.char_start >= self.char_end:
            raise ValueError("text anchor requires char_start < char_end")
        return self


class RecognitionEvidence(StructuralModel):
    """Exact deterministic rule evidence used to create one node."""

    rule_id: str = Field(min_length=1)
    recognition_method: RecognitionMethod
    matched_text: str = Field(min_length=1)

    @field_validator("recognition_method", mode="before")
    @classmethod
    def coerce_method(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return RecognitionMethod(value)
            except ValueError:
                return value
        return value


class StructuralNode(StructuralModel):
    """One parser-neutral unit in the recovered document hierarchy."""

    id: str = Field(min_length=1)
    kind: StructuralNodeKind
    parent_id: str | None
    depth: NonNegativeInt
    ordinal_raw: str | None = Field(default=None, min_length=1)
    ordinal_key: str | None = Field(default=None, min_length=1)
    marker_text: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    heading_anchors: tuple[PhysicalAnchor, ...] = Field(default_factory=tuple)
    direct_content_anchors: tuple[PhysicalAnchor, ...] = Field(default_factory=tuple)
    recognition_evidence: RecognitionEvidence
    canonical_path: str = Field(min_length=1)

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return StructuralNodeKind(value)
            except ValueError:
                return value
        return value

    @field_validator("heading_anchors", "direct_content_anchors", mode="before")
    @classmethod
    def coerce_anchor_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_node_shape(self) -> Self:
        """Enforce root and anchor-role invariants independent of a physical document."""
        if self.kind == StructuralNodeKind.DOCUMENT:
            if self.parent_id is not None or self.depth != 0:
                raise ValueError("DOCUMENT root requires parent_id=null and depth=0")
            if any(
                value is not None
                for value in (self.ordinal_raw, self.ordinal_key, self.marker_text, self.title)
            ):
                raise ValueError("DOCUMENT root cannot carry marker, ordinal, or title fields")
            if self.heading_anchors:
                raise ValueError("DOCUMENT root cannot have heading anchors")
            if self.canonical_path != "document":
                raise ValueError("DOCUMENT root canonical_path must be 'document'")
            if self.recognition_evidence.recognition_method != RecognitionMethod.DOCUMENT_ROOT:
                raise ValueError("DOCUMENT root requires document_root recognition evidence")
        else:
            if self.parent_id is None or self.depth == 0:
                raise ValueError("non-root nodes require a parent and positive depth")
            if self.marker_text is None:
                raise ValueError("non-root nodes require marker_text")
            if self.recognition_evidence.recognition_method == RecognitionMethod.DOCUMENT_ROOT:
                raise ValueError("non-root nodes cannot use document_root recognition evidence")
        if (self.ordinal_raw is None) != (self.ordinal_key is None):
            raise ValueError("ordinal_raw and ordinal_key must both be present or both be null")
        ordinal_system = _node_ordinal_system(
            self.kind, self.recognition_evidence.recognition_method
        )
        expected_ordinal = ordinal_key_for_system(ordinal_system, self.ordinal_raw)
        if self.ordinal_key != expected_ordinal:
            raise ValueError(
                "ordinal_key does not match kind-aware semantics for ordinal_raw "
                "and recognition method"
            )
        if any(
            anchor.role not in {AnchorRole.MARKER, AnchorRole.TITLE}
            for anchor in self.heading_anchors
        ):
            raise ValueError("heading_anchors may contain only MARKER or TITLE roles")
        if any(
            anchor.role not in {AnchorRole.BODY, AnchorRole.OBJECT}
            for anchor in self.direct_content_anchors
        ):
            raise ValueError("direct_content_anchors may contain only BODY or OBJECT roles")
        if self.kind != StructuralNodeKind.DOCUMENT:
            segment = self.canonical_path.rsplit("/", maxsplit=1)[-1]
            expected_name = (
                "generic" if self.kind == StructuralNodeKind.GENERIC_SECTION else self.kind.value
            )
            match = re.fullmatch(rf"{expected_name}:(?P<value>[^/~]+)", segment)
            if match is None:
                raise ValueError("canonical path segment does not match node kind")
            if self.ordinal_key is not None and match.group("value") != self.ordinal_key:
                raise ValueError("canonical path ordinal does not match ordinal_key")
        return self


def _node_ordinal_system(kind: StructuralNodeKind, method: RecognitionMethod) -> OrdinalSystem:
    if kind == StructuralNodeKind.DOCUMENT:
        return OrdinalSystem.NONE
    if kind == StructuralNodeKind.ARTICLE:
        return OrdinalSystem.ARTICLE
    if kind == StructuralNodeKind.CLAUSE:
        return OrdinalSystem.CLAUSE
    if kind == StructuralNodeKind.POINT:
        return OrdinalSystem.POINT
    if kind == StructuralNodeKind.GENERIC_SECTION:
        generic_systems = {
            RecognitionMethod.GENERIC_ROMAN_HEADING: OrdinalSystem.GENERIC_ROMAN,
            RecognitionMethod.GENERIC_DECIMAL_HEADING: OrdinalSystem.GENERIC_DECIMAL,
            RecognitionMethod.GENERIC_LETTER_HEADING: OrdinalSystem.GENERIC_LETTER,
        }
        try:
            return generic_systems[method]
        except KeyError as error:
            raise ValueError(
                "GENERIC_SECTION requires an explicit generic ordinal method"
            ) from error
    return OrdinalSystem.FORMAL_CONTAINER


_ALLOWED_PARENTS: dict[StructuralNodeKind, frozenset[StructuralNodeKind]] = {
    StructuralNodeKind.PART: frozenset({StructuralNodeKind.DOCUMENT, StructuralNodeKind.APPENDIX}),
    StructuralNodeKind.CHAPTER: frozenset(
        {StructuralNodeKind.DOCUMENT, StructuralNodeKind.PART, StructuralNodeKind.APPENDIX}
    ),
    StructuralNodeKind.SECTION: frozenset(
        {
            StructuralNodeKind.DOCUMENT,
            StructuralNodeKind.PART,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.APPENDIX,
        }
    ),
    StructuralNodeKind.SUBSECTION: frozenset(
        {
            StructuralNodeKind.DOCUMENT,
            StructuralNodeKind.PART,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.SECTION,
            StructuralNodeKind.APPENDIX,
        }
    ),
    StructuralNodeKind.ARTICLE: frozenset(
        {
            StructuralNodeKind.DOCUMENT,
            StructuralNodeKind.PART,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.SECTION,
            StructuralNodeKind.SUBSECTION,
            StructuralNodeKind.APPENDIX,
        }
    ),
    StructuralNodeKind.CLAUSE: frozenset({StructuralNodeKind.ARTICLE}),
    StructuralNodeKind.POINT: frozenset({StructuralNodeKind.CLAUSE}),
    StructuralNodeKind.APPENDIX: frozenset({StructuralNodeKind.DOCUMENT}),
    StructuralNodeKind.GENERIC_SECTION: frozenset(
        {
            StructuralNodeKind.DOCUMENT,
            StructuralNodeKind.APPENDIX,
            StructuralNodeKind.ARTICLE,
            StructuralNodeKind.CLAUSE,
            StructuralNodeKind.GENERIC_SECTION,
        }
    ),
}


class StructuralDocument(StructuralModel):
    """Structural IR v1 bound cryptographically to one Physical IR v1 input."""

    structural_ir_version: Literal[1] = 1
    profile: StructuralProfile = StructuralProfile.VI_LEGAL_PLANNING_V1
    source_physical_ir_version: Literal[2] = 2
    source_physical_ir_sha256: Sha256Digest
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    nodes: tuple[StructuralNode, ...]

    @field_validator("profile", mode="before")
    @classmethod
    def coerce_profile(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return StructuralProfile(value)
            except ValueError:
                return value
        return value

    @field_validator("nodes", mode="before")
    @classmethod
    def coerce_node_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_hierarchy(self) -> Self:
        """Enforce one ordered root, unique identity/path, and legal parent relationships."""
        if not self.nodes or self.nodes[0].kind != StructuralNodeKind.DOCUMENT:
            raise ValueError("nodes must begin with exactly one DOCUMENT root")
        if sum(node.kind == StructuralNodeKind.DOCUMENT for node in self.nodes) != 1:
            raise ValueError("StructuralDocument requires exactly one DOCUMENT root")
        by_id: dict[str, StructuralNode] = {}
        paths: set[str] = set()
        for node in self.nodes:
            if node.id in by_id:
                raise ValueError(f"duplicate structural node id {node.id!r}")
            if node.canonical_path in paths:
                raise ValueError(f"duplicate canonical path {node.canonical_path!r}")
            if node.kind != StructuralNodeKind.DOCUMENT:
                parent = by_id.get(node.parent_id or "")
                if parent is None:
                    raise ValueError(f"parent must exist and precede child {node.id!r}")
                if node.depth != parent.depth + 1:
                    raise ValueError(f"wrong depth for node {node.id!r}")
                if parent.kind not in _ALLOWED_PARENTS[node.kind]:
                    raise ValueError(f"{node.kind.value} cannot be a child of {parent.kind.value}")
                segment = node.canonical_path.rsplit("/", maxsplit=1)[-1]
                expected_path = (
                    segment
                    if parent.kind == StructuralNodeKind.DOCUMENT
                    else f"{parent.canonical_path}/{segment}"
                )
                if node.canonical_path != expected_path:
                    raise ValueError(
                        f"canonical path does not exactly extend parent for {node.id!r}"
                    )
            by_id[node.id] = node
            paths.add(node.canonical_path)
        return self


__all__ = [
    "AnchorRole",
    "PhysicalAnchor",
    "RecognitionEvidence",
    "RecognitionMethod",
    "StructuralDocument",
    "StructuralModel",
    "StructuralNode",
    "StructuralNodeKind",
    "StructuralProfile",
]
