"""Strict parser-independent Domain Semantic IR v1 models."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from vlm_rag.physical_ir.models import BoundingBox
from vlm_rag.registry.models import Identifier, Sha256Digest, VersionIdentifier


class SemanticModel(BaseModel):
    """Shared immutable, strict wire-model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


class SemanticProfile(StrEnum):
    VI_LEGAL_PLANNING_SEMANTIC_V1 = "vi_legal_planning_semantic_v1"


class SemanticMentionKind(StrEnum):
    LEGAL_REFERENCE = "legal_reference"
    DOCUMENT_IDENTIFIER = "document_identifier"
    TEMPORAL_EXPRESSION = "temporal_expression"
    QUANTITY = "quantity"
    ORGANIZATION_OR_AUTHORITY = "organization_or_authority"


class SemanticProvenanceKind(StrEnum):
    RULE_BASED_TEXT = "rule_based_text"
    NORMALIZED_TEXT = "normalized_text"
    VLM_TRANSCRIPTION = "vlm_transcription"
    VLM_TABLE_RECOVERY = "vlm_table_recovery"
    VLM_VISUAL_OBSERVATION = "vlm_visual_observation"


class LegalReferenceScope(StrEnum):
    SELF_DOCUMENT = "self_document"
    EXTERNAL_DOCUMENT = "external_document"
    UNKNOWN = "unknown"


class TextSemanticAnchor(SemanticModel):
    anchor_type: Literal["text"] = "text"
    block_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    char_start: NonNegativeInt
    char_end: PositiveInt

    @model_validator(mode="after")
    def validate_span(self) -> Self:
        if self.char_start >= self.char_end:
            raise ValueError("text semantic anchor requires char_start < char_end")
        return self


class ObjectSemanticAnchor(SemanticModel):
    anchor_type: Literal["object"] = "object"
    block_id: str = Field(min_length=1)
    page_index: NonNegativeInt


class VisualSemanticAnchor(SemanticModel):
    anchor_type: Literal["visual"] = "visual"
    source_artifact_sha256: Sha256Digest
    page_index: NonNegativeInt
    bbox: BoundingBox
    render_or_asset_sha256: Sha256Digest
    byte_size: PositiveInt
    media_type: str = Field(min_length=1)


SemanticEvidenceAnchor = TextSemanticAnchor | ObjectSemanticAnchor | VisualSemanticAnchor


class LegalReferenceComponents(SemanticModel):
    instrument_type: str | None = Field(default=None, min_length=1)
    instrument_number_raw: str | None = Field(default=None, min_length=1)
    instrument_number_normalized: str | None = Field(default=None, min_length=1)
    article: str | None = Field(default=None, min_length=1)
    clause: str | None = Field(default=None, min_length=1)
    point: str | None = Field(default=None, min_length=1)
    scope: LegalReferenceScope

    @field_validator("scope", mode="before")
    @classmethod
    def coerce_scope(cls, value: object) -> object:
        return LegalReferenceScope(value) if isinstance(value, str) else value


class QuantityComponents(SemanticModel):
    raw_value: str = Field(min_length=1)
    normalized_numeric_value: str | None = Field(default=None, min_length=1)
    raw_unit: str = Field(min_length=1)
    normalized_unit: str | None = Field(default=None, min_length=1)


class SemanticMention(SemanticModel):
    id: str = Field(min_length=1)
    kind: SemanticMentionKind
    statement_id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    normalized_value: str | None = Field(default=None, min_length=1)
    evidence_anchors: tuple[TextSemanticAnchor | VisualSemanticAnchor, ...]
    provenance: SemanticProvenanceKind
    legal_reference: LegalReferenceComponents | None = None
    quantity: QuantityComponents | None = None

    @field_validator("evidence_anchors", mode="before")
    @classmethod
    def coerce_evidence_anchors(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, value: object) -> object:
        return SemanticMentionKind(value) if isinstance(value, str) else value

    @field_validator("provenance", mode="before")
    @classmethod
    def coerce_provenance(cls, value: object) -> object:
        return SemanticProvenanceKind(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_kind_details(self) -> Self:
        if not self.evidence_anchors:
            raise ValueError("semantic mention requires evidence")
        if (self.kind == SemanticMentionKind.LEGAL_REFERENCE) != (self.legal_reference is not None):
            raise ValueError("legal_reference details are required exactly for LEGAL_REFERENCE")
        if (self.kind == SemanticMentionKind.QUANTITY) != (self.quantity is not None):
            raise ValueError("quantity details are required exactly for QUANTITY")
        return self


class SemanticStatement(SemanticModel):
    id: str = Field(min_length=1)
    structural_node_id: str = Field(min_length=1)
    structural_canonical_path: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_anchors: tuple[TextSemanticAnchor | VisualSemanticAnchor, ...]
    mention_ids: tuple[str, ...] = Field(default_factory=tuple)
    provenance: SemanticProvenanceKind

    @field_validator("evidence_anchors", "mention_ids", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("provenance", mode="before")
    @classmethod
    def coerce_provenance(cls, value: object) -> object:
        return SemanticProvenanceKind(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        if not self.evidence_anchors:
            raise ValueError("semantic statement requires evidence")
        return self


class VisualObservation(SemanticModel):
    id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    task_type: Literal[
        "region_transcription",
        "ocr_recovery",
        "table_text_recovery",
        "table_header_recovery",
        "figure_or_image_observation",
    ]
    source_block_ids: tuple[str, ...]
    text: str | None = Field(default=None, min_length=1)
    table_rows: tuple[tuple[str, ...], ...] | None = None
    evidence_anchor: VisualSemanticAnchor
    provenance: SemanticProvenanceKind

    @field_validator("source_block_ids", mode="before")
    @classmethod
    def coerce_source_block_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("table_rows", mode="before")
    @classmethod
    def coerce_table_rows(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(tuple(row) if isinstance(row, list) else row for row in value)
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def coerce_provenance(cls, value: object) -> object:
        return SemanticProvenanceKind(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if not self.source_block_ids:
            raise ValueError("visual observation requires source_block_ids")
        if self.text is None and self.table_rows is None:
            raise ValueError("visual observation requires text or table rows")
        if self.provenance not in {
            SemanticProvenanceKind.VLM_TRANSCRIPTION,
            SemanticProvenanceKind.VLM_TABLE_RECOVERY,
            SemanticProvenanceKind.VLM_VISUAL_OBSERVATION,
        }:
            raise ValueError("visual observation requires VLM provenance")
        return self


class SemanticDocument(SemanticModel):
    semantic_ir_version: Literal[1] = 1
    profile: SemanticProfile = SemanticProfile.VI_LEGAL_PLANNING_SEMANTIC_V1
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    source_physical_ir_sha256: Sha256Digest
    source_structural_ir_sha256: Sha256Digest
    statements: tuple[SemanticStatement, ...] = Field(default_factory=tuple)
    mentions: tuple[SemanticMention, ...] = Field(default_factory=tuple)
    visual_observations: tuple[VisualObservation, ...] = Field(default_factory=tuple)

    @field_validator("statements", "mentions", "visual_observations", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("profile", mode="before")
    @classmethod
    def coerce_profile(cls, value: object) -> object:
        return SemanticProfile(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        statement_by_id = {statement.id: statement for statement in self.statements}
        mention_by_id = {mention.id: mention for mention in self.mentions}
        observation_ids = {observation.id for observation in self.visual_observations}
        if len(statement_by_id) != len(self.statements):
            raise ValueError("duplicate semantic statement id")
        if len(mention_by_id) != len(self.mentions):
            raise ValueError("duplicate semantic mention id")
        if len(observation_ids) != len(self.visual_observations):
            raise ValueError("duplicate visual observation id")
        referenced_mentions: list[str] = []
        for statement in self.statements:
            referenced_mentions.extend(statement.mention_ids)
            if len(set(statement.mention_ids)) != len(statement.mention_ids):
                raise ValueError("statement contains duplicate mention_ids")
        if set(referenced_mentions) != set(mention_by_id):
            raise ValueError("statement mention_ids must reference every mention exactly once")
        if len(referenced_mentions) != len(mention_by_id):
            raise ValueError("semantic mention cannot belong to multiple statements")
        for mention in self.mentions:
            if mention.statement_id not in statement_by_id:
                raise ValueError("semantic mention references missing statement")
            if mention.id not in statement_by_id[mention.statement_id].mention_ids:
                raise ValueError("semantic mention ownership is inconsistent")
        return self


__all__ = [
    "LegalReferenceComponents",
    "LegalReferenceScope",
    "ObjectSemanticAnchor",
    "QuantityComponents",
    "SemanticDocument",
    "SemanticEvidenceAnchor",
    "SemanticMention",
    "SemanticMentionKind",
    "SemanticProfile",
    "SemanticProvenanceKind",
    "SemanticStatement",
    "TextSemanticAnchor",
    "VisualObservation",
    "VisualSemanticAnchor",
]
