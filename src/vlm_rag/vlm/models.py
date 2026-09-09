"""Provider-neutral selective-VLM request, response, and observation contracts."""

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
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
from vlm_rag.semantic_ir.models import SemanticProvenanceKind, VisualSemanticAnchor


class VLMModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


class VLMTaskType(StrEnum):
    REGION_TRANSCRIPTION = "region_transcription"
    OCR_RECOVERY = "ocr_recovery"
    TABLE_TEXT_RECOVERY = "table_text_recovery"
    TABLE_HEADER_RECOVERY = "table_header_recovery"
    FIGURE_OR_IMAGE_OBSERVATION = "figure_or_image_observation"


class SelectionReason(StrEnum):
    FIGURE_OR_IMAGE = "figure_or_image"
    TABLE_MISSING_STRUCTURE = "table_missing_structure"
    OCR_LOW_CONFIDENCE = "ocr_low_confidence"
    OCR_CORRUPTION_SIGNAL = "ocr_corruption_signal"
    UNKNOWN_TEXT_EXTRACTION = "unknown_text_extraction"
    EMPTY_OR_TEXT_DEFICIENT_VISUAL_REGION = "empty_or_text_deficient_visual_region"


class NonSelectionReason(StrEnum):
    PAGE_BUDGET_EXHAUSTED = "page_budget_exhausted"
    DOCUMENT_BUDGET_EXHAUSTED = "document_budget_exhausted"
    LOWER_PRIORITY = "lower_priority"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED_VISUAL_SOURCE = "unsupported_visual_source"
    AMBIGUOUS_STRUCTURAL_OWNER = "ambiguous_structural_owner"


class RenderSpecification(VLMModel):
    dpi: PositiveInt = 144
    color_mode: Literal["rgb"] = "rgb"
    image_format: Literal["png"] = "png"


class RetainedVisualAsset(VLMModel):
    relative_path: str = Field(min_length=1)
    sha256: Sha256Digest
    byte_size: PositiveInt
    media_type: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def validate_safe_path(cls, value: str) -> str:
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            "\\" in value
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ValueError("visual asset path must be a safe relative POSIX path")
        return value


class VisualEvidenceRequest(VLMModel):
    request_id: str = Field(min_length=1)
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    source_physical_ir_sha256: Sha256Digest | None = None
    source_structural_ir_sha256: Sha256Digest | None = None
    structural_node_id: str | None = Field(default=None, min_length=1)
    structural_canonical_path: str | None = Field(default=None, min_length=1)
    page_index: NonNegativeInt
    bbox: BoundingBox
    source_block_ids: tuple[str, ...]
    task_type: VLMTaskType
    selection_reason: SelectionReason
    priority: NonNegativeInt
    render_specification: RenderSpecification = Field(default_factory=RenderSpecification)
    retained_visual_asset: RetainedVisualAsset | None = None

    @field_validator("source_block_ids", mode="before")
    @classmethod
    def coerce_source_block_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("task_type", mode="before")
    @classmethod
    def coerce_task_type(cls, value: object) -> object:
        return VLMTaskType(value) if isinstance(value, str) else value

    @field_validator("selection_reason", mode="before")
    @classmethod
    def coerce_selection_reason(cls, value: object) -> object:
        return SelectionReason(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_blocks(self) -> Self:
        if not self.source_block_ids or len(set(self.source_block_ids)) != len(
            self.source_block_ids
        ):
            raise ValueError("request requires unique source_block_ids")
        context = (
            self.source_physical_ir_sha256,
            self.source_structural_ir_sha256,
            self.structural_node_id,
            self.structural_canonical_path,
        )
        if any(value is not None for value in context) and not all(
            value is not None for value in context
        ):
            raise ValueError("structural request context fields must be supplied together")
        return self


class VLMNonSelection(VLMModel):
    request_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    source_block_ids: tuple[str, ...]
    task_type: VLMTaskType
    selection_reason: SelectionReason
    priority: NonNegativeInt
    reason: NonSelectionReason

    @field_validator("source_block_ids", mode="before")
    @classmethod
    def coerce_source_block_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("task_type", mode="before")
    @classmethod
    def coerce_task_type(cls, value: object) -> object:
        return VLMTaskType(value) if isinstance(value, str) else value

    @field_validator("selection_reason", mode="before")
    @classmethod
    def coerce_selection_reason(cls, value: object) -> object:
        return SelectionReason(value) if isinstance(value, str) else value

    @field_validator("reason", mode="before")
    @classmethod
    def coerce_reason(cls, value: object) -> object:
        return NonSelectionReason(value) if isinstance(value, str) else value


class VLMSelectionResult(VLMModel):
    requests: tuple[VisualEvidenceRequest, ...] = Field(default_factory=tuple)
    non_selections: tuple[VLMNonSelection, ...] = Field(default_factory=tuple)

    @field_validator("requests", "non_selections", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


ParameterValue = str | int | float | bool


class VLMRequestRecord(VLMModel):
    request: VisualEvidenceRequest
    model_id: str = Field(min_length=1)
    provider_protocol: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    image_sha256: Sha256Digest
    image_byte_size: PositiveInt
    parameters: dict[str, ParameterValue] = Field(default_factory=dict)
    timestamp: AwareDatetime | None = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("parameters")
    @classmethod
    def reject_secret_parameters(
        cls, value: dict[str, ParameterValue]
    ) -> dict[str, ParameterValue]:
        if any(token in key.casefold() for key in value for token in ("key", "token", "secret")):
            raise ValueError("request parameters cannot contain secrets")
        return value


class RawVLMResponse(VLMModel):
    request_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    provider_protocol: str = Field(min_length=1)
    raw_response: str
    raw_response_sha256: Sha256Digest
    finish_state: Literal["completed", "failed"]
    error: str | None = None
    latency_ms: NonNegativeInt | None = None

    @model_validator(mode="after")
    def validate_raw_response(self) -> Self:
        observed = hashlib.sha256(self.raw_response.encode("utf-8")).hexdigest()
        if observed != self.raw_response_sha256:
            raise ValueError("raw response SHA-256 mismatch")
        if (self.finish_state == "failed") != (self.error is not None):
            raise ValueError("error is required exactly for failed responses")
        return self


class VLMObservationRecord(VLMModel):
    observation_schema_version: Literal[2] = 2
    request_id: str = Field(min_length=1)
    task_type: VLMTaskType
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    source_physical_ir_sha256: Sha256Digest
    source_structural_ir_sha256: Sha256Digest
    structural_node_id: str = Field(min_length=1)
    structural_canonical_path: str = Field(min_length=1)
    request_record_sha256: Sha256Digest
    model_id: str = Field(min_length=1)
    provider_protocol: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    raw_response_sha256: Sha256Digest
    image_sha256: Sha256Digest
    source_block_ids: tuple[str, ...]
    transcription: str | None = Field(default=None, min_length=1)
    table_rows: tuple[tuple[str, ...], ...] | None = None
    description: str | None = Field(default=None, min_length=1)
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

    @field_validator("task_type", mode="before")
    @classmethod
    def coerce_task_type(cls, value: object) -> object:
        return VLMTaskType(value) if isinstance(value, str) else value

    @field_validator("provenance", mode="before")
    @classmethod
    def coerce_provenance(cls, value: object) -> object:
        return SemanticProvenanceKind(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_task_payload(self) -> Self:
        if not self.source_block_ids:
            raise ValueError("VLM observation requires source_block_ids")
        payloads = sum(
            value is not None for value in (self.transcription, self.table_rows, self.description)
        )
        if payloads != 1:
            raise ValueError("VLM observation requires exactly one task payload")
        if self.task_type in {VLMTaskType.REGION_TRANSCRIPTION, VLMTaskType.OCR_RECOVERY}:
            if (
                self.transcription is None
                or self.provenance != SemanticProvenanceKind.VLM_TRANSCRIPTION
            ):
                raise ValueError("transcription task requires VLM_TRANSCRIPTION")
        elif self.task_type in {
            VLMTaskType.TABLE_TEXT_RECOVERY,
            VLMTaskType.TABLE_HEADER_RECOVERY,
        }:
            if (
                self.table_rows is None
                or self.provenance != SemanticProvenanceKind.VLM_TABLE_RECOVERY
            ):
                raise ValueError("table task requires VLM_TABLE_RECOVERY")
        elif (
            self.description is None
            or self.provenance != SemanticProvenanceKind.VLM_VISUAL_OBSERVATION
        ):
            raise ValueError("figure task requires VLM_VISUAL_OBSERVATION")
        return self


def canonical_request_sha256(request: VisualEvidenceRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_request_record_sha256(record: VLMRequestRecord) -> str:
    """Hash deterministic request/model/prompt/image/parameter identity, excluding timestamp."""
    payload = json.dumps(
        record.model_dump(mode="json", exclude={"timestamp"}),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "NonSelectionReason",
    "ParameterValue",
    "RawVLMResponse",
    "RenderSpecification",
    "RetainedVisualAsset",
    "SelectionReason",
    "VLMNonSelection",
    "VLMObservationRecord",
    "VLMRequestRecord",
    "VLMSelectionResult",
    "VLMTaskType",
    "VisualEvidenceRequest",
    "canonical_request_record_sha256",
    "canonical_request_sha256",
]
