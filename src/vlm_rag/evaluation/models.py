"""Typed domain models for reference layout annotation and spatial evaluation."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    StringConstraints,
    field_validator,
    model_validator,
)

Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class EvaluationBaseModel(BaseModel):
    """Shared strict, immutable configuration for evaluation records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class RegionKind(StrEnum):
    """Semantic/physical category of an audited region."""

    TEXT = "text"
    HEADING = "heading"
    TABLE = "table"
    LIST_ITEM = "list_item"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    SCANNED_BLOCK = "scanned_block"
    TITLE = "title"
    UNKNOWN = "unknown"


class AnnotatorType(StrEnum):
    """Controlled identity class for an annotation producer."""

    HUMAN_EXPERT = "human_expert"
    AI_VISUAL_AUDIT = "ai_visual_audit"
    SYNTHETIC = "synthetic"


class AnnotationMethod(StrEnum):
    """Controlled method used to establish reference regions."""

    VISUAL_PDF_AUDIT = "visual_pdf_audit"
    VISUAL_PDF_REAUDIT = "visual_pdf_reaudit"
    PARSER_BOOTSTRAPPED = "parser_bootstrapped"
    SYNTHETIC = "synthetic"


class AnnotationBoundingBox(EvaluationBaseModel):
    """Normalized 1000-based bounding box for reference annotations."""

    x0: float = Field(ge=0.0, le=1000.0)
    y0: float = Field(ge=0.0, le=1000.0)
    x1: float = Field(ge=0.0, le=1000.0)
    y1: float = Field(ge=0.0, le=1000.0)
    coordinate_system: Literal["normalized_1000"] = "normalized_1000"

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        """Ensure coordinate ordering is topologically valid and non-zero area."""
        if self.x0 >= self.x1:
            raise ValueError(f"x0 ({self.x0}) cannot exceed or equal x1 ({self.x1})")
        if self.y0 >= self.y1:
            raise ValueError(f"y0 ({self.y0}) cannot exceed or equal y1 ({self.y1})")
        return self


class ReferenceRegion(EvaluationBaseModel):
    """An audited reference layout segment on a document page."""

    id: str = Field(min_length=1)
    reading_order: NonNegativeInt
    kind: RegionKind
    bbox: AnnotationBoundingBox
    text: str = Field(default="")
    heading_level: int | None = Field(default=None)

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return RegionKind(value)
            except ValueError:
                return value
        return value


class AuditedPage(EvaluationBaseModel):
    """Audited reference regions and documented phenomena for one page."""

    page_index: NonNegativeInt
    phenomena: tuple[str, ...] = Field(default_factory=tuple)
    regions: tuple[ReferenceRegion, ...] = Field(default_factory=tuple)
    page_modality: Literal["digital_vector", "scanned_raster"] = "digital_vector"

    @field_validator("phenomena", "regions", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_page_invariants(self) -> Self:
        """Validate unique IDs and a contiguous zero-based reading order on page."""
        seen_ids: set[str] = set()
        seen_reading_orders: set[int] = set()
        for r in self.regions:
            if r.id in seen_ids:
                raise ValueError(f"duplicate region id on page: {r.id}")
            seen_ids.add(r.id)
            if r.reading_order in seen_reading_orders:
                raise ValueError(f"duplicate reading order on page: {r.reading_order}")
            seen_reading_orders.add(r.reading_order)
        actual_order = sorted(seen_reading_orders)
        expected_order = list(range(len(self.regions)))
        if actual_order != expected_order:
            raise ValueError(
                "reading order must be contiguous and zero-based: "
                f"expected {expected_order}, got {actual_order}"
            )
        return self


class DocumentAnnotation(EvaluationBaseModel):
    """Complete reference audit dataset for a document version."""

    annotation_schema_version: Literal[2]
    annotation_version: str = Field(min_length=1)
    annotator: str = Field(min_length=1)
    annotator_type: AnnotatorType
    annotation_method: AnnotationMethod
    created_at: AwareDatetime
    prior_parser_output_exposure: bool
    parser_output_used_as_reference: bool
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: Sha256Digest
    audited_pages: tuple[AuditedPage, ...] = Field(default_factory=tuple)

    @field_validator("created_at", mode="before")
    @classmethod
    def _parse_created_at(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("audited_pages", mode="before")
    @classmethod
    def _coerce_pages_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_annotation_invariants(self) -> Self:
        """Validate unique page indices across audited pages."""
        seen_pages: set[int] = set()
        for p in self.audited_pages:
            if p.page_index in seen_pages:
                raise ValueError(f"duplicate audited page index: {p.page_index}")
            seen_pages.add(p.page_index)
        return self


__all__ = [
    "AnnotationBoundingBox",
    "AnnotationMethod",
    "AnnotatorType",
    "AuditedPage",
    "DocumentAnnotation",
    "EvaluationBaseModel",
    "ReferenceRegion",
    "RegionKind",
    "Sha256Digest",
]
