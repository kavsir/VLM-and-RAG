"""Typed domain models for ground-truth layout annotation and spatial evaluation."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)


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
    UNKNOWN = "unknown"


class AnnotationBoundingBox(EvaluationBaseModel):
    """Normalized 1000-based bounding box for ground truth."""

    x0: float = Field(ge=0.0, le=1000.0)
    y0: float = Field(ge=0.0, le=1000.0)
    x1: float = Field(ge=0.0, le=1000.0)
    y1: float = Field(ge=0.0, le=1000.0)
    coordinate_system: Literal["normalized_1000"] = "normalized_1000"

    @model_validator(mode="after")
    def validate_coordinates(self) -> Self:
        """Ensure coordinate ordering is topologically valid."""
        if self.x0 > self.x1:
            raise ValueError(f"x0 ({self.x0}) cannot exceed x1 ({self.x1})")
        if self.y0 > self.y1:
            raise ValueError(f"y0 ({self.y0}) cannot exceed y1 ({self.y1})")
        return self


class GroundTruthRegion(EvaluationBaseModel):
    """An audited ground-truth layout segment on a document page."""

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
    """Audited regions and documented phenomena for one page."""

    page_index: NonNegativeInt
    phenomena: tuple[str, ...] = Field(default_factory=tuple)
    regions: tuple[GroundTruthRegion, ...] = Field(default_factory=tuple)

    @field_validator("phenomena", "regions", mode="before")
    @classmethod
    def _coerce_tuples(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_reading_order(self) -> Self:
        """Validate unique IDs on page."""
        seen_ids: set[str] = set()
        for r in self.regions:
            if r.id in seen_ids:
                raise ValueError(f"duplicate region id on page: {r.id}")
            seen_ids.add(r.id)
        return self


class DocumentAnnotation(EvaluationBaseModel):
    """Complete ground-truth audit dataset for a document version."""

    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(min_length=1)
    audited_pages: tuple[AuditedPage, ...] = Field(default_factory=tuple)

    @field_validator("audited_pages", mode="before")
    @classmethod
    def _coerce_pages_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value
