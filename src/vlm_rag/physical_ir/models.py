"""Typed domain models for Physical Document Intermediate Representation (v0)."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from vlm_rag.registry.models import Identifier, Sha256Digest, VersionIdentifier


class PhysicalIRModel(BaseModel):
    """Shared strict, immutable configuration for physical IR records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class BlockKind(StrEnum):
    """Category of physical content observed by the parser."""

    TEXT = "text"
    TITLE = "title"
    HEADER = "header"
    PAGE_NUMBER = "page_number"
    UNKNOWN = "unknown"


class BlockDisposition(StrEnum):
    """Disposition of a physical block within document content structure."""

    CONTENT = "content"
    DISCARDED = "discarded"
    UNKNOWN = "unknown"


class BoundingBox(PhysicalIRModel):
    """Parser-independent 2D bounding box on a document page.

    For Physical Document IR v0, coordinates use MinerU's native normalized layout grid:
    - Origin (0, 0) is at the top-left corner of the page.
    - x0, y0: Top-left corner coordinates (0 <= x0 <= x1 <= 1000, 0 <= y0 <= y1 <= 1000).
    - x1, y1: Bottom-right corner coordinates.
    - coordinate_system: Strictly "normalized_1000".
    """

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


class BlockProvenance(PhysicalIRModel):
    """Traceability linking a physical block back to raw parser output."""

    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_backend: str = Field(min_length=1)
    source_raw_artifact: str = Field(min_length=1)
    source_raw_index: NonNegativeInt


class PhysicalBlock(PhysicalIRModel):
    """A physical segment or layout block on a document page."""

    id: str = Field(min_length=1)
    page_index: NonNegativeInt
    reading_order: NonNegativeInt
    kind: BlockKind
    disposition: BlockDisposition
    text: str
    bbox: BoundingBox
    heading_level: PositiveInt | None = Field(default=None)
    provenance: BlockProvenance

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return BlockKind(value)
            except ValueError:
                return value
        return value

    @field_validator("disposition", mode="before")
    @classmethod
    def _coerce_disposition(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return BlockDisposition(value)
            except ValueError:
                return value
        return value

    @model_validator(mode="after")
    def validate_heading_level(self) -> Self:
        """Enforce that heading_level is strictly constrained to TITLE blocks."""
        if self.kind != BlockKind.TITLE and self.heading_level is not None:
            raise ValueError(
                f"heading_level is only permitted for TITLE blocks, got {self.kind.value!r}"
            )
        return self


class PhysicalPage(PhysicalIRModel):
    """A single page of a physical document and its ordered layout blocks."""

    page_index: NonNegativeInt
    width: PositiveFloat | None = Field(
        default=None,
        description="Parser-reported native page width in PDF canvas/native page units.",
    )
    height: PositiveFloat | None = Field(
        default=None,
        description="Parser-reported native page height in PDF canvas/native page units.",
    )
    blocks: tuple[PhysicalBlock, ...] = Field(default_factory=tuple)

    @field_validator("blocks", mode="before")
    @classmethod
    def _coerce_blocks_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_page_blocks(self) -> Self:
        """Enforce reading order and page index consistency for blocks on this page."""
        for idx, block in enumerate(self.blocks):
            if block.page_index != self.page_index:
                raise ValueError(
                    f"block {block.id!r} page_index ({block.page_index}) "
                    f"does not match page ({self.page_index})"
                )
            if block.reading_order != idx:
                raise ValueError(
                    f"block {block.id!r} reading_order ({block.reading_order}) "
                    f"must match sequence index {idx}"
                )
        return self


class PhysicalDocument(PhysicalIRModel):
    """Parser-independent representation of what physically exists in a document."""

    physical_ir_version: Literal[1] = 1
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_backend: str = Field(min_length=1)
    page_count: NonNegativeInt
    pages: tuple[PhysicalPage, ...] = Field(default_factory=tuple)

    @field_validator("pages", mode="before")
    @classmethod
    def _coerce_pages_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        """Enforce document-wide page and block invariants."""
        if self.page_count != len(self.pages):
            raise ValueError(
                f"page_count ({self.page_count}) does not match number of pages ({len(self.pages)})"
            )
        seen_pages: set[int] = set()
        seen_blocks: set[str] = set()
        for idx, page in enumerate(self.pages):
            if page.page_index in seen_pages:
                raise ValueError(f"duplicate page_index: {page.page_index}")
            seen_pages.add(page.page_index)
            if page.page_index != idx:
                raise ValueError(
                    f"pages must be contiguous starting from 0; expected index {idx}, "
                    f"got {page.page_index}"
                )
            for block in page.blocks:
                if block.id in seen_blocks:
                    raise ValueError(f"duplicate block id across document: {block.id!r}")
                seen_blocks.add(block.id)
        return self


__all__ = [
    "BlockDisposition",
    "BlockKind",
    "BlockProvenance",
    "BoundingBox",
    "PhysicalBlock",
    "PhysicalDocument",
    "PhysicalIRModel",
    "PhysicalPage",
]
