"""Physical Document IR v1 domain models (wire/schema version 2).

Block disposition describes the current physical observation: ``content`` is body
content, ``discarded`` is intentional boilerplate exclusion, and ``unknown`` means
the normalizer cannot determine the disposition.  It is not inherited uncertainty
about a v0 block kind.
"""

import math
from enum import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import (
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from vlm_rag.physical_ir.models import BlockDisposition, BoundingBox, PhysicalIRModel
from vlm_rag.registry.models import Identifier, Sha256Digest, VersionIdentifier


class BlockKindV1(StrEnum):
    """Parser-independent physical categories supported by IR v1."""

    TEXT = "text"
    TITLE = "title"
    HEADER = "header"
    PAGE_NUMBER = "page_number"
    TABLE = "table"
    FIGURE = "figure"
    IMAGE = "image"
    UNKNOWN = "unknown"


class TextExtractionMethod(StrEnum):
    """How a parser obtained block text, when its evidence establishes the method."""

    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    UNKNOWN = "unknown"


class VisualAssetStorage(StrEnum):
    """How visual bytes are represented by the retained parser evidence."""

    RELATIVE_FILE = "relative_file"
    EMBEDDED_RAW = "embedded_raw"
    UNAVAILABLE = "unavailable"


class BlockProvenanceV1(PhysicalIRModel):
    """Trace a v1 block to its parser-native raw record."""

    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_backend: str = Field(min_length=1)
    source_raw_artifact: str = Field(min_length=1)
    source_raw_index: NonNegativeInt
    source_raw_type: str = Field(min_length=1)


class TextExtractionEvidence(PhysicalIRModel):
    """Parser-supported text extraction method and optional explicit confidence."""

    method: TextExtractionMethod
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("method", mode="before")
    @classmethod
    def _coerce_method(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return TextExtractionMethod(value)
            except ValueError:
                return value
        return value

    @field_validator("confidence")
    @classmethod
    def _require_finite_confidence(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value


class TableCell(PhysicalIRModel):
    """One logical table cell; coordinates are zero-based grid positions."""

    row_start: NonNegativeInt
    column_start: NonNegativeInt
    row_span: PositiveInt = 1
    column_span: PositiveInt = 1
    text: str
    is_header: bool | None


class TableStructure(PhysicalIRModel):
    """Optional parser-independent logical table grid."""

    row_count: NonNegativeInt
    column_count: NonNegativeInt
    cells: tuple[TableCell, ...] = Field(default_factory=tuple)

    @field_validator("cells", mode="before")
    @classmethod
    def _coerce_cells_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_grid(self) -> Self:
        """Require all cells to fit and forbid overlapping logical positions."""
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            row_end = cell.row_start + cell.row_span
            column_end = cell.column_start + cell.column_span
            if row_end > self.row_count or column_end > self.column_count:
                raise ValueError(
                    "table cell extent exceeds declared grid: "
                    f"cell=({cell.row_start}, {cell.column_start}, "
                    f"{cell.row_span}, {cell.column_span}), "
                    f"grid=({self.row_count}, {self.column_count})"
                )
            for row in range(cell.row_start, row_end):
                for column in range(cell.column_start, column_end):
                    position = (row, column)
                    if position in occupied:
                        raise ValueError(f"table cells overlap at logical position {position}")
                    occupied.add(position)
        return self


class VisualAssetEvidence(PhysicalIRModel):
    """Safe reference to visual bytes without embedding parser-specific schema."""

    storage_kind: VisualAssetStorage
    relative_path: str | None = None
    media_type: str | None = Field(default=None, min_length=1)
    sha256: Sha256Digest | None = None
    byte_size: PositiveInt | None = None

    @field_validator("storage_kind", mode="before")
    @classmethod
    def _coerce_storage_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return VisualAssetStorage(value)
            except ValueError:
                return value
        return value

    @field_validator("relative_path")
    @classmethod
    def _require_safe_relative_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        posix = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if (
            not value
            or "\\" in value
            or posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ValueError("relative_path must be a safe relative POSIX path")
        return value

    @model_validator(mode="after")
    def validate_storage_contract(self) -> Self:
        """Keep storage state and byte metadata internally consistent."""
        has_hash = self.sha256 is not None
        has_size = self.byte_size is not None
        if has_hash != has_size:
            raise ValueError("sha256 and byte_size must either both be present or both be null")
        if self.storage_kind == VisualAssetStorage.RELATIVE_FILE:
            if self.relative_path is None or not has_hash:
                raise ValueError(
                    "relative_file storage requires relative_path, sha256, and byte_size"
                )
        elif self.storage_kind == VisualAssetStorage.EMBEDDED_RAW:
            if self.relative_path is not None:
                raise ValueError("embedded_raw storage cannot carry relative_path")
            if not has_hash or self.media_type is None:
                raise ValueError("embedded_raw storage requires media_type, sha256, and byte_size")
        elif any(
            value is not None
            for value in (self.relative_path, self.media_type, self.sha256, self.byte_size)
        ):
            raise ValueError("unavailable storage cannot carry path or byte metadata")
        return self


class PhysicalBlockV1(PhysicalIRModel):
    """One physical parser observation in Physical IR v1.

    Disposition is about this v1 observation: CONTENT is body content, DISCARDED is
    intentionally excluded boilerplate, and UNKNOWN means disposition is not known.
    """

    id: str = Field(min_length=1)
    page_index: NonNegativeInt
    reading_order: NonNegativeInt
    kind: BlockKindV1
    disposition: BlockDisposition
    text: str
    bbox: BoundingBox
    heading_level: PositiveInt | None = None
    provenance: BlockProvenanceV1
    text_extraction: TextExtractionEvidence | None = None
    table_structure: TableStructure | None = None
    visual_asset: VisualAssetEvidence | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return BlockKindV1(value)
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

    @field_validator("bbox")
    @classmethod
    def _require_finite_bbox(cls, value: BoundingBox) -> BoundingBox:
        if not all(math.isfinite(item) for item in (value.x0, value.y0, value.x1, value.y1)):
            raise ValueError("v1 bounding-box coordinates must be finite")
        return value

    @model_validator(mode="after")
    def validate_conditional_fields(self) -> Self:
        """Constrain optional evidence to the physical observations it describes."""
        if self.kind != BlockKindV1.TITLE and self.heading_level is not None:
            raise ValueError("heading_level is only permitted for TITLE blocks")
        if self.kind != BlockKindV1.TABLE and self.table_structure is not None:
            raise ValueError("table_structure is only permitted for TABLE blocks")
        if (
            self.kind not in {BlockKindV1.FIGURE, BlockKindV1.IMAGE}
            and self.visual_asset is not None
        ):
            raise ValueError("visual_asset is only permitted for FIGURE or IMAGE blocks")
        if not self.text and self.text_extraction is not None:
            raise ValueError("text_extraction requires non-empty block text")
        return self


class PhysicalPageV1(PhysicalIRModel):
    """A page with ordered Physical IR v1 blocks."""

    page_index: NonNegativeInt
    width: float | None = Field(default=None, gt=0.0)
    height: float | None = Field(default=None, gt=0.0)
    blocks: tuple[PhysicalBlockV1, ...] = Field(default_factory=tuple)

    @field_validator("blocks", mode="before")
    @classmethod
    def _coerce_blocks_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("width", "height")
    @classmethod
    def _require_finite_dimension(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("v1 page dimensions must be finite")
        return value

    @model_validator(mode="after")
    def validate_page_blocks(self) -> Self:
        for index, block in enumerate(self.blocks):
            if block.page_index != self.page_index:
                raise ValueError(f"block {block.id!r} does not belong to page {self.page_index}")
            if block.reading_order != index:
                raise ValueError(
                    f"block {block.id!r} reading_order must equal sequence index {index}"
                )
        return self


class PhysicalDocumentV1(PhysicalIRModel):
    """Physical IR v1 document; research generation v1 has wire version 2."""

    physical_ir_version: Literal[2] = 2
    document_id: Identifier
    version_id: VersionIdentifier
    source_artifact_sha256: Sha256Digest
    parser: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    parser_backend: str = Field(min_length=1)
    page_count: NonNegativeInt
    pages: tuple[PhysicalPageV1, ...] = Field(default_factory=tuple)

    @field_validator("pages", mode="before")
    @classmethod
    def _coerce_pages_tuple(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_document(self) -> Self:
        if self.page_count != len(self.pages):
            raise ValueError("page_count must equal the number of pages")
        block_ids: set[str] = set()
        for index, page in enumerate(self.pages):
            if page.page_index != index:
                raise ValueError("pages must be contiguous and start at zero")
            for block in page.blocks:
                if block.id in block_ids:
                    raise ValueError(f"duplicate block id across document: {block.id!r}")
                block_ids.add(block.id)
        return self


__all__ = [
    "BlockKindV1",
    "BlockProvenanceV1",
    "PhysicalBlockV1",
    "PhysicalDocumentV1",
    "PhysicalPageV1",
    "TableCell",
    "TableStructure",
    "TextExtractionEvidence",
    "TextExtractionMethod",
    "VisualAssetEvidence",
    "VisualAssetStorage",
]
