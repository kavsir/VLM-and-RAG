"""Normalization of raw MinerU parser outputs into Physical Document IR v0."""

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_rag.parsers.mineru import ParserRun
from vlm_rag.physical_ir.models import (
    BlockDisposition,
    BlockKind,
    BlockProvenance,
    BoundingBox,
    PhysicalBlock,
    PhysicalDocument,
    PhysicalPage,
)
from vlm_rag.physical_ir.serialization import physical_document_to_json
from vlm_rag.registry.models import DocumentManifest


class NormalizationError(ValueError):
    """Raised when raw parser output cannot be normalized into Physical Document IR."""


class MinerUPhysicalNormalizer:
    """Normalize raw MinerU layout artifacts into parser-independent PhysicalDocumentIR."""

    def normalize(
        self,
        raw_directory: Path,
        *,
        manifest: DocumentManifest | None = None,
        run: ParserRun | None = None,
        document_id: str | None = None,
        version_id: str | None = None,
        source_artifact_sha256: str | None = None,
        parser: str | None = None,
        parser_version: str | None = None,
        parser_backend: str | None = None,
    ) -> PhysicalDocument:
        """Normalize raw MinerU outputs into a validated PhysicalDocument."""
        if not raw_directory.exists():
            raise NormalizationError(f"raw directory does not exist: {raw_directory}")

        # Locate raw content list file
        content_list_path = self._locate_file(raw_directory, "_content_list.json")
        if content_list_path is None:
            raise NormalizationError(
                f"could not find MinerU content list JSON (*_content_list.json) in {raw_directory}"
            )

        # Locate middle.json and run.json if available
        middle_path = self._locate_file(raw_directory, "_middle.json")
        run_data = self._read_run_json(raw_directory)

        # Resolve document and parser metadata
        doc_id = document_id or (manifest.document.id if manifest else None)
        ver_id = version_id or (manifest.version.id if manifest else None)
        sha256 = (
            source_artifact_sha256
            or (manifest.artifact.sha256 if manifest else None)
            or (run.input_sha256 if run else None)
            or (run_data.get("input_sha256") if run_data else None)
        )

        p_name = (
            parser
            or (run.parser if run else None)
            or (run_data.get("parser") if run_data else None)
            or "mineru"
        )
        p_version = (
            parser_version
            or (run.parser_version if run else None)
            or (run_data.get("parser_version") if run_data else None)
        )
        p_backend = (
            parser_backend
            or (run.backend if run else None)
            or (run_data.get("backend") if run_data else None)
        )

        # Read middle metadata if needed
        middle_data: dict[str, Any] | None = None
        if middle_path is not None:
            middle_data = self._read_json_dict(middle_path)
            if p_version is None:
                p_version = middle_data.get("_version_name")
            if p_backend is None:
                p_backend = middle_data.get("_backend")

        if doc_id is None:
            raise NormalizationError("missing document_id; provide manifest or document_id")
        if ver_id is None:
            raise NormalizationError("missing version_id; provide manifest or version_id")
        if sha256 is None:
            raise NormalizationError(
                "missing source_artifact_sha256; provide manifest or source_artifact_sha256"
            )
        if p_version is None:
            p_version = "3.4.5"
        if p_backend is None:
            p_backend = "pipeline"

        # Read content list
        cl_items = self._read_content_list(content_list_path)

        # Extract page dimensions from middle.json if available
        page_dimensions: dict[int, tuple[float, float]] = {}
        total_pages_middle = 0
        if middle_data is not None:
            pdf_info = middle_data.get("pdf_info", [])
            if isinstance(pdf_info, list):
                total_pages_middle = len(pdf_info)
                for page in pdf_info:
                    if isinstance(page, dict):
                        pidx = page.get("page_idx")
                        psize = page.get("page_size")
                        if isinstance(pidx, int) and isinstance(psize, list) and len(psize) == 2:
                            with suppress(TypeError, ValueError):
                                page_dimensions[pidx] = (float(psize[0]), float(psize[1]))

        # Determine total page count
        max_page_idx = -1
        for item in cl_items:
            if "page_idx" in item and isinstance(item["page_idx"], int):
                max_page_idx = max(max_page_idx, item["page_idx"])
        page_count = max(total_pages_middle, max_page_idx + 1)

        # Group blocks by page index preserving reading order
        pages_blocks: dict[int, list[PhysicalBlock]] = {idx: [] for idx in range(page_count)}
        try:
            rel_artifact_path = content_list_path.relative_to(raw_directory).as_posix()
        except ValueError:
            rel_artifact_path = content_list_path.name

        for raw_idx, item in enumerate(cl_items):
            page_idx = item.get("page_idx")
            if not isinstance(page_idx, int) or page_idx < 0:
                raise NormalizationError(
                    f"item at index {raw_idx} in {content_list_path.name} "
                    f"has invalid page_idx: {page_idx}"
                )

            if page_idx not in pages_blocks:
                pages_blocks[page_idx] = []

            reading_order = len(pages_blocks[page_idx])
            block_id = f"{doc_id}_{ver_id}_p{page_idx:04d}_b{reading_order:04d}"

            block = self._map_block(
                raw_item=item,
                raw_index=raw_idx,
                block_id=block_id,
                page_index=page_idx,
                reading_order=reading_order,
                parser=p_name,
                parser_version=p_version,
                parser_backend=p_backend,
                source_raw_artifact=rel_artifact_path,
            )
            pages_blocks[page_idx].append(block)

        # Build PhysicalPages
        pages: list[PhysicalPage] = []
        for page_idx in range(page_count):
            dims = page_dimensions.get(page_idx)
            width = dims[0] if dims else None
            height = dims[1] if dims else None
            try:
                pages.append(
                    PhysicalPage(
                        page_index=page_idx,
                        width=width,
                        height=height,
                        blocks=tuple(pages_blocks.get(page_idx, [])),
                    )
                )
            except ValidationError as exc:
                raise NormalizationError(
                    f"failed to construct PhysicalPage for page {page_idx}:\n{exc}"
                ) from exc

        # Construct and validate PhysicalDocument
        try:
            return PhysicalDocument(
                physical_ir_version=1,
                document_id=doc_id,
                version_id=ver_id,
                source_artifact_sha256=sha256,
                parser=p_name,
                parser_version=p_version,
                parser_backend=p_backend,
                page_count=len(pages),
                pages=tuple(pages),
            )
        except ValidationError as exc:
            raise NormalizationError(
                f"failed to construct PhysicalDocument from {raw_directory}:\n{exc}"
            ) from exc

    def normalize_to_file(
        self,
        raw_directory: Path,
        output_path: Path,
        *,
        indent: int | None = 2,
        **kwargs: Any,
    ) -> PhysicalDocument:
        """Normalize raw MinerU output and atomically serialize to output_path."""
        document = self.normalize(raw_directory, **kwargs)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = physical_document_to_json(document, indent=indent)

        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            temporary_path.write_text(serialized, encoding="utf-8")
            os.replace(temporary_path, output_path)
        except OSError as exc:
            raise NormalizationError(
                f"failed to write normalized PhysicalDocument to {output_path}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)

        return document

    @staticmethod
    def _map_block(
        *,
        raw_item: Mapping[str, Any],
        raw_index: int,
        block_id: str,
        page_index: int,
        reading_order: int,
        parser: str,
        parser_version: str,
        parser_backend: str,
        source_raw_artifact: str,
    ) -> PhysicalBlock:
        raw_type = raw_item.get("type")
        raw_text = raw_item.get("text")
        text = str(raw_text) if raw_text is not None else ""
        raw_bbox = raw_item.get("bbox")

        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise NormalizationError(
                f"item at index {raw_index} has invalid or missing bbox: {raw_bbox}"
            )

        try:
            bbox = BoundingBox(
                x0=float(raw_bbox[0]),
                y0=float(raw_bbox[1]),
                x1=float(raw_bbox[2]),
                y1=float(raw_bbox[3]),
                coordinate_system="normalized_1000",
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise NormalizationError(
                f"item at index {raw_index} has invalid bounding box coordinates {raw_bbox}: {exc}"
            ) from exc

        has_text_level = "text_level" in raw_item
        heading_level: int | None = None

        if raw_type == "text" and not has_text_level:
            kind = BlockKind.TEXT
            disposition = BlockDisposition.CONTENT
        elif raw_type == "text" and has_text_level:
            kind = BlockKind.TITLE
            disposition = BlockDisposition.CONTENT
            try:
                heading_level = int(raw_item["text_level"])
                if heading_level < 1:
                    raise ValueError("text_level must be >= 1")
            except (TypeError, ValueError) as exc:
                raise NormalizationError(
                    f"item at index {raw_index} has invalid text_level "
                    f"{raw_item.get('text_level')!r}: {exc}"
                ) from exc
        elif raw_type == "header":
            kind = BlockKind.HEADER
            disposition = BlockDisposition.DISCARDED
        elif raw_type == "page_number":
            kind = BlockKind.PAGE_NUMBER
            disposition = BlockDisposition.DISCARDED
        else:
            kind = BlockKind.UNKNOWN
            disposition = BlockDisposition.UNKNOWN

        provenance = BlockProvenance(
            parser=parser,
            parser_version=parser_version,
            parser_backend=parser_backend,
            source_raw_artifact=source_raw_artifact,
            source_raw_index=raw_index,
        )

        try:
            return PhysicalBlock(
                id=block_id,
                page_index=page_index,
                reading_order=reading_order,
                kind=kind,
                disposition=disposition,
                text=text,
                bbox=bbox,
                heading_level=heading_level,
                provenance=provenance,
            )
        except ValidationError as exc:
            raise NormalizationError(
                f"failed to construct PhysicalBlock for item at index {raw_index}:\n{exc}"
            ) from exc

    @staticmethod
    def _locate_file(directory: Path, suffix: str) -> Path | None:
        """Find a file matching suffix in directory or immediate subdirectories."""
        # Check direct children first
        for candidate in sorted(directory.iterdir()):
            if candidate.is_file() and candidate.name.endswith(suffix):
                return candidate

        # Check subdirectories
        for candidate in sorted(directory.rglob(f"*{suffix}")):
            if candidate.is_file():
                return candidate

        return None

    @staticmethod
    def _read_content_list(path: Path) -> list[dict[str, Any]]:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NormalizationError(f"cannot read MinerU content list from {path}: {exc}") from exc

        if not isinstance(raw, list):
            raise NormalizationError(
                f"MinerU content list at {path} must be a JSON array, got {type(raw).__name__}"
            )

        items: list[dict[str, Any]] = []
        for idx, elem in enumerate(raw):
            if not isinstance(elem, dict):
                raise NormalizationError(
                    f"item {idx} in {path.name} is not an object: {type(elem).__name__}"
                )
            items.append(elem)
        return items

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, Any]:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NormalizationError(f"cannot read JSON from {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise NormalizationError(f"JSON at {path} must be an object, got {type(raw).__name__}")
        return raw

    def _read_run_json(self, raw_directory: Path) -> dict[str, Any] | None:
        candidates = [
            raw_directory / "run.json",
            raw_directory.parent / "run.json",
        ]
        for c in candidates:
            if c.is_file():
                try:
                    return self._read_json_dict(c)
                except NormalizationError:
                    pass
        return None


__all__ = ["MinerUPhysicalNormalizer", "NormalizationError"]
