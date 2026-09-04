"""Normalization of raw MinerU parser outputs into Physical Document IR v0."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
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


def _resolve_consistent_value(
    field_name: str,
    candidates: Sequence[tuple[str, Any]],
    *,
    required: bool = True,
    normalize: Callable[[str], str] | None = None,
) -> str | None:
    """Resolve a field value across multiple provenance sources, rejecting contradictions.

    - If no values are present and required is True, raises NormalizationError.
    - If multiple values are present, normalizes and verifies all are identical.
    - If values conflict, raises NormalizationError with conflicting sources and values.
    - Returns the resolved value, or None if not present and required is False.
    """
    present: list[tuple[str, str]] = []
    for source_name, raw_val in candidates:
        if raw_val is not None:
            val_str = str(raw_val).strip()
            if val_str:
                norm_val = normalize(val_str) if normalize else val_str
                present.append((source_name, norm_val))

    if not present:
        if required:
            sources_checked = [source_name for source_name, _ in candidates]
            raise NormalizationError(
                f"missing required provenance field {field_name!r}; "
                f"not provided in any available source: {sources_checked}"
            )
        return None

    _, first_val = present[0]
    for _, val in present[1:]:
        if val != first_val:
            conflict_details = ", ".join(f"{s}={v!r}" for s, v in present)
            raise NormalizationError(f"provenance conflict for {field_name!r}: {conflict_details}")

    return first_val


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

        # Locate raw content list file (unambiguous canonical match required)
        content_list_path = self._locate_canonical_file(
            raw_directory, "_content_list.json", required=True
        )
        assert content_list_path is not None  # guaranteed by required=True

        # Locate middle.json and run.json if available
        middle_path = self._locate_canonical_file(raw_directory, "_middle.json", required=False)
        run_data, _ = self._read_run_json(raw_directory)

        # Read middle metadata if available
        middle_data: dict[str, Any] | None = None
        if middle_path is not None:
            middle_data = self._read_json_dict(middle_path)

        # Resolve document and artifact provenance across all sources
        doc_id = _resolve_consistent_value(
            "document_id",
            [
                ("argument 'document_id'", document_id),
                ("manifest.document.id", manifest.document.id if manifest else None),
                ("run.document_id", run.document_id if run else None),
                ("run.json 'document_id'", run_data.get("document_id") if run_data else None),
            ],
            required=True,
        )
        assert doc_id is not None

        ver_id = _resolve_consistent_value(
            "version_id",
            [
                ("argument 'version_id'", version_id),
                ("manifest.version.id", manifest.version.id if manifest else None),
                ("run.version_id", run.version_id if run else None),
                ("run.json 'version_id'", run_data.get("version_id") if run_data else None),
            ],
            required=True,
        )
        assert ver_id is not None

        sha256 = _resolve_consistent_value(
            "source_artifact_sha256",
            [
                ("argument 'source_artifact_sha256'", source_artifact_sha256),
                ("manifest.artifact.sha256", manifest.artifact.sha256 if manifest else None),
                ("run.input_sha256", run.input_sha256 if run else None),
                ("run.json 'input_sha256'", run_data.get("input_sha256") if run_data else None),
            ],
            required=True,
            normalize=str.lower,
        )
        assert sha256 is not None

        # Resolve parser provenance across all sources
        p_name = (
            _resolve_consistent_value(
                "parser",
                [
                    ("argument 'parser'", parser),
                    ("run.parser", run.parser if run else None),
                    ("run.json 'parser'", run_data.get("parser") if run_data else None),
                ],
                required=False,
            )
            or "mineru"
        )

        p_version = _resolve_consistent_value(
            "parser_version",
            [
                ("argument 'parser_version'", parser_version),
                ("run.parser_version", run.parser_version if run else None),
                ("run.json 'parser_version'", run_data.get("parser_version") if run_data else None),
                (
                    "middle.json '_version_name'",
                    middle_data.get("_version_name") if middle_data else None,
                ),
            ],
            required=True,
        )
        assert p_version is not None

        p_backend = _resolve_consistent_value(
            "parser_backend",
            [
                ("argument 'parser_backend'", parser_backend),
                ("run.backend", run.backend if run else None),
                ("run.json 'backend'", run_data.get("backend") if run_data else None),
                ("middle.json '_backend'", middle_data.get("_backend") if middle_data else None),
            ],
            required=True,
        )
        assert p_backend is not None

        # Read content list
        cl_items = self._read_content_list(content_list_path)

        # Extract and strictly validate page geometry from middle.json if present
        page_dimensions: dict[int, tuple[float, float]] = {}
        if middle_data is not None and middle_path is not None:
            page_dimensions = self._validate_and_extract_middle_pages(middle_data, middle_path)

        # Determine page count and validate content list page bounds
        max_page_idx = -1
        for item in cl_items:
            if "page_idx" in item and isinstance(item["page_idx"], int):
                max_page_idx = max(max_page_idx, item["page_idx"])

        if middle_path is not None:
            page_count = len(page_dimensions)
            if max_page_idx >= page_count:
                raise NormalizationError(
                    f"content list block specifies page_idx={max_page_idx}, "
                    f"which exceeds middle.json page count ({page_count})"
                )
        else:
            page_count = max_page_idx + 1 if max_page_idx >= 0 else 0

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

            if middle_path is not None and page_idx not in page_dimensions:
                max_middle_page = len(page_dimensions) - 1
                raise NormalizationError(
                    f"item at index {raw_idx} in {content_list_path.name} specifies "
                    f"page_idx={page_idx}, outside middle range (0..{max_middle_page})"
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
        if not raw_directory.exists():
            raise NormalizationError(f"raw directory does not exist: {raw_directory}")

        raw_resolved = raw_directory.resolve()
        output_resolved = output_path.resolve()

        # Pre-write safety checks: Reject writing inside raw evidence directory or subdirectories
        if output_resolved == raw_resolved or output_resolved.is_relative_to(raw_resolved):
            raise NormalizationError(
                f"output_path {output_path} must not be inside raw directory {raw_directory}"
            )

        # Check against run.json if present outside raw_directory
        parent_run = raw_directory.parent / "run.json"
        if parent_run.is_file() and output_resolved == parent_run.resolve():
            raise NormalizationError(
                f"output_path {output_path} cannot overwrite execution evidence {parent_run}"
            )

        # Perform normalization completely in memory before touching destination filesystem
        document = self.normalize(raw_directory, **kwargs)

        # Serialize to deterministic UTF-8 bytes (using LF newlines)
        serialized_str = physical_document_to_json(document, indent=indent)
        serialized_bytes = serialized_str.encode("utf-8")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            temporary_path.write_bytes(serialized_bytes)
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
    def _locate_canonical_file(
        directory: Path,
        suffix: str,
        *,
        required: bool = True,
    ) -> Path | None:
        """Locate an unambiguous canonical file matching suffix in directory tree.

        If multiple files match, raises NormalizationError listing the candidate paths.
        If 0 matches and required is True, raises NormalizationError.
        """
        matches = [
            p for p in directory.rglob(f"*{suffix}") if p.is_file() and p.name.endswith(suffix)
        ]
        matches.sort()

        if len(matches) > 1:
            rel_paths: list[str] = []
            for p in matches:
                try:
                    rel_paths.append(str(p.relative_to(directory)))
                except ValueError:
                    rel_paths.append(str(p))
            raise NormalizationError(
                f"ambiguous raw parser input: found multiple files matching '*{suffix}' "
                f"in {directory}: {rel_paths}"
            )

        if not matches:
            if required:
                raise NormalizationError(
                    f"missing required raw parser input matching '*{suffix}' in {directory}"
                )
            return None

        return matches[0]

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

    def _read_run_json(self, raw_directory: Path) -> tuple[dict[str, Any] | None, Path | None]:
        candidates: list[Path] = []
        direct = raw_directory / "run.json"
        parent_file = raw_directory.parent / "run.json"
        if direct.is_file():
            candidates.append(direct)
        if parent_file.is_file() and parent_file not in candidates:
            candidates.append(parent_file)

        if not candidates:
            return None, None

        if len(candidates) > 1:
            raise NormalizationError(
                f"ambiguous run.json: found run.json at both {direct} and {parent_file}"
            )

        run_path = candidates[0]
        data = self._read_json_dict(run_path)
        return data, run_path

    @staticmethod
    def _validate_and_extract_middle_pages(
        middle_data: dict[str, Any],
        middle_path: Path,
    ) -> dict[int, tuple[float, float]]:
        """Strictly validate middle.json page structure and return page dimensions."""
        if "pdf_info" not in middle_data:
            raise NormalizationError(f"{middle_path.name} is missing required 'pdf_info' field")
        pdf_info = middle_data["pdf_info"]
        if not isinstance(pdf_info, list):
            raise NormalizationError(
                f"'pdf_info' in {middle_path.name} must be a list, got {type(pdf_info).__name__}"
            )

        page_dimensions: dict[int, tuple[float, float]] = {}
        seen_page_indices: set[int] = set()

        for idx, page_record in enumerate(pdf_info):
            if not isinstance(page_record, dict):
                raise NormalizationError(
                    f"page record at index {idx} in {middle_path.name} 'pdf_info' must be a dict"
                )
            if "page_idx" not in page_record:
                raise NormalizationError(
                    f"page record at index {idx} in {middle_path.name} is missing 'page_idx'"
                )
            page_idx = page_record["page_idx"]
            if not isinstance(page_idx, int) or page_idx < 0:
                raise NormalizationError(
                    f"page record {idx} in {middle_path.name} has invalid page_idx: {page_idx!r}"
                )
            if page_idx in seen_page_indices:
                raise NormalizationError(
                    f"duplicate page_idx {page_idx} found in {middle_path.name} 'pdf_info'"
                )
            seen_page_indices.add(page_idx)

            if "page_size" not in page_record:
                raise NormalizationError(
                    f"page record {page_idx} in {middle_path.name} is missing 'page_size'"
                )
            page_size = page_record["page_size"]
            if not isinstance(page_size, list) or len(page_size) != 2:
                raise NormalizationError(
                    f"page_size for page {page_idx} in {middle_path.name} "
                    f"must be a list of 2 numbers, got {page_size!r}"
                )
            try:
                width = float(page_size[0])
                height = float(page_size[1])
            except (TypeError, ValueError) as exc:
                raise NormalizationError(
                    f"page_size dimensions for page {page_idx} in {middle_path.name} "
                    f"must be numeric: {page_size!r}"
                ) from exc
            if width <= 0 or height <= 0:
                raise NormalizationError(
                    f"page_size for page {page_idx} in {middle_path.name} "
                    f"must be positive, got ({width}, {height})"
                )
            page_dimensions[page_idx] = (width, height)

        expected_indices = set(range(len(pdf_info)))
        if seen_page_indices != expected_indices:
            if 0 not in seen_page_indices:
                raise NormalizationError(
                    f"page indexes in {middle_path.name} must start at 0, but 0 was not found"
                )
            max_idx = len(pdf_info) - 1
            raise NormalizationError(
                f"page indexes in {middle_path.name} must be contiguous from 0 to {max_idx}; "
                f"got {sorted(seen_page_indices)}"
            )

        return page_dimensions


__all__ = ["MinerUPhysicalNormalizer", "NormalizationError"]
