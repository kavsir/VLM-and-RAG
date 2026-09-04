"""Normalize observed Marker 2.0 JSON into Physical Document IR v0."""

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_rag.parsers.marker import (
    MARKER_BACKEND,
    MARKER_MODE,
    MARKER_NAME,
    MARKER_OUTPUT_FORMAT,
    MARKER_VERSION,
    MarkerOutputError,
    MarkerRun,
    discover_marker_document_json,
)
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

_PAGE_ID = re.compile(r"^/page/(?P<index>[0-9]+)/Page/[0-9]+$")
_CONTENT_REF = re.compile(
    r"<content-ref\s+src=(?P<quote>['\"])(?P<id>[^'\"]+)(?P=quote)\s*></content-ref>"
)
_BLOCK_TAGS = frozenset(
    {
        "address",
        "article",
        "aside",
        "blockquote",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
        "ol",
    }
)
_GEOMETRY_TOLERANCE = 1e-6
_MAX_CONTENT_REF_DEPTH = 64


class MarkerNormalizationError(ValueError):
    """Raised when raw Marker evidence cannot be safely normalized."""


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.casefold() == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


class _HeadingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading_level: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.casefold()
        if (
            self.heading_level is None
            and len(normalized) == 2
            and normalized[0] == "h"
            and normalized[1] in "123456"
        ):
            self.heading_level = int(normalized[1])


def _resolve_consistent_value(
    field_name: str,
    candidates: Sequence[tuple[str, object]],
    *,
    normalize: Callable[[str], str] | None = None,
) -> str:
    present: list[tuple[str, str]] = []
    for source, raw_value in candidates:
        if raw_value is not None:
            value = str(raw_value).strip()
            if value:
                present.append((source, normalize(value) if normalize else value))
    if not present:
        raise MarkerNormalizationError(f"missing required provenance field {field_name!r}")
    expected = present[0][1]
    if any(value != expected for _, value in present[1:]):
        detail = ", ".join(f"{source}={value!r}" for source, value in present)
        raise MarkerNormalizationError(f"provenance conflict for {field_name!r}: {detail}")
    return expected


def _resolve_consistent_bool(field_name: str, candidates: Sequence[tuple[str, object]]) -> bool:
    present: list[tuple[str, bool]] = []
    for source, value in candidates:
        if value is not None:
            if not isinstance(value, bool):
                raise MarkerNormalizationError(
                    f"provenance field {field_name!r} from {source} must be boolean"
                )
            present.append((source, value))
    if not present:
        raise MarkerNormalizationError(f"missing required provenance field {field_name!r}")
    expected = present[0][1]
    if any(value is not expected for _, value in present[1:]):
        detail = ", ".join(f"{source}={value!r}" for source, value in present)
        raise MarkerNormalizationError(f"provenance conflict for {field_name!r}: {detail}")
    return expected


class MarkerPhysicalNormalizer:
    """Map canonical page-level Marker blocks into the frozen Physical IR v0."""

    def normalize(
        self,
        raw_directory: Path,
        *,
        manifest: DocumentManifest | None = None,
        run: MarkerRun | None = None,
        document_id: str | None = None,
        version_id: str | None = None,
        source_artifact_sha256: str | None = None,
        parser: str | None = None,
        parser_version: str | None = None,
        parser_backend: str | None = None,
        mode: str | None = None,
        disable_ocr: bool | None = None,
        output_format: str | None = None,
    ) -> PhysicalDocument:
        """Normalize one unambiguous Marker document tree entirely in memory."""
        if not raw_directory.is_dir():
            raise MarkerNormalizationError(f"raw directory does not exist: {raw_directory}")
        try:
            document_path = discover_marker_document_json(raw_directory)
        except MarkerOutputError as exc:
            raise MarkerNormalizationError(str(exc)) from exc
        raw_root = self._read_json(document_path)
        run_data, _ = self._read_run_json(raw_directory)

        doc_id = _resolve_consistent_value(
            "document_id",
            [
                ("argument", document_id),
                ("manifest", manifest.document.id if manifest else None),
                ("run", run.document_id if run else None),
                ("run.json", run_data.get("document_id") if run_data else None),
            ],
        )
        ver_id = _resolve_consistent_value(
            "version_id",
            [
                ("argument", version_id),
                ("manifest", manifest.version.id if manifest else None),
                ("run", run.version_id if run else None),
                ("run.json", run_data.get("version_id") if run_data else None),
            ],
        )
        source_sha = _resolve_consistent_value(
            "source_artifact_sha256",
            [
                ("argument", source_artifact_sha256),
                ("manifest", manifest.artifact.sha256 if manifest else None),
                ("run", run.input_sha256 if run else None),
                ("run.json", run_data.get("input_sha256") if run_data else None),
            ],
            normalize=str.lower,
        )
        parser_name = _resolve_consistent_value(
            "parser",
            [
                ("normalizer", MARKER_NAME),
                ("argument", parser),
                ("run", run.parser if run else None),
                ("run.json", run_data.get("parser") if run_data else None),
            ],
        )
        resolved_version = _resolve_consistent_value(
            "parser_version",
            [
                ("argument", parser_version),
                ("run", run.parser_version if run else None),
                ("run.json", run_data.get("parser_version") if run_data else None),
            ],
        )
        resolved_backend = _resolve_consistent_value(
            "parser_backend",
            [
                ("argument", parser_backend),
                ("run", run.backend if run else None),
                ("run.json", run_data.get("backend") if run_data else None),
            ],
        )
        resolved_mode = _resolve_consistent_value(
            "mode",
            [
                ("argument", mode),
                ("run", run.mode if run else None),
                ("run.json", run_data.get("mode") if run_data else None),
            ],
        )
        resolved_disable_ocr = _resolve_consistent_bool(
            "disable_ocr",
            [
                ("argument", disable_ocr),
                ("run", run.disable_ocr if run else None),
                ("run.json", run_data.get("disable_ocr") if run_data else None),
            ],
        )
        resolved_format = _resolve_consistent_value(
            "output_format",
            [
                ("argument", output_format),
                ("run", run.output_format if run else None),
                ("run.json", run_data.get("output_format") if run_data else None),
            ],
        )
        expected_configuration = (
            ("parser_version", resolved_version, MARKER_VERSION),
            ("parser_backend", resolved_backend, MARKER_BACKEND),
            ("mode", resolved_mode, MARKER_MODE),
            ("output_format", resolved_format, MARKER_OUTPUT_FORMAT),
        )
        for field, actual, expected in expected_configuration:
            if actual != expected:
                raise MarkerNormalizationError(
                    f"unsupported Marker baseline {field}: expected {expected!r}, got {actual!r}"
                )
        if not resolved_disable_ocr:
            raise MarkerNormalizationError("Marker baseline requires disable_ocr=true")

        pages_data = self._validate_root(raw_root, document_path)
        try:
            source_raw_artifact = document_path.relative_to(raw_directory).as_posix()
        except ValueError:
            source_raw_artifact = document_path.name
        pages: list[PhysicalPage] = []
        global_raw_index = 0
        for page_index, page_data in enumerate(pages_data):
            page, global_raw_index = self._map_page(
                page_data,
                page_index=page_index,
                starting_raw_index=global_raw_index,
                document_id=doc_id,
                version_id=ver_id,
                parser=parser_name,
                parser_version=resolved_version,
                parser_backend=resolved_backend,
                source_raw_artifact=source_raw_artifact,
            )
            pages.append(page)
        try:
            return PhysicalDocument(
                physical_ir_version=1,
                document_id=doc_id,
                version_id=ver_id,
                source_artifact_sha256=source_sha,
                parser=parser_name,
                parser_version=resolved_version,
                parser_backend=resolved_backend,
                page_count=len(pages),
                pages=tuple(pages),
            )
        except ValidationError as exc:
            raise MarkerNormalizationError(f"invalid normalized PhysicalDocument:\n{exc}") from exc

    def normalize_to_file(
        self,
        raw_directory: Path,
        output_path: Path,
        *,
        source_artifact_path: Path | None = None,
        indent: int | None = 2,
        **kwargs: Any,
    ) -> PhysicalDocument:
        """Normalize in memory, then atomically persist deterministic UTF-8/LF bytes."""
        self._protect_raw_evidence(raw_directory, output_path, source_artifact_path)
        document = self.normalize(raw_directory, **kwargs)
        payload = physical_document_to_json(document, indent=indent).encode("utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary_path = Path(name)
            temporary_path.write_bytes(payload)
            os.replace(temporary_path, output_path)
        except OSError as exc:
            raise MarkerNormalizationError(
                f"cannot write normalized output {output_path}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
        return document

    @staticmethod
    def _protect_raw_evidence(
        raw_directory: Path,
        output_path: Path,
        source_artifact_path: Path | None,
    ) -> None:
        if not raw_directory.is_dir():
            raise MarkerNormalizationError(f"raw directory does not exist: {raw_directory}")
        raw_resolved = raw_directory.resolve()
        output_resolved = output_path.resolve()
        if source_artifact_path is not None and output_resolved == source_artifact_path.resolve():
            raise MarkerNormalizationError(
                "normalized output would overwrite the authoritative source artifact"
            )
        if output_resolved == raw_resolved or output_resolved.is_relative_to(raw_resolved):
            raise MarkerNormalizationError("normalized output must not be inside the raw directory")
        run_directory = raw_resolved.parent
        protected_names = {"run.json", "stdout.log", "stderr.log", "artifact_manifest.json"}
        if output_resolved.parent == run_directory and output_resolved.name in protected_names:
            raise MarkerNormalizationError(
                f"normalized output would overwrite {output_resolved.name}"
            )

    def _map_page(
        self,
        raw_page: Mapping[str, Any],
        *,
        page_index: int,
        starting_raw_index: int,
        document_id: str,
        version_id: str,
        parser: str,
        parser_version: str,
        parser_backend: str,
        source_raw_artifact: str,
    ) -> tuple[PhysicalPage, int]:
        if raw_page.get("block_type") != "Page":
            raise MarkerNormalizationError(f"document child {page_index} is not a Page")
        page_id = raw_page.get("id")
        if not isinstance(page_id, str) or (match := _PAGE_ID.fullmatch(page_id)) is None:
            raise MarkerNormalizationError(f"page {page_index} has invalid Marker id {page_id!r}")
        if int(match.group("index")) != page_index:
            raise MarkerNormalizationError(
                f"page id {page_id!r} does not match contiguous index {page_index}"
            )
        page_bbox = _read_bbox(raw_page, f"page {page_index}")
        page_x0, page_y0, page_x1, page_y1 = page_bbox
        width = page_x1 - page_x0
        height = page_y1 - page_y0
        if width <= 0 or height <= 0:
            raise MarkerNormalizationError(f"page {page_index} has non-positive dimensions")
        raw_children = raw_page.get("children")
        if not isinstance(raw_children, list):
            raise MarkerNormalizationError(f"page {page_index} children must be a list")
        blocks: list[PhysicalBlock] = []
        raw_index = starting_raw_index
        for reading_order, raw_block in enumerate(raw_children):
            if not isinstance(raw_block, Mapping):
                raise MarkerNormalizationError(
                    f"page {page_index} child {reading_order} must be an object"
                )
            bbox = _project_bbox(
                _read_bbox(raw_block, f"page {page_index} child {reading_order}"), page_bbox
            )
            raw_type = raw_block.get("block_type")
            if not isinstance(raw_type, str) or not raw_type:
                raise MarkerNormalizationError(
                    f"page {page_index} child {reading_order} has invalid block_type"
                )
            html = _render_block_html(raw_block)
            text = _visible_text(html)
            kind, disposition = _map_type(raw_type)
            heading_level = _heading_level(html) if kind == BlockKind.TITLE else None
            block_id = f"{document_id}_{version_id}_p{page_index:04d}_b{reading_order:04d}"
            try:
                blocks.append(
                    PhysicalBlock(
                        id=block_id,
                        page_index=page_index,
                        reading_order=reading_order,
                        kind=kind,
                        disposition=disposition,
                        text=text,
                        bbox=bbox,
                        heading_level=heading_level,
                        provenance=BlockProvenance(
                            parser=parser,
                            parser_version=parser_version,
                            parser_backend=parser_backend,
                            source_raw_artifact=source_raw_artifact,
                            source_raw_index=raw_index,
                        ),
                    )
                )
            except ValidationError as exc:
                raise MarkerNormalizationError(
                    f"cannot map page {page_index} child {reading_order}:\n{exc}"
                ) from exc
            raw_index += 1
        try:
            return (
                PhysicalPage(
                    page_index=page_index, width=width, height=height, blocks=tuple(blocks)
                ),
                raw_index,
            )
        except ValidationError as exc:
            raise MarkerNormalizationError(f"cannot map page {page_index}:\n{exc}") from exc

    @staticmethod
    def _validate_root(raw_root: object, path: Path) -> list[Mapping[str, Any]]:
        if not isinstance(raw_root, Mapping):
            raise MarkerNormalizationError(f"Marker document root in {path} must be an object")
        if raw_root.get("block_type") != "Document":
            raise MarkerNormalizationError("Marker document root block_type must be 'Document'")
        raw_pages = raw_root.get("children")
        if not isinstance(raw_pages, list):
            raise MarkerNormalizationError("Marker Document children must be a list")
        pages: list[Mapping[str, Any]] = []
        for index, page in enumerate(raw_pages):
            if not isinstance(page, Mapping):
                raise MarkerNormalizationError(f"Marker page {index} must be an object")
            pages.append(page)
        return pages

    @staticmethod
    def _read_json(path: Path) -> object:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MarkerNormalizationError(f"cannot read JSON from {path}: {exc}") from exc

    def _read_run_json(self, raw_directory: Path) -> tuple[dict[str, Any] | None, Path | None]:
        candidates = [
            path
            for path in (raw_directory / "run.json", raw_directory.parent / "run.json")
            if path.is_file()
        ]
        if len(candidates) > 1:
            raise MarkerNormalizationError("ambiguous run.json inside and beside raw directory")
        if not candidates:
            return None, None
        path = candidates[0]
        value = self._read_json(path)
        if not isinstance(value, dict):
            raise MarkerNormalizationError(f"run.json at {path} must be an object")
        return value, path


def _read_bbox(raw: Mapping[str, Any], label: str) -> tuple[float, float, float, float]:
    value = raw.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        raise MarkerNormalizationError(f"{label} bbox must contain four numbers")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise MarkerNormalizationError(f"{label} bbox must contain four numbers")
    x0, y0, x1, y1 = (float(item) for item in value)
    if x0 > x1 or y0 > y1:
        raise MarkerNormalizationError(f"{label} bbox coordinates are reversed")
    return x0, y0, x1, y1


def _project_bbox(
    block: tuple[float, float, float, float], page: tuple[float, float, float, float]
) -> BoundingBox:
    bx0, by0, bx1, by1 = block
    px0, py0, px1, py1 = page
    if (
        bx0 < px0 - _GEOMETRY_TOLERANCE
        or by0 < py0 - _GEOMETRY_TOLERANCE
        or bx1 > px1 + _GEOMETRY_TOLERANCE
        or by1 > py1 + _GEOMETRY_TOLERANCE
    ):
        raise MarkerNormalizationError(f"block bbox {block} lies outside page bounds {page}")
    width = px1 - px0
    height = py1 - py0
    if width <= 0 or height <= 0:
        raise MarkerNormalizationError("cannot project bbox from non-positive page bounds")
    bx0 = px0 if bx0 < px0 else bx0
    by0 = py0 if by0 < py0 else by0
    bx1 = px1 if bx1 > px1 else bx1
    by1 = py1 if by1 > py1 else by1
    normalized = (
        round((bx0 - px0) / width * 1000, 6),
        round((by0 - py0) / height * 1000, 6),
        round((bx1 - px0) / width * 1000, 6),
        round((by1 - py0) / height * 1000, 6),
    )
    if any(value < 0 or value > 1000 for value in normalized):
        raise MarkerNormalizationError(f"projected bbox is outside normalized_1000: {normalized}")
    try:
        return BoundingBox(
            x0=0.0 if normalized[0] == -0.0 else normalized[0],
            y0=0.0 if normalized[1] == -0.0 else normalized[1],
            x1=1000.0 if normalized[2] == 1000.0 else normalized[2],
            y1=1000.0 if normalized[3] == 1000.0 else normalized[3],
            coordinate_system="normalized_1000",
        )
    except ValidationError as exc:
        raise MarkerNormalizationError(f"invalid projected bbox {normalized}:\n{exc}") from exc


def _render_block_html(
    raw_block: Mapping[str, Any],
    *,
    active_ids: frozenset[str] = frozenset(),
    depth: int = 0,
) -> str:
    if depth > _MAX_CONTENT_REF_DEPTH:
        raise MarkerNormalizationError(
            f"Marker content-ref nesting exceeds {_MAX_CONTENT_REF_DEPTH} levels"
        )
    raw_id = raw_block.get("id")
    if isinstance(raw_id, str):
        if raw_id in active_ids:
            raise MarkerNormalizationError(f"cyclic Marker content-ref involving {raw_id!r}")
        active_ids = active_ids | {raw_id}
    html = raw_block.get("html")
    if not isinstance(html, str):
        raise MarkerNormalizationError("canonical Marker block html must be a string")
    raw_children = raw_block.get("children")
    if raw_children is None:
        return html
    if not isinstance(raw_children, list):
        raise MarkerNormalizationError("Marker block children must be a list or null")
    children: dict[str, Mapping[str, Any]] = {}
    for child in raw_children:
        if not isinstance(child, Mapping) or not isinstance(child.get("id"), str):
            raise MarkerNormalizationError("nested Marker child must have a string id")
        child_id = str(child["id"])
        if child_id in children:
            raise MarkerNormalizationError(f"duplicate nested Marker child id {child_id!r}")
        children[child_id] = child

    def replace(match: re.Match[str]) -> str:
        child_id = match.group("id")
        child = children.get(child_id)
        if child is None:
            raise MarkerNormalizationError(f"unresolved Marker content-ref {child_id!r}")
        return _render_block_html(child, active_ids=active_ids, depth=depth + 1)

    return _CONTENT_REF.sub(replace, html)


def _visible_text(html: str) -> str:
    parser = _VisibleTextParser()
    try:
        parser.feed(html)
        parser.close()
    except ValueError as exc:
        raise MarkerNormalizationError(f"cannot parse Marker HTML: {exc}") from exc
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def _heading_level(html: str) -> int | None:
    parser = _HeadingParser()
    try:
        parser.feed(html)
        parser.close()
    except ValueError as exc:
        raise MarkerNormalizationError(f"cannot parse Marker heading HTML: {exc}") from exc
    return parser.heading_level


def _map_type(raw_type: str) -> tuple[BlockKind, BlockDisposition]:
    if raw_type == "Text":
        return BlockKind.TEXT, BlockDisposition.CONTENT
    if raw_type == "SectionHeader":
        return BlockKind.TITLE, BlockDisposition.CONTENT
    if raw_type == "PageHeader":
        return BlockKind.HEADER, BlockDisposition.DISCARDED
    if raw_type == "PageFooter":
        return BlockKind.UNKNOWN, BlockDisposition.DISCARDED
    return BlockKind.UNKNOWN, BlockDisposition.CONTENT


__all__ = ["MarkerNormalizationError", "MarkerPhysicalNormalizer"]
