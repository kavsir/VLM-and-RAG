"""Marker raw-output normalization into Physical Document IR v1."""

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_rag.normalizers._assets_v1 import VisualAssetSourceError, embedded_visual_asset
from vlm_rag.normalizers.marker import MarkerNormalizationError, MarkerPhysicalNormalizer
from vlm_rag.parsers.marker import MarkerRun, discover_marker_document_json
from vlm_rag.physical_ir.models import PhysicalBlock
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.table_html import TableHTMLStructureError, table_structure_from_html
from vlm_rag.physical_ir.v1 import (
    BlockKindV1,
    BlockProvenanceV1,
    PhysicalBlockV1,
    PhysicalDocumentV1,
    PhysicalPageV1,
    TextExtractionEvidence,
    TextExtractionMethod,
    VisualAssetEvidence,
    VisualAssetStorage,
)
from vlm_rag.registry.models import DocumentManifest


class MarkerPhysicalNormalizerV1:
    """Enrich the frozen Marker v0 mapping from the same retained raw evidence."""

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
    ) -> PhysicalDocumentV1:
        """Normalize in memory; no asset or IR files are created."""
        arguments: dict[str, Any] = {
            "manifest": manifest,
            "run": run,
            "document_id": document_id,
            "version_id": version_id,
            "source_artifact_sha256": source_artifact_sha256,
            "parser": parser,
            "parser_version": parser_version,
            "parser_backend": parser_backend,
            "mode": mode,
            "disable_ocr": disable_ocr,
            "output_format": output_format,
        }
        historical = MarkerPhysicalNormalizer().normalize(raw_directory, **arguments)
        document_path = discover_marker_document_json(raw_directory)
        try:
            raw_root = json.loads(document_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MarkerNormalizationError(f"cannot read JSON from {document_path}: {exc}") from exc
        if not isinstance(raw_root, Mapping) or not isinstance(raw_root.get("children"), list):
            raise MarkerNormalizationError("Marker document tree changed after v0 validation")
        raw_pages = raw_root["children"]
        if len(raw_pages) != len(historical.pages):
            raise MarkerNormalizationError("Marker raw page count changed after v0 validation")

        pages: list[PhysicalPageV1] = []
        for historical_page, raw_page in zip(historical.pages, raw_pages, strict=True):
            if not isinstance(raw_page, Mapping) or not isinstance(raw_page.get("children"), list):
                raise MarkerNormalizationError("Marker raw page changed after v0 validation")
            raw_blocks = raw_page["children"]
            if len(raw_blocks) != len(historical_page.blocks):
                raise MarkerNormalizationError("Marker raw block count changed after v0 validation")
            blocks = [
                self._enrich_block(block, raw_block)
                for block, raw_block in zip(historical_page.blocks, raw_blocks, strict=True)
                if isinstance(raw_block, Mapping)
            ]
            if len(blocks) != len(raw_blocks):
                raise MarkerNormalizationError("Marker raw page contains a non-object block")
            pages.append(
                PhysicalPageV1(
                    page_index=historical_page.page_index,
                    width=historical_page.width,
                    height=historical_page.height,
                    blocks=tuple(blocks),
                )
            )
        try:
            return PhysicalDocumentV1(
                physical_ir_version=2,
                document_id=historical.document_id,
                version_id=historical.version_id,
                source_artifact_sha256=historical.source_artifact_sha256,
                parser=historical.parser,
                parser_version=historical.parser_version,
                parser_backend=historical.parser_backend,
                page_count=historical.page_count,
                pages=tuple(pages),
            )
        except ValidationError as exc:
            raise MarkerNormalizationError(f"invalid Marker Physical IR v1:\n{exc}") from exc

    @staticmethod
    def _enrich_block(historical: PhysicalBlock, raw_block: Mapping[str, Any]) -> PhysicalBlockV1:
        raw_type = raw_block.get("block_type")
        if not isinstance(raw_type, str) or not raw_type:
            raise MarkerNormalizationError("Marker raw block has invalid block_type")
        kind = {
            "Text": BlockKindV1.TEXT,
            "SectionHeader": BlockKindV1.TITLE,
            "PageHeader": BlockKindV1.HEADER,
            "Table": BlockKindV1.TABLE,
            "Figure": BlockKindV1.FIGURE,
            "Picture": BlockKindV1.IMAGE,
        }.get(raw_type, BlockKindV1.UNKNOWN)

        table_structure = None
        if kind == BlockKindV1.TABLE:
            html = raw_block.get("html")
            if isinstance(html, str):
                try:
                    table_structure = table_structure_from_html(html)
                except TableHTMLStructureError:
                    table_structure = None

        visual_asset: VisualAssetEvidence | None = None
        if kind in {BlockKindV1.FIGURE, BlockKindV1.IMAGE}:
            visual_asset = VisualAssetEvidence(storage_kind=VisualAssetStorage.UNAVAILABLE)
            images = raw_block.get("images")
            raw_id = raw_block.get("id")
            if isinstance(images, Mapping) and isinstance(raw_id, str):
                encoded = images.get(raw_id)
                if isinstance(encoded, str):
                    try:
                        visual_asset = embedded_visual_asset(encoded)
                    except VisualAssetSourceError as exc:
                        raise MarkerNormalizationError(str(exc)) from exc

        text_extraction = (
            TextExtractionEvidence(method=TextExtractionMethod.NATIVE_TEXT, confidence=None)
            if historical.text
            else None
        )
        provenance = historical.provenance
        try:
            return PhysicalBlockV1(
                id=historical.id,
                page_index=historical.page_index,
                reading_order=historical.reading_order,
                kind=kind,
                disposition=historical.disposition,
                text=historical.text,
                bbox=historical.bbox,
                heading_level=historical.heading_level if kind == BlockKindV1.TITLE else None,
                provenance=BlockProvenanceV1(
                    parser=provenance.parser,
                    parser_version=provenance.parser_version,
                    parser_backend=provenance.parser_backend,
                    source_raw_artifact=provenance.source_raw_artifact,
                    source_raw_index=provenance.source_raw_index,
                    source_raw_type=raw_type,
                ),
                text_extraction=text_extraction,
                table_structure=table_structure,
                visual_asset=visual_asset,
            )
        except ValidationError as exc:
            raise MarkerNormalizationError(f"cannot enrich Marker block as v1:\n{exc}") from exc

    def normalize_to_file(
        self,
        raw_directory: Path,
        output_path: Path,
        *,
        source_artifact_path: Path | None = None,
        indent: int | None = 2,
        **kwargs: Any,
    ) -> PhysicalDocumentV1:
        """Atomically write deterministic v1 bytes outside authoritative evidence."""
        self._protect_evidence(raw_directory, output_path, source_artifact_path)
        document = self.normalize(raw_directory, **kwargs)
        payload = physical_document_v1_to_json(document, indent=indent).encode("utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary = Path(name)
            temporary.write_bytes(payload)
            os.replace(temporary, output_path)
        except OSError as exc:
            raise MarkerNormalizationError(f"cannot write v1 output {output_path}: {exc}") from exc
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
        return document

    @staticmethod
    def _protect_evidence(
        raw_directory: Path, output_path: Path, source_artifact_path: Path | None
    ) -> None:
        if not raw_directory.is_dir():
            raise MarkerNormalizationError(f"raw directory does not exist: {raw_directory}")
        raw = raw_directory.resolve()
        output = output_path.resolve()
        if output == raw or output.is_relative_to(raw):
            raise MarkerNormalizationError("v1 output must not be inside the raw directory")
        if source_artifact_path is not None and output == source_artifact_path.resolve():
            raise MarkerNormalizationError("v1 output would overwrite the source artifact")
        protected = {"run.json", "stdout.log", "stderr.log", "artifact_manifest.json"}
        if output.parent == raw.parent and output.name in protected:
            raise MarkerNormalizationError(f"v1 output would overwrite {output.name}")


__all__ = ["MarkerPhysicalNormalizerV1"]
