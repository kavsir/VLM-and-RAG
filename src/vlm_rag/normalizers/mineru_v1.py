"""MinerU raw-output normalization into Physical Document IR v1."""

import json
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vlm_rag.normalizers._assets_v1 import VisualAssetSourceError, relative_visual_asset
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError
from vlm_rag.parsers.mineru import ParserRun
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


class MinerUPhysicalNormalizerV1:
    """Enrich the frozen MinerU v0 mapping from retained raw evidence."""

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
    ) -> PhysicalDocumentV1:
        """Normalize in memory without using annotations or writing derived assets."""
        arguments: dict[str, Any] = {
            "manifest": manifest,
            "run": run,
            "document_id": document_id,
            "version_id": version_id,
            "source_artifact_sha256": source_artifact_sha256,
            "parser": parser,
            "parser_version": parser_version,
            "parser_backend": parser_backend,
        }
        historical = MinerUPhysicalNormalizer().normalize(raw_directory, **arguments)
        content_path = self._locate_content_list(raw_directory)
        try:
            raw_items = json.loads(content_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NormalizationError(
                f"cannot read MinerU content list {content_path}: {exc}"
            ) from exc
        if not isinstance(raw_items, list):
            raise NormalizationError("MinerU content list changed after v0 validation")

        pages: list[PhysicalPageV1] = []
        for historical_page in historical.pages:
            blocks: list[PhysicalBlockV1] = []
            for historical_block in historical_page.blocks:
                raw_index = historical_block.provenance.source_raw_index
                if raw_index >= len(raw_items) or not isinstance(raw_items[raw_index], Mapping):
                    raise NormalizationError("MinerU source_raw_index does not identify an object")
                blocks.append(
                    self._enrich_block(
                        historical_block,
                        raw_items[raw_index],
                        raw_directory,
                        content_path.parent,
                    )
                )
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
            raise NormalizationError(f"invalid MinerU Physical IR v1:\n{exc}") from exc

    @staticmethod
    def _locate_content_list(raw_directory: Path) -> Path:
        matches = sorted(
            path
            for path in raw_directory.rglob("*_content_list.json")
            if path.is_file() and path.name.endswith("_content_list.json")
        )
        if len(matches) != 1:
            raise NormalizationError(
                f"expected one MinerU content list after v0 validation, found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _enrich_block(
        historical: PhysicalBlock,
        raw_item: Mapping[str, Any],
        raw_directory: Path,
        referring_directory: Path,
    ) -> PhysicalBlockV1:
        raw_type = raw_item.get("type")
        if not isinstance(raw_type, str) or not raw_type:
            raise NormalizationError("MinerU raw item has invalid type")
        if raw_type == "text":
            kind = BlockKindV1.TITLE if "text_level" in raw_item else BlockKindV1.TEXT
        else:
            kind = {
                "header": BlockKindV1.HEADER,
                "page_number": BlockKindV1.PAGE_NUMBER,
                "table": BlockKindV1.TABLE,
                "image": BlockKindV1.IMAGE,
            }.get(raw_type, BlockKindV1.UNKNOWN)

        table_structure = None
        if kind == BlockKindV1.TABLE:
            table_body = raw_item.get("table_body")
            if isinstance(table_body, str):
                try:
                    table_structure = table_structure_from_html(table_body)
                except TableHTMLStructureError:
                    table_structure = None

        visual_asset: VisualAssetEvidence | None = None
        if kind in {BlockKindV1.FIGURE, BlockKindV1.IMAGE}:
            visual_asset = VisualAssetEvidence(storage_kind=VisualAssetStorage.UNAVAILABLE)
            parser_path = raw_item.get("img_path")
            if isinstance(parser_path, str):
                try:
                    visual_asset = relative_visual_asset(
                        raw_directory, referring_directory, parser_path
                    )
                except VisualAssetSourceError as exc:
                    raise NormalizationError(str(exc)) from exc

        text_extraction = (
            TextExtractionEvidence(method=TextExtractionMethod.UNKNOWN, confidence=None)
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
            raise NormalizationError(f"cannot enrich MinerU block as v1:\n{exc}") from exc

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
            raise NormalizationError(f"cannot write v1 output {output_path}: {exc}") from exc
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
            raise NormalizationError(f"raw directory does not exist: {raw_directory}")
        raw = raw_directory.resolve()
        output = output_path.resolve()
        if output == raw or output.is_relative_to(raw):
            raise NormalizationError("v1 output must not be inside the raw directory")
        if source_artifact_path is not None and output == source_artifact_path.resolve():
            raise NormalizationError("v1 output would overwrite the source artifact")
        protected = {"run.json", "stdout.log", "stderr.log", "artifact_manifest.json"}
        if output.parent == raw.parent and output.name in protected:
            raise NormalizationError(f"v1 output would overwrite {output.name}")


__all__ = ["MinerUPhysicalNormalizerV1"]
