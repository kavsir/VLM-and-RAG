"""Collect and render retained-corpus evidence for Physical Document IR v1."""

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from vlm_rag.normalizers.marker import MarkerPhysicalNormalizer
from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1
from vlm_rag.physical_ir.models import PhysicalDocument
from vlm_rag.physical_ir.serialization import physical_document_to_json
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.physical_ir.table_html import TableHTMLStructureError, table_structure_from_html
from vlm_rag.physical_ir.v1 import PhysicalDocumentV1


def _read_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return value


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nested_counts(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for (outer, inner), count in sorted(counter.items()):
        result.setdefault(outer, {})[inner] = count
    return result


def _merge_nested_counts(counter: Counter[tuple[str, str]], values: Mapping[str, object]) -> None:
    for outer, raw_inner in values.items():
        if not isinstance(raw_inner, Mapping):
            raise ValueError("expected nested validation counts")
        for inner, raw_count in raw_inner.items():
            if not isinstance(inner, str) or not isinstance(raw_count, int):
                raise ValueError("invalid nested validation count")
            counter[(outer, inner)] += raw_count


def _raw_records(parser: str, raw_directory: Path) -> list[Mapping[str, Any]]:
    if parser == "marker":
        candidates = [
            path
            for path in raw_directory.rglob("*.json")
            if not path.name.casefold().endswith("_meta.json")
        ]
        roots = [_read_json_object(path) for path in candidates]
        documents = [root for root in roots if root.get("block_type") == "Document"]
        if len(documents) != 1 or not isinstance(documents[0].get("children"), list):
            raise ValueError("cannot identify one Marker document for rowspan audit")
        records: list[Mapping[str, Any]] = []
        for page in documents[0]["children"]:
            if not isinstance(page, Mapping) or not isinstance(page.get("children"), list):
                raise ValueError("invalid Marker page during rowspan audit")
            for block in page["children"]:
                if not isinstance(block, Mapping):
                    raise ValueError("invalid Marker block during rowspan audit")
                records.append(block)
        return records
    matches = sorted(path for path in raw_directory.rglob("*_content_list.json") if path.is_file())
    if len(matches) != 1:
        raise ValueError("cannot identify one MinerU content list for rowspan audit")
    value: object = json.loads(matches[0].read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError("invalid MinerU content list during rowspan audit")
    return list(value)


def _rowspan_audit(parser: str, raw_directory: Path, blocks: list[Any]) -> dict[str, int]:
    records = _raw_records(parser, raw_directory)
    markup_count = 0
    rowspan_markup_count = 0
    explicit_row_violations = 0
    for block in blocks:
        if block.kind.value != "table":
            continue
        raw_index = block.provenance.source_raw_index
        if raw_index >= len(records):
            raise ValueError("table provenance is outside raw records during rowspan audit")
        record = records[raw_index]
        field = "html" if parser == "marker" else "table_body"
        html = record.get(field)
        if not isinstance(html, str) or "<table" not in html.casefold():
            continue
        markup_count += 1
        if "rowspan" in html.casefold():
            rowspan_markup_count += 1
        try:
            table_structure_from_html(html)
        except TableHTMLStructureError as exc:
            if "rowspan exceeds the number of explicit tr rows" in str(exc):
                explicit_row_violations += 1
    return {
        "table_markup_count": markup_count,
        "rowspan_markup_table_count": rowspan_markup_count,
        "rowspan_explicit_row_violation_count": explicit_row_violations,
    }


def _normalize_v0(parser: str, raw_directory: Path) -> PhysicalDocument:
    if parser == "marker":
        return MarkerPhysicalNormalizer().normalize(raw_directory)
    if parser == "mineru":
        return MinerUPhysicalNormalizer().normalize(raw_directory)
    raise ValueError(f"unsupported parser in determinism evidence: {parser!r}")


def _normalize_v1(parser: str, raw_directory: Path) -> PhysicalDocumentV1:
    if parser == "marker":
        return MarkerPhysicalNormalizerV1().normalize(raw_directory)
    if parser == "mineru":
        return MinerUPhysicalNormalizerV1().normalize(raw_directory)
    raise ValueError(f"unsupported parser in determinism evidence: {parser!r}")


def _summarize_pair(
    root: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any], PhysicalDocument, PhysicalDocumentV1]:
    parser = str(expected["parser"])
    raw_relative = str(expected["raw_directory"])
    raw_directory = root / raw_relative
    if not raw_directory.is_dir():
        raise ValueError(f"retained raw evidence is unavailable: {raw_directory}")

    historical = _normalize_v0(parser, raw_directory)
    historical_bytes = physical_document_to_json(historical).encode("utf-8")
    expected_sha = str(expected["sha256_a"])
    expected_size = int(expected["byte_size_a"])
    actual_v0_sha = _hash(historical_bytes)
    v0_preserved = actual_v0_sha == expected_sha and len(historical_bytes) == expected_size
    if not v0_preserved:
        raise ValueError(
            f"frozen v0 mismatch for {historical.document_id}/{parser}: "
            f"expected {expected_sha}/{expected_size}, got {actual_v0_sha}/{len(historical_bytes)}"
        )

    current_a = _normalize_v1(parser, raw_directory)
    current_b = _normalize_v1(parser, raw_directory)
    payload_a = physical_document_v1_to_json(current_a).encode("utf-8")
    payload_b = physical_document_v1_to_json(current_b).encode("utf-8")
    if current_a.source_artifact_sha256 != historical.source_artifact_sha256:
        raise ValueError("v1 normalization changed the authoritative source SHA-256")

    historical_blocks = [block for page in historical.pages for block in page.blocks]
    current_blocks = [block for page in current_a.pages for block in page.blocks]
    if [block.id for block in historical_blocks] != [block.id for block in current_blocks]:
        raise ValueError("v1 normalization changed the frozen block identity/order sequence")

    kind_histogram = Counter(block.kind.value for block in current_blocks)
    extraction_histogram = Counter(
        block.text_extraction.method.value if block.text_extraction else "not_recorded"
        for block in current_blocks
    )
    transitions = Counter(
        f"{old.kind.value}->{new.kind.value}"
        for old, new in zip(historical_blocks, current_blocks, strict=True)
        if old.kind.value != new.kind.value
    )
    tables = [block for block in current_blocks if block.kind.value == "table"]
    structures = [block.table_structure for block in tables if block.table_structure is not None]
    cells = [cell for structure in structures for cell in structure.cells]
    visual_evidence = [block.visual_asset for block in current_blocks if block.visual_asset]
    kind_disposition = Counter(
        (block.kind.value, block.disposition.value) for block in current_blocks
    )
    raw_type_extraction = Counter(
        (
            block.provenance.source_raw_type,
            block.text_extraction.method.value if block.text_extraction else "not_recorded",
        )
        for block in current_blocks
    )
    unknown_kind_by_raw_type = Counter(
        block.provenance.source_raw_type
        for block in current_blocks
        if block.kind.value == "unknown"
    )
    unknown_disposition_by_raw_type = Counter(
        block.provenance.source_raw_type
        for block in current_blocks
        if block.disposition.value == "unknown"
    )
    unknown_extraction_by_raw_type = Counter(
        block.provenance.source_raw_type
        for block in current_blocks
        if block.text_extraction is not None and block.text_extraction.method.value == "unknown"
    )
    rowspan_audit = _rowspan_audit(parser, raw_directory, current_blocks)
    result: dict[str, Any] = {
        "document_id": current_a.document_id,
        "version_id": current_a.version_id,
        "parser": parser,
        "raw_directory": raw_relative,
        "source_artifact_sha256": current_a.source_artifact_sha256,
        "pages": current_a.page_count,
        "total_blocks": len(current_blocks),
        "kind_histogram": dict(sorted(kind_histogram.items())),
        "kind_disposition_matrix": _nested_counts(kind_disposition),
        "table_count": len(tables),
        "tables_with_structure": len(structures),
        "table_cell_count": len(cells),
        "rowspan_cell_count": sum(cell.row_span > 1 for cell in cells),
        "colspan_cell_count": sum(cell.column_span > 1 for cell in cells),
        "header_cell_count": sum(cell.is_header is True for cell in cells),
        "figure_count": kind_histogram["figure"],
        "image_count": kind_histogram["image"],
        "unknown_count": kind_histogram["unknown"],
        "text_extraction_method_histogram": dict(sorted(extraction_histogram.items())),
        "source_raw_type_extraction_method_matrix": _nested_counts(raw_type_extraction),
        "unknown_kind_by_source_raw_type": dict(sorted(unknown_kind_by_raw_type.items())),
        "unknown_disposition_by_source_raw_type": dict(
            sorted(unknown_disposition_by_raw_type.items())
        ),
        "unknown_text_extraction_by_source_raw_type": dict(
            sorted(unknown_extraction_by_raw_type.items())
        ),
        "explicit_text_extraction_count": sum(
            block.text_extraction is not None and block.text_extraction.method.value != "unknown"
            for block in current_blocks
        ),
        "ocr_confidence_count": sum(
            block.text_extraction is not None and block.text_extraction.confidence is not None
            for block in current_blocks
        ),
        "visual_asset_evidence_count": len(visual_evidence),
        "visual_asset_hash_count": sum(asset.sha256 is not None for asset in visual_evidence),
        "verified_visual_asset_count": sum(
            asset.storage_kind.value in {"relative_file", "embedded_raw"}
            and asset.sha256 is not None
            and asset.byte_size is not None
            for asset in visual_evidence
        ),
        "visual_asset_storage_histogram": dict(
            sorted(Counter(asset.storage_kind.value for asset in visual_evidence).items())
        ),
        **rowspan_audit,
        "v0_expected_byte_size": expected_size,
        "v0_actual_byte_size": len(historical_bytes),
        "v0_expected_sha256": expected_sha,
        "v0_actual_sha256": actual_v0_sha,
        "v0_preserved": v0_preserved,
        "v0_to_v1_kind_transitions": dict(sorted(transitions.items())),
        "byte_size_a": len(payload_a),
        "sha256_a": _hash(payload_a),
        "byte_size_b": len(payload_b),
        "sha256_b": _hash(payload_b),
        "equal": payload_a == payload_b,
    }
    if current_a.document_id == "tt-04-2023-bkhdt-so-do-ban-do":
        selected = [
            {
                "page_index": block.page_index,
                "block_id": block.id,
                "source_raw_index": block.provenance.source_raw_index,
                "source_raw_type": block.provenance.source_raw_type,
                "normalized_kind": block.kind.value,
            }
            for block in current_blocks
            if block.page_index == 21
            and (
                (parser == "marker" and block.provenance.source_raw_type == "Figure")
                or (parser == "mineru" and block.provenance.source_raw_type == "table")
            )
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected exactly one TT04/2023 page-index-21 {parser} anti-leakage object"
            )
        result["anti_evaluation_leakage_observation"] = selected[0]
    return result, historical, current_a


def collect_physical_ir_v1_validation(root: Path) -> dict[str, Any]:
    """Re-normalize all ten retained runs twice and return machine evidence."""
    source_path = root / "data/benchmarks/normalization_determinism.v1.json"
    source = _read_json_object(source_path)
    raw_entries = source.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"missing entries in {source_path}")

    entries: list[dict[str, Any]] = []
    aggregate_kinds: Counter[str] = Counter()
    aggregate_extraction: Counter[str] = Counter()
    aggregate_transitions: Counter[str] = Counter()
    aggregate_kind_disposition: Counter[tuple[str, str]] = Counter()
    aggregate_raw_type_extraction: dict[str, Counter[tuple[str, str]]] = {
        "marker": Counter(),
        "mineru": Counter(),
    }
    aggregate_unknown_kind: dict[str, Counter[str]] = {
        "marker": Counter(),
        "mineru": Counter(),
    }
    aggregate_unknown_disposition: dict[str, Counter[str]] = {
        "marker": Counter(),
        "mineru": Counter(),
    }
    aggregate_unknown_extraction: dict[str, Counter[str]] = {
        "marker": Counter(),
        "mineru": Counter(),
    }
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("v0 determinism entry must be an object")
        entry, _, _ = _summarize_pair(root, raw_entry)
        entries.append(entry)
        aggregate_kinds.update(entry["kind_histogram"])
        aggregate_extraction.update(entry["text_extraction_method_histogram"])
        aggregate_transitions.update(entry["v0_to_v1_kind_transitions"])
        _merge_nested_counts(aggregate_kind_disposition, entry["kind_disposition_matrix"])
        parser = str(entry["parser"])
        _merge_nested_counts(
            aggregate_raw_type_extraction[parser],
            entry["source_raw_type_extraction_method_matrix"],
        )
        aggregate_unknown_kind[parser].update(entry["unknown_kind_by_source_raw_type"])
        aggregate_unknown_disposition[parser].update(
            entry["unknown_disposition_by_source_raw_type"]
        )
        aggregate_unknown_extraction[parser].update(
            entry["unknown_text_extraction_by_source_raw_type"]
        )

    def total(field: str) -> int:
        return sum(int(entry[field]) for entry in entries)

    anti_leakage = {
        str(entry["parser"]): entry["anti_evaluation_leakage_observation"]
        for entry in entries
        if "anti_evaluation_leakage_observation" in entry
    }
    expected_anti_leakage = {
        "marker": ("Figure", "figure"),
        "mineru": ("table", "table"),
    }
    for parser, (raw_type, kind) in expected_anti_leakage.items():
        observation = anti_leakage.get(parser)
        if not isinstance(observation, dict) or (
            observation.get("source_raw_type"),
            observation.get("normalized_kind"),
        ) != (raw_type, kind):
            raise ValueError(f"TT04/2023 anti-evaluation-leakage check failed for {parser}")

    by_parser: dict[str, dict[str, Any]] = {}
    for parser in ("marker", "mineru"):
        parser_entries = [entry for entry in entries if entry["parser"] == parser]
        parser_kind_disposition: Counter[tuple[str, str]] = Counter()
        for entry in parser_entries:
            _merge_nested_counts(parser_kind_disposition, entry["kind_disposition_matrix"])
        by_parser[parser] = {
            "pairs": len(parser_entries),
            "table_count": sum(int(entry["table_count"]) for entry in parser_entries),
            "tables_with_structure": sum(
                int(entry["tables_with_structure"]) for entry in parser_entries
            ),
            "table_cell_count": sum(int(entry["table_cell_count"]) for entry in parser_entries),
            "figure_count": sum(int(entry["figure_count"]) for entry in parser_entries),
            "image_count": sum(int(entry["image_count"]) for entry in parser_entries),
            "explicit_text_extraction_count": sum(
                int(entry["explicit_text_extraction_count"]) for entry in parser_entries
            ),
            "ocr_confidence_count": sum(
                int(entry["ocr_confidence_count"]) for entry in parser_entries
            ),
            "visual_asset_hash_count": sum(
                int(entry["visual_asset_hash_count"]) for entry in parser_entries
            ),
            "verified_visual_asset_count": sum(
                int(entry["verified_visual_asset_count"]) for entry in parser_entries
            ),
            "kind_disposition_matrix": _nested_counts(parser_kind_disposition),
            "source_raw_type_extraction_method_matrix": _nested_counts(
                aggregate_raw_type_extraction[parser]
            ),
            "unknown_kind_by_source_raw_type": dict(sorted(aggregate_unknown_kind[parser].items())),
            "unknown_disposition_by_source_raw_type": dict(
                sorted(aggregate_unknown_disposition[parser].items())
            ),
            "unknown_text_extraction_by_source_raw_type": dict(
                sorted(aggregate_unknown_extraction[parser].items())
            ),
            "rowspan_explicit_row_violation_count": sum(
                int(entry["rowspan_explicit_row_violation_count"]) for entry in parser_entries
            ),
        }

    return {
        "validation_schema_version": 2,
        "validation_protocol": "physical_ir_v1_retained_corpus",
        "validation_protocol_revision": 2,
        "physical_ir_research_generation": "v1",
        "physical_ir_wire_schema_version": 2,
        "input_v0_determinism_evidence": source_path.relative_to(root).as_posix(),
        "method": (
            "two fresh v1 normalizations from each retained raw run; fresh v0 normalization "
            "compared with frozen Issue #006 SHA and byte size"
        ),
        "normalizer_inputs": (
            "raw parser artifacts and run provenance only; annotations and evaluation reports "
            "are excluded"
        ),
        "available_pairs": len(entries),
        "all_v0_hashes_preserved": all(bool(entry["v0_preserved"]) for entry in entries),
        "all_v1_equal": all(bool(entry["equal"]) for entry in entries),
        "anti_evaluation_leakage_case": {
            "document_id": "tt-04-2023-bkhdt-so-do-ban-do",
            "pdf_page_number_1_based": 22,
            "page_index": 21,
            "normalizer_input_policy": "raw parser evidence only; reference annotations excluded",
            "observations": anti_leakage,
        },
        "entries": entries,
        "aggregate": {
            "pages": total("pages"),
            "total_blocks": total("total_blocks"),
            "kind_histogram": dict(sorted(aggregate_kinds.items())),
            "kind_disposition_matrix": _nested_counts(aggregate_kind_disposition),
            "table_count": total("table_count"),
            "tables_with_structure": total("tables_with_structure"),
            "table_cell_count": total("table_cell_count"),
            "rowspan_cell_count": total("rowspan_cell_count"),
            "colspan_cell_count": total("colspan_cell_count"),
            "header_cell_count": total("header_cell_count"),
            "figure_count": total("figure_count"),
            "image_count": total("image_count"),
            "unknown_count": total("unknown_count"),
            "text_extraction_method_histogram": dict(sorted(aggregate_extraction.items())),
            "source_raw_type_extraction_method_matrix_by_parser": {
                parser: _nested_counts(counter)
                for parser, counter in aggregate_raw_type_extraction.items()
            },
            "unknown_kind_by_parser_source_raw_type": {
                parser: dict(sorted(counter.items()))
                for parser, counter in aggregate_unknown_kind.items()
            },
            "unknown_disposition_by_parser_source_raw_type": {
                parser: dict(sorted(counter.items()))
                for parser, counter in aggregate_unknown_disposition.items()
            },
            "unknown_text_extraction_by_parser_source_raw_type": {
                parser: dict(sorted(counter.items()))
                for parser, counter in aggregate_unknown_extraction.items()
            },
            "explicit_text_extraction_count": total("explicit_text_extraction_count"),
            "ocr_confidence_count": total("ocr_confidence_count"),
            "visual_asset_evidence_count": total("visual_asset_evidence_count"),
            "visual_asset_hash_count": total("visual_asset_hash_count"),
            "verified_visual_asset_count": total("verified_visual_asset_count"),
            "table_markup_count": total("table_markup_count"),
            "rowspan_markup_table_count": total("rowspan_markup_table_count"),
            "rowspan_explicit_row_violation_count": total("rowspan_explicit_row_violation_count"),
            "v0_to_v1_kind_transitions": dict(sorted(aggregate_transitions.items())),
            "by_parser": by_parser,
        },
    }


def render_physical_ir_v1_validation(evidence: dict[str, Any]) -> str:
    """Render the human report exclusively from machine validation evidence."""
    entries = evidence["entries"]
    aggregate = evidence["aggregate"]
    if not isinstance(entries, list) or not isinstance(aggregate, dict):
        raise ValueError("invalid Physical IR v1 validation evidence")
    lines = [
        "# Physical IR v1 retained-corpus validation",
        "",
        (
            "> Generated from `data/benchmarks/physical_ir_v1_validation.v1.json`. "
            "Do not edit measured values by hand."
        ),
        "",
        (
            "Physical IR research generation **v1** uses wire/schema version **2**. "
            "Historical Physical IR v0 remains wire version 1."
        ),
        (
            "This file remains `.v1.json` because that is the Physical IR v1 validation "
            "protocol name; `validation_schema_version: 2` records the corrected evidence "
            "artifact shape."
        ),
        "",
        "## Validation result",
        "",
        f"- Retained parser/document pairs: **{evidence['available_pairs']}**",
        f"- Frozen v0 SHA/size checks preserved: **{evidence['all_v0_hashes_preserved']}**",
        f"- Byte-identical v1 A/B normalizations: **{evidence['all_v1_equal']}**",
        f"- Pages / blocks: **{aggregate['pages']} / {aggregate['total_blocks']}**",
        "",
        "## Per-pair representation",
        "",
        (
            "| Document | Parser | Pages | Blocks | Tables | Structured | Cells | Figures | "
            "Images | Unknown | v1 SHA-256 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        lines.append(
            f"| {entry['document_id']} | {entry['parser']} | {entry['pages']} | "
            f"{entry['total_blocks']} | {entry['table_count']} | "
            f"{entry['tables_with_structure']} | {entry['table_cell_count']} | "
            f"{entry['figure_count']} | {entry['image_count']} | {entry['unknown_count']} | "
            f"`{entry['sha256_a']}` |"
        )
    by_parser = aggregate["by_parser"]
    if not isinstance(by_parser, dict):
        raise ValueError("missing per-parser evidence")
    lines.extend(
        [
            "",
            "## Table structure coverage by parser",
            "",
            "| Parser | Native tables | Structured tables | Cells |",
            "|---|---:|---:|---:|",
        ]
    )
    for parser in ("marker", "mineru"):
        parser_summary = by_parser[parser]
        if not isinstance(parser_summary, dict):
            raise ValueError(f"invalid per-parser evidence for {parser}")
        lines.append(
            f"| {parser} | {parser_summary['table_count']} | "
            f"{parser_summary['tables_with_structure']} | "
            f"{parser_summary['table_cell_count']} |"
        )
    lines.extend(
        [
            "",
            "## Kind by disposition by parser",
            "",
            (
                "Disposition describes the current v1 physical observation: `content` is body "
                "content, `discarded` is intentional boilerplate exclusion, and `unknown` means "
                "the disposition cannot be determined. It is not copied v0 kind uncertainty."
            ),
            "",
            "| Parser | Kind | Content | Discarded | Unknown |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for parser in ("marker", "mineru"):
        parser_summary = by_parser[parser]
        matrix = parser_summary["kind_disposition_matrix"]
        if not isinstance(matrix, dict):
            raise ValueError(f"invalid kind/disposition matrix for {parser}")
        for kind, raw_counts in matrix.items():
            if not isinstance(raw_counts, dict):
                raise ValueError("invalid kind/disposition row")
            lines.append(
                f"| {parser} | {kind} | {raw_counts.get('content', 0)} | "
                f"{raw_counts.get('discarded', 0)} | {raw_counts.get('unknown', 0)} |"
            )
    transitions = aggregate["v0_to_v1_kind_transitions"]
    extraction = aggregate["text_extraction_method_histogram"]
    anti_leakage = evidence["anti_evaluation_leakage_case"]
    if not isinstance(anti_leakage, dict):
        raise ValueError("missing anti-evaluation-leakage evidence")
    observations = anti_leakage["observations"]
    if not isinstance(observations, dict):
        raise ValueError("missing anti-evaluation-leakage observations")
    marker_observation = observations["marker"]
    mineru_observation = observations["mineru"]
    if not isinstance(marker_observation, dict) or not isinstance(mineru_observation, dict):
        raise ValueError("invalid anti-evaluation-leakage observations")
    lines.extend(
        [
            "",
            "## Representation fidelity gained from raw evidence",
            "",
            (
                f"- Tables: **{aggregate['table_count']}**; recoverable logical structure: "
                f"**{aggregate['tables_with_structure']}**; cells: "
                f"**{aggregate['table_cell_count']}**."
            ),
            (
                f"- Table spans: rowspan **{aggregate['rowspan_cell_count']}**, colspan "
                f"**{aggregate['colspan_cell_count']}**, parser `<th>` cells "
                f"**{aggregate['header_cell_count']}**."
            ),
            (
                "- Explicit-row rowspan audit: "
                f"**{aggregate['rowspan_explicit_row_violation_count']}** violations across "
                f"**{aggregate['rowspan_markup_table_count']}** table markups containing "
                "`rowspan`. A cell may not extend beyond the observed `<tr>` count."
            ),
            (
                f"- Visuals: FIGURE **{aggregate['figure_count']}**, IMAGE "
                f"**{aggregate['image_count']}**; asset hashes "
                f"**{aggregate['visual_asset_hash_count']}/"
                f"{aggregate['visual_asset_evidence_count']}**."
            ),
            (
                "- Verified visual assets satisfying the storage contract: "
                f"**{aggregate['verified_visual_asset_count']}/"
                f"{aggregate['visual_asset_evidence_count']}**. A `relative_file` requires a "
                "safe path, SHA-256, and positive byte size; media type may be null. Missing "
                "referenced bytes are represented as `unavailable`."
            ),
            f"- Kind transitions from frozen v0: `{json.dumps(transitions, sort_keys=True)}`.",
            (
                f"- Extraction methods: `{json.dumps(extraction, sort_keys=True)}`; "
                "confidence coverage "
                f"**{aggregate['ocr_confidence_count']}**."
            ),
            "",
            (
                "These are representation-fidelity changes, not parser-accuracy improvements. "
                f"On TT04/2023 page index {anti_leakage['page_index']}, Marker raw "
                f"`{marker_observation['source_raw_type']}` maps to "
                f"{str(marker_observation['normalized_kind']).upper()}, while MinerU raw "
                f"`{mineru_observation['source_raw_type']}` maps to "
                f"{str(mineru_observation['normalized_kind']).upper()}."
            ),
            "",
            "## Extraction evidence policy",
            "",
            (
                "Marker non-empty blocks are `native_text` only where both the validated run "
                "records `disable_ocr=true` and that page's retained `source_meta.json` records "
                "`text_extraction_method=pdftext`; empty blocks carry no text-extraction record. "
                "Missing or different page/provider evidence is `unknown`. MinerU 3.4.5 "
                "pipeline artifacts expose span/layout scores but no field "
                "distinguishing OCR from native PDF text, so non-empty MinerU text is `unknown` "
                "and confidence remains null. Annotation modality labels are never normalizer "
                "inputs."
            ),
            "",
            "### Marker source raw type by extraction method",
            "",
            "| Source raw type | Native text | OCR | Unknown | Not recorded |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    marker_matrix = by_parser["marker"]["source_raw_type_extraction_method_matrix"]
    if not isinstance(marker_matrix, dict):
        raise ValueError("missing Marker extraction matrix")
    for raw_type, raw_counts in marker_matrix.items():
        if not isinstance(raw_counts, dict):
            raise ValueError("invalid Marker extraction matrix row")
        lines.append(
            f"| {raw_type} | {raw_counts.get('native_text', 0)} | "
            f"{raw_counts.get('ocr', 0)} | {raw_counts.get('unknown', 0)} | "
            f"{raw_counts.get('not_recorded', 0)} |"
        )
    unknown_kind = aggregate["unknown_kind_by_parser_source_raw_type"]
    unknown_extraction = aggregate["unknown_text_extraction_by_parser_source_raw_type"]
    unknown_kind_json = json.dumps(unknown_kind, sort_keys=True)
    unknown_extraction_json = json.dumps(unknown_extraction, sort_keys=True)
    lines.extend(
        [
            "",
            "## Remaining unknown evidence",
            "",
            f"- UNKNOWN kind by parser/source raw type: `{unknown_kind_json}`.",
            (f"- UNKNOWN text extraction by parser/source raw type: `{unknown_extraction_json}`."),
            "",
            "## Limitations",
            "",
            (
                "`table_structure: null` can mean absent markup or rejected markup; v1 does not "
                "add a field distinguishing those causes. Tables whose observed HTML cannot be "
                "converted safely retain TABLE identity. UNKNOWN remains intentional for "
                "unsupported raw types. No OCR "
                "accuracy, semantic map meaning, or structural/legal interpretation is claimed."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_physical_ir_v1_validation(root: Path, *, collect: bool) -> None:
    """Collect machine evidence when requested and always render its report."""
    evidence_path = root / "data/benchmarks/physical_ir_v1_validation.v1.json"
    if collect:
        evidence = collect_physical_ir_v1_validation(root)
        evidence_path.write_bytes(
            (json.dumps(evidence, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )
    else:
        evidence = _read_json_object(evidence_path)
    report_path = root / "docs/research/physical-ir-v1-validation.md"
    report_path.write_bytes(render_physical_ir_v1_validation(evidence).encode("utf-8"))


__all__ = [
    "collect_physical_ir_v1_validation",
    "render_physical_ir_v1_validation",
    "write_physical_ir_v1_validation",
]
