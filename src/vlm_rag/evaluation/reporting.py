"""Deterministic benchmark evidence collection and offline report rendering."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from vlm_rag.evaluation.evaluator import evaluate_physical_document
from vlm_rag.evaluation.models import DocumentAnnotation
from vlm_rag.evaluation.serialization import dump_evaluation_report
from vlm_rag.normalizers.marker import MarkerPhysicalNormalizer
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer
from vlm_rag.parsers.marker import discover_marker_document_json
from vlm_rag.physical_ir.models import BlockKind, PhysicalDocument
from vlm_rag.registry import load_manifest

BENCHMARK_MANIFEST = Path("data/benchmarks/benchmark_manifest.v1.json")
DETERMINISM_MANIFEST = Path("data/benchmarks/normalization_determinism.v1.json")
ANNOTATION_AUDIT = Path("data/annotations/reference_annotation_audit.v3.json")
REPORT_PATHS = (
    Path("docs/research/parser-benchmark-v1.md"),
    Path("docs/research/corpus-vietnamese-legal-planning-v1.md"),
    Path("docs/research/physical-ir-v1-gaps.md"),
)


def _marker_run(root: Path, doc_slug: str) -> dict[str, Any]:
    base = (
        root
        / "data"
        / "golden"
        / doc_slug
        / "v1"
        / "parser_runs"
        / "marker"
        / "2.0.0"
        / "fast-no-ocr"
    )
    return {
        "parser": "marker",
        "parser_version": "2.0.0",
        "backend": "fast-no-ocr",
        "mode": "fast",
        "configuration": {"disable_ocr": True, "output_format": "json"},
        "raw_directory": base / "raw",
        "physical_ir_path": base / "normalized" / "physical_ir.json",
        "run_json_path": base / "run.json",
        "benchmark_result_path": root / "data" / "benchmarks" / f"marker_{doc_slug}.v1.json",
    }


def _mineru_run(root: Path, doc_slug: str) -> dict[str, Any]:
    base = (
        root / "data" / "golden" / doc_slug / "v1" / "parser_runs" / "mineru" / "3.4.5" / "pipeline"
    )
    return {
        "parser": "mineru",
        "parser_version": "3.4.5",
        "backend": "pipeline",
        "mode": "pipeline",
        "configuration": {"device": "cpu"},
        "raw_directory": base / "raw",
        "physical_ir_path": base / "physical_ir_v0.json",
        "run_json_path": base / "run.json",
        "benchmark_result_path": root / "data" / "benchmarks" / f"mineru_{doc_slug}.v1.json",
    }


def get_corpus_config(root: Path) -> list[dict[str, Any]]:
    """Return the six registered documents and ten available parser runs."""
    definitions = (
        ("hanoi-master-plan-100y", "hanoi_master_plan_100y", ("marker", "mineru")),
        ("luat-112-2025-qh15", "luat_112_2025_qh15", ("marker",)),
        ("vbhn-103-2026-quy-hoach-tong-the", "vbhn_103_2026_quy_hoach_tong_the", ("marker",)),
        ("tt-04-2026-bxd-pl2-dinh-muc", "tt_04_2026_bxd_pl2_dinh_muc", ("marker", "mineru")),
        ("tt-04-2023-bkhdt-so-do-ban-do", "tt_04_2023_bkhdt_so_do_ban_do", ("marker", "mineru")),
        (
            "qd-23-2008-ubnd-hanoi-vien-quy-hoach",
            "qd_23_2008_ubnd_hanoi_vien_quy_hoach",
            ("marker", "mineru"),
        ),
    )
    result: list[dict[str, Any]] = []
    for document_id, slug, parsers in definitions:
        runs = {
            parser: (_marker_run(root, slug) if parser == "marker" else _mineru_run(root, slug))
            for parser in parsers
        }
        result.append(
            {
                "document_id": document_id,
                "slug": slug,
                "manifest_path": root / "data" / "manifests" / f"{slug}.v1.yaml",
                "annotation_path": root / "data" / "annotations" / f"{slug}.v1.json",
                "source_path": root / "data" / "golden" / slug / "v1" / "source.pdf",
                "runs": runs,
            }
        )
    return result


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_evidence(path: Path, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": _relative(path, root),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _duration_seconds(run_data: dict[str, Any]) -> float:
    execution = run_data.get("execution")
    value = execution.get("duration_seconds") if isinstance(execution, dict) else None
    if value is None:
        value = run_data.get("duration_seconds")
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError("run.json must contain a positive duration_seconds value")
    return float(value)


def _validate_run_provenance(
    run_data: dict[str, Any],
    run: dict[str, Any],
    document: PhysicalDocument,
) -> None:
    expected = {
        "status": "succeeded",
        "parser": run["parser"],
        "parser_version": run["parser_version"],
        "backend": run["backend"],
        "document_id": document.document_id,
        "version_id": document.version_id,
        "input_sha256": document.source_artifact_sha256,
    }
    conflicts = {
        field: {"expected": value, "observed": run_data.get(field)}
        for field, value in expected.items()
        if run_data.get(field) != value
    }
    if conflicts:
        raise ValueError(f"run.json provenance conflicts with Physical IR: {conflicts}")


def _raw_artifact_catalog(
    run_data: dict[str, Any], raw_directory: Path, root: Path
) -> list[dict[str, Any]]:
    artifacts = run_data.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("run.json must contain a non-empty raw artifact inventory")
    raw_root = raw_directory.resolve()
    catalog: list[dict[str, Any]] = []
    for item in artifacts:
        if not isinstance(item, dict):
            raise ValueError("run.json artifact entries must be objects")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("run.json artifact relative_path must be a non-empty string")
        artifact_path = (raw_directory / relative_path).resolve()
        if not artifact_path.is_relative_to(raw_root):
            raise ValueError(f"run.json artifact escapes raw directory: {relative_path}")
        evidence = _file_evidence(artifact_path, root)
        if (
            item.get("sha256") != evidence["sha256"]
            or item.get("byte_size") != evidence["byte_size"]
        ):
            raise ValueError(f"raw artifact evidence mismatch: {relative_path}")
        catalog.append(
            {
                "path": evidence["path"],
                "kind": item.get("kind"),
                "sha256": evidence["sha256"],
                "byte_size": evidence["byte_size"],
            }
        )
    return catalog


def _table_object_mapping(run: dict[str, Any], document: PhysicalDocument) -> dict[str, Any]:
    raw_directory = run["raw_directory"]
    if run["parser"] == "marker":
        raw_root = json.loads(
            discover_marker_document_json(raw_directory).read_text(encoding="utf-8")
        )
        raw_index = 0
        table_indices: set[int] = set()
        for page in raw_root["children"]:
            for block in page["children"]:
                if block.get("block_type") == "Table":
                    table_indices.add(raw_index)
                raw_index += 1
        raw_object_name = "Table"
    else:
        candidates = sorted(raw_directory.rglob("*_content_list.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected one MinerU content list below {raw_directory}")
        content = json.loads(candidates[0].read_text(encoding="utf-8"))
        table_indices = {
            index
            for index, item in enumerate(content)
            if isinstance(item, dict) and str(item.get("type", "")).casefold() == "table"
        }
        raw_object_name = "table"
    mapped = Counter(
        block.kind.value
        for page in document.pages
        for block in page.blocks
        if block.provenance.source_raw_index in table_indices
    )
    return {
        "raw_object_name": raw_object_name,
        "raw_table_object_count": len(table_indices),
        "physical_ir_kind_counts": dict(sorted(mapped.items())),
        "interpretation": (
            "Counts follow parser-native table objects through source_raw_index provenance. "
            "Separately emitted nested text is not table-object preservation."
        ),
    }


def _raw_type_counts(run: dict[str, Any]) -> Counter[str]:
    raw_directory = run["raw_directory"]
    if run["parser"] == "marker":
        raw_root = json.loads(
            discover_marker_document_json(raw_directory).read_text(encoding="utf-8")
        )
        return Counter(
            str(block.get("block_type", "unknown"))
            for page in raw_root["children"]
            for block in page["children"]
        )
    candidates = sorted(raw_directory.rglob("*_content_list.json"))
    if len(candidates) != 1:
        raise ValueError(f"expected one MinerU content list below {raw_directory}")
    content = json.loads(candidates[0].read_text(encoding="utf-8"))
    return Counter(
        str(item.get("type", "unknown")) if isinstance(item, dict) else "invalid"
        for item in content
    )


def _layout_diagram_mapping(
    run: dict[str, Any], document: PhysicalDocument, page_index: int
) -> dict[str, Any]:
    raw_directory = run["raw_directory"]
    selected_indices: set[int] = set()
    selected_types: Counter[str] = Counter()
    if run["parser"] == "marker":
        raw_root = json.loads(
            discover_marker_document_json(raw_directory).read_text(encoding="utf-8")
        )
        raw_index = 0
        for current_page_index, page in enumerate(raw_root["children"]):
            for block in page["children"]:
                raw_type = str(block.get("block_type", "unknown"))
                if current_page_index == page_index and raw_type in {"Figure", "Picture"}:
                    selected_indices.add(raw_index)
                    selected_types[raw_type] += 1
                raw_index += 1
    else:
        candidates = sorted(raw_directory.rglob("*_content_list.json"))
        if len(candidates) != 1:
            raise ValueError(f"expected one MinerU content list below {raw_directory}")
        content = json.loads(candidates[0].read_text(encoding="utf-8"))
        for raw_index, item in enumerate(content):
            if not isinstance(item, dict) or item.get("page_idx") != page_index:
                continue
            raw_type = str(item.get("type", "unknown"))
            if raw_type in {"image", "table"}:
                selected_indices.add(raw_index)
                selected_types[raw_type] += 1
    mapped = Counter(
        block.kind.value
        for block in document.pages[page_index].blocks
        if block.provenance.source_raw_index in selected_indices
    )
    return {
        "page_index": page_index,
        "raw_object_type_counts": dict(sorted(selected_types.items())),
        "physical_ir_kind_counts": dict(sorted(mapped.items())),
    }


def _normalize_once(document_item: dict[str, Any], run: dict[str, Any], output_path: Path) -> None:
    manifest = load_manifest(document_item["manifest_path"])
    if run["parser"] == "marker":
        MarkerPhysicalNormalizer().normalize_to_file(
            run["raw_directory"],
            output_path,
            manifest=manifest,
            source_artifact_path=document_item["source_path"],
        )
    else:
        MinerUPhysicalNormalizer().normalize_to_file(
            run["raw_directory"], output_path, manifest=manifest
        )


def _collect_normalization_determinism(root: Path, config: list[dict[str, Any]]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vlm-rag-determinism-") as temporary:
        temporary_root = Path(temporary)
        for document_item in config:
            for parser, run in document_item["runs"].items():
                first = temporary_root / f"{document_item['slug']}-{parser}-a.json"
                second = temporary_root / f"{document_item['slug']}-{parser}-b.json"
                _normalize_once(document_item, run, first)
                _normalize_once(document_item, run, second)
                first_evidence = _file_evidence(first, temporary_root)
                second_evidence = _file_evidence(second, temporary_root)
                entries.append(
                    {
                        "document_id": document_item["document_id"],
                        "version_id": "v1",
                        "parser": parser,
                        "raw_directory": _relative(run["raw_directory"], root),
                        "byte_size_a": first_evidence["byte_size"],
                        "sha256_a": first_evidence["sha256"],
                        "byte_size_b": second_evidence["byte_size"],
                        "sha256_b": second_evidence["sha256"],
                        "equal": first.read_bytes() == second.read_bytes(),
                    }
                )
    if not all(entry["equal"] for entry in entries):
        raise ValueError("normalization determinism failed for at least one retained parser run")
    return {
        "determinism_schema_version": 1,
        "normalization_target": "Physical Document IR v0 deterministic UTF-8/LF JSON",
        "method": "two fresh normalizations from the same retained immutable raw parser evidence",
        "available_pairs": len(entries),
        "all_equal": True,
        "entries": entries,
    }


def collect_benchmark_evidence(root: Path) -> dict[str, Any]:
    """Stage A: evaluate retained external artifacts and write committed machine evidence."""
    config = get_corpus_config(root)
    audit_path = root / ANNOTATION_AUDIT
    audit_evidence = _file_evidence(audit_path, root)
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    documents: list[dict[str, Any]] = []
    scanned_case: dict[str, Any] = {}
    annotations: list[DocumentAnnotation] = []

    for document_item in config:
        manifest = load_manifest(document_item["manifest_path"])
        annotation = DocumentAnnotation.model_validate_json(
            document_item["annotation_path"].read_text(encoding="utf-8")
        )
        annotations.append(annotation)
        annotation_evidence = _file_evidence(document_item["annotation_path"], root)
        if (
            annotation.document_id != manifest.document.id
            or annotation.source_sha256 != manifest.artifact.sha256
        ):
            raise ValueError(
                "annotation identity or source SHA-256 does not match registry manifest"
            )
        run_records: dict[str, Any] = {}
        page_count: int | None = None
        for parser, run in document_item["runs"].items():
            physical_document = PhysicalDocument.model_validate_json(
                run["physical_ir_path"].read_text(encoding="utf-8")
            )
            if page_count is None:
                page_count = physical_document.page_count
            elif page_count != physical_document.page_count:
                raise ValueError("parser runs disagree on document page count")
            run_data = json.loads(run["run_json_path"].read_text(encoding="utf-8"))
            _validate_run_provenance(run_data, run, physical_document)
            duration = _duration_seconds(run_data)
            result = evaluate_physical_document(physical_document, annotation).to_dict()
            result["wall_clock_seconds"] = round(duration, 4)
            result["pages_per_second"] = round(physical_document.page_count / duration, 4)
            result["total_document_pages"] = physical_document.page_count
            dump_evaluation_report(result, run["benchmark_result_path"])
            physical_evidence = _file_evidence(run["physical_ir_path"], root)
            run_evidence = _file_evidence(run["run_json_path"], root)
            result_evidence = _file_evidence(run["benchmark_result_path"], root)
            metrics = result["metrics"]
            run_records[parser] = {
                "parser": physical_document.parser,
                "parser_version": physical_document.parser_version,
                "backend": physical_document.parser_backend,
                "mode": run["mode"],
                "configuration": run["configuration"],
                "source_artifact_sha256": physical_document.source_artifact_sha256,
                "wall_clock_seconds": result["wall_clock_seconds"],
                "pages_per_second": result["pages_per_second"],
                "spatial_precision": metrics["spatial_precision"],
                "spatial_recall": metrics["spatial_recall"],
                "spatial_f1": metrics["spatial_f1"],
                "mean_iou": metrics["mean_iou"],
                "pairwise_order_accuracy": metrics["pairwise_order_accuracy"],
                "classification_accuracy_on_matched": metrics["classification_accuracy_on_matched"],
                "run_json_path": run_evidence["path"],
                "run_json_sha256": run_evidence["sha256"],
                "raw_artifacts": _raw_artifact_catalog(run_data, run["raw_directory"], root),
                "physical_ir_path": physical_evidence["path"],
                "physical_ir_sha256": physical_evidence["sha256"],
                "physical_ir_byte_size": physical_evidence["byte_size"],
                "reference_annotation_path": annotation_evidence["path"],
                "reference_annotation_sha256": annotation_evidence["sha256"],
                "evaluation_result_path": result_evidence["path"],
                "evaluation_result_sha256": result_evidence["sha256"],
                "table_object_mapping": _table_object_mapping(run, physical_document),
            }
            if document_item["document_id"] == "tt-04-2023-bkhdt-so-do-ban-do":
                diagram_page_index = audit_data["map_figure_page_evidence"]["layout_diagram"][
                    "page_index"
                ]
                run_records[parser]["layout_diagram_object_mapping"] = _layout_diagram_mapping(
                    run, physical_document, diagram_page_index
                )
            if document_item["document_id"].startswith("qd-23-2008"):
                blocks = [block for page in physical_document.pages for block in page.blocks]
                scanned_case[parser] = {
                    "raw_object_type_counts": dict(sorted(_raw_type_counts(run).items())),
                    "total_structural_physical_blocks": len(blocks),
                    "physical_ir_kind_counts": dict(
                        sorted(Counter(block.kind.value for block in blocks).items())
                    ),
                    "non_empty_text_blocks": sum(bool(block.text.strip()) for block in blocks),
                    "text_or_title_blocks": sum(
                        block.kind in {BlockKind.TEXT, BlockKind.TITLE} for block in blocks
                    ),
                    "non_empty_text_or_title_blocks": sum(
                        block.kind in {BlockKind.TEXT, BlockKind.TITLE} and bool(block.text.strip())
                        for block in blocks
                    ),
                }
        if page_count is None:
            raise ValueError("each corpus document requires at least one retained parser run")
        documents.append(
            {
                "document_id": manifest.document.id,
                "version_id": manifest.version.id,
                "document_number": manifest.document.document_number,
                "title": manifest.document.title,
                "issuer": manifest.document.issuer,
                "issued_on": str(manifest.version.issued_on),
                "effective_on": str(manifest.version.effective_on)
                if manifest.version.effective_on is not None
                else None,
                "signer": manifest.source.signer,
                "pages": page_count,
                "source_artifact_byte_size": manifest.artifact.byte_size,
                "source_artifact_path": _relative(document_item["source_path"], root),
                "source_artifact_sha256": manifest.artifact.sha256,
                "manifest_path": _relative(document_item["manifest_path"], root),
                "reference_annotation_path": annotation_evidence["path"],
                "reference_annotation_sha256": annotation_evidence["sha256"],
                "audited_pages": len(annotation.audited_pages),
                "reference_regions": sum(len(page.regions) for page in annotation.audited_pages),
                "runs": run_records,
            }
        )

    provenance = {
        (
            item.annotation_schema_version,
            item.annotation_version,
            item.annotator,
            item.annotator_type.value,
            item.annotation_method.value,
            item.created_at.isoformat(),
            item.prior_parser_output_exposure,
            item.parser_output_used_as_reference,
        )
        for item in annotations
    }
    if len(provenance) != 1:
        raise ValueError("reference annotation files disagree on version or provenance")
    (
        annotation_schema,
        annotation_version,
        annotator,
        annotator_type,
        method,
        created_at,
        prior_exposure,
        used_as_reference,
    ) = provenance.pop()
    total_pages = sum(document["pages"] for document in documents)
    coverage: dict[str, Any] = {}
    for parser in ("marker", "mineru"):
        covered = [document for document in documents if parser in document["runs"]]
        first_run = covered[0]["runs"][parser]
        coverage[parser] = {
            "parser": parser,
            "version": first_run["parser_version"],
            "backend": first_run["backend"],
            "mode": first_run["mode"],
            "configuration": first_run["configuration"],
            "evaluated_documents": len(covered),
            "corpus_documents": len(documents),
            "evaluated_pages": sum(document["pages"] for document in covered),
            "corpus_pages": total_pages,
            "omitted_documents": [
                document["document_id"] for document in documents if parser not in document["runs"]
            ],
        }
    annotations_by_document = {annotation.document_id: annotation for annotation in annotations}
    table_example_annotation = annotations_by_document["tt-04-2026-bxd-pl2-dinh-muc"]
    table_example_page = next(
        page
        for page in table_example_annotation.audited_pages
        if "norm_table_header" in page.phenomena
    )
    scanned_annotation = annotations_by_document["qd-23-2008-ubnd-hanoi-vien-quy-hoach"]
    scanned_page_indices = [
        page.page_index
        for page in scanned_annotation.audited_pages
        if page.page_modality == "scanned_raster"
    ]
    determinism = _collect_normalization_determinism(root, config)
    _write_json(root / DETERMINISM_MANIFEST, determinism)
    manifest_data = {
        "benchmark_schema_version": 2,
        "annotation_schema_version": annotation_schema,
        "annotation_version": annotation_version,
        "annotation_provenance": {
            "annotator": annotator,
            "annotator_type": annotator_type,
            "annotation_method": method,
            "created_at": created_at,
            "prior_parser_output_exposure": prior_exposure,
            "parser_output_used_as_reference": used_as_reference,
            "audit_evidence_path": audit_evidence["path"],
            "audit_evidence_sha256": audit_evidence["sha256"],
        },
        "corpus_total_pages": total_pages,
        "corpus_documents_count": len(documents),
        "total_audited_pages": sum(document["audited_pages"] for document in documents),
        "total_reference_regions": sum(document["reference_regions"] for document in documents),
        "evaluation_configuration": {
            "coordinate_system": "normalized_1000",
            "iou_threshold": 0.5,
            "matching_primary_objective": (
                "maximum cardinality among edges at or above IoU threshold"
            ),
            "matching_secondary_objective": (
                "maximum exact sum of computed IEEE-754 IoU values among "
                "maximum-cardinality matchings"
            ),
            "deterministic_tie_breaking": "stable zero-based graph and edge ordering",
        },
        "metric_definitions": {
            "spatial_precision": {
                "formula": "TP_spatial / (TP_spatial + FP_spatial)",
                "zero_denominator": 0.0,
            },
            "spatial_recall": {
                "formula": "TP_spatial / (TP_spatial + FN_spatial)",
                "zero_denominator": 0.0,
            },
            "spatial_f1": {
                "formula": (
                    "2 * spatial_precision * spatial_recall / (spatial_precision + spatial_recall)"
                ),
                "zero_denominator": 0.0,
            },
            "mean_iou": {"formula": "sum(IoU of spatial matches) / spatial_matches"},
            "classification_accuracy_on_matched": {
                "formula": "correct_classified_spatial_matches / spatial_matches"
            },
            "pairwise_order_accuracy": {
                "formula": "concordant comparable pairs / total comparable pairs",
                "prediction_ties": "non-concordant",
                "null_when": "fewer than two matched regions",
            },
        },
        "parser_coverage_matrix": coverage,
        "scanned_case_evidence": {
            "document_id": "qd-23-2008-ubnd-hanoi-vien-quy-hoach",
            "page_modality": "scanned_raster",
            "text_reference_available": False,
            "ocr_recall_cer_wer_available": False,
            "runs": scanned_case,
        },
        "map_figure_page_evidence": audit_data["map_figure_page_evidence"],
        "gap_example_pages": {
            "table_localization": {
                "document_id": table_example_annotation.document_id,
                "pdf_page_number_1_based": table_example_page.page_index + 1,
                "page_index": table_example_page.page_index,
            },
            "scanned_modality": {
                "document_id": scanned_annotation.document_id,
                "pdf_page_numbers_1_based": [index + 1 for index in scanned_page_indices],
                "page_indices": scanned_page_indices,
            },
        },
        "normalization_determinism": _file_evidence(root / DETERMINISM_MANIFEST, root),
        "clean_clone_reproducibility": {
            "can": [
                "validate committed benchmark, annotation, and determinism artifacts",
                "regenerate all Markdown reports offline",
                "run evaluation unit tests",
            ],
            "cannot_without_external_artifacts": [
                "regenerate parser outputs",
                "recompute Physical IR normalization from retained raw parser evidence",
                "recompute benchmark evaluations from untracked Physical IR files",
            ],
        },
        "documents": documents,
    }
    _write_json(root / BENCHMARK_MANIFEST, manifest_data)
    return manifest_data


def _validated_machine_evidence(root: Path) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads((root / BENCHMARK_MANIFEST).read_text(encoding="utf-8"))
    provenance = manifest["annotation_provenance"]
    audit_path = root / provenance["audit_evidence_path"]
    if _file_evidence(audit_path, root)["sha256"] != provenance["audit_evidence_sha256"]:
        raise ValueError("reference annotation audit hash mismatch")
    determinism_ref = manifest["normalization_determinism"]
    determinism_path = root / determinism_ref["path"]
    if _file_evidence(determinism_path, root)["sha256"] != determinism_ref["sha256"]:
        raise ValueError("normalization determinism artifact hash mismatch")
    determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
    if not determinism["all_equal"] or not all(item["equal"] for item in determinism["entries"]):
        raise ValueError("normalization determinism evidence contains a failed pair")
    for document in manifest["documents"]:
        annotation = root / document["reference_annotation_path"]
        if _file_evidence(annotation, root)["sha256"] != document["reference_annotation_sha256"]:
            raise ValueError("reference annotation hash mismatch")
        annotation_data = DocumentAnnotation.model_validate_json(
            annotation.read_text(encoding="utf-8")
        )
        if (
            annotation_data.document_id != document["document_id"]
            or annotation_data.version_id != document["version_id"]
            or annotation_data.source_sha256 != document["source_artifact_sha256"]
        ):
            raise ValueError("reference annotation identity conflicts with benchmark manifest")
        for parser, run in document["runs"].items():
            if (
                run["reference_annotation_path"] != document["reference_annotation_path"]
                or run["reference_annotation_sha256"] != document["reference_annotation_sha256"]
                or run["source_artifact_sha256"] != document["source_artifact_sha256"]
            ):
                raise ValueError("run evidence conflicts with document evidence")
            result_path = root / run["evaluation_result_path"]
            if _file_evidence(result_path, root)["sha256"] != run["evaluation_result_sha256"]:
                raise ValueError("evaluation result hash mismatch")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            expected_result_values = {
                "document_id": document["document_id"],
                "parser": parser,
                "parser_version": run["parser_version"],
                "parser_backend": run["backend"],
                "pages_audited": document["audited_pages"],
                "total_document_pages": document["pages"],
                "wall_clock_seconds": run["wall_clock_seconds"],
                "pages_per_second": run["pages_per_second"],
            }
            for field, expected in expected_result_values.items():
                if result.get(field) != expected:
                    raise ValueError(f"evaluation result {field} conflicts with benchmark manifest")
            for metric in (
                "spatial_precision",
                "spatial_recall",
                "spatial_f1",
                "mean_iou",
                "pairwise_order_accuracy",
                "classification_accuracy_on_matched",
            ):
                if result["metrics"].get(metric) != run[metric]:
                    raise ValueError(
                        f"evaluation result metric {metric} conflicts with benchmark manifest"
                    )
    manifest["_determinism"] = determinism
    return manifest


def _format_order(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _render_parser_report(manifest: dict[str, Any]) -> str:
    coverage = manifest["parser_coverage_matrix"]
    lines = [
        "# Parser Benchmark: Vietnamese Legal and Planning Corpus v1",
        "",
        "## Scope and provenance",
        "",
        (
            f"This benchmark contains {manifest['corpus_documents_count']} authoritative "
            f"documents spanning {manifest['corpus_total_pages']} pages. It evaluates "
            f"{manifest['total_reference_regions']} reference regions on "
            f"{manifest['total_audited_pages']} selected pages."
        ),
        "",
        (
            f"The annotations are {manifest['annotation_version']} AI visual reference "
            "annotations created with "
            f"`{manifest['annotation_provenance']['annotation_method']}`. Prior parser output "
            "exposure was "
            f"`{str(manifest['annotation_provenance']['prior_parser_output_exposure']).lower()}`; "
            "parser outputs were not used as the reference source for region geometry or type."
        ),
        "",
        "## Parser coverage",
        "",
    ]
    for parser, item in coverage.items():
        lines.append(
            f"- **{parser} {item['version']}**: "
            f"{item['evaluated_documents']}/{item['corpus_documents']} documents and "
            f"{item['evaluated_pages']}/{item['corpus_pages']} pages "
            f"(`{item['backend']}`)."
        )
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Document ID | Parser | Pages | Runtime (s) | Pages/s | Audited pages | "
            "Spatial P | Spatial R | Spatial F1 | Mean IoU | Pairwise order |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for document in manifest["documents"]:
        for parser, run in document["runs"].items():
            lines.append(
                f"| `{document['document_id']}` | {parser} | {document['pages']} | "
                f"{run['wall_clock_seconds']:.4f} | {run['pages_per_second']:.4f} | "
                f"{document['audited_pages']} | {run['spatial_precision']:.4f} | "
                f"{run['spatial_recall']:.4f} | {run['spatial_f1']:.4f} | "
                f"{run['mean_iou']:.4f} | {_format_order(run['pairwise_order_accuracy'])} |"
            )
    lines.extend(["", "## Metric definitions", ""])
    for name, definition in manifest["metric_definitions"].items():
        line = f"- `{name}`: `{definition['formula']}`."
        if "null_when" in definition:
            line += (
                f" Null when {definition['null_when']}; prediction ties are "
                f"{definition['prediction_ties']}."
            )
        lines.append(line)
    scanned = manifest["scanned_case_evidence"]
    lines.extend(["", "## Scanned-document evidence", ""])
    for parser, facts in scanned["runs"].items():
        lines.append(
            f"- **{parser}** produced {facts['total_structural_physical_blocks']} structural "
            f"Physical IR blocks, {facts['non_empty_text_blocks']} with non-empty text, and "
            f"{facts['text_or_title_blocks']} typed TEXT/TITLE."
        )
    lines.append(
        "No OCR recall, CER, or WER is reported because no text transcription reference exists."
    )
    map_evidence = manifest["map_figure_page_evidence"]
    diagram = map_evidence["layout_diagram"]
    prose = map_evidence["pre_symbology_prose"]
    sheets = map_evidence["symbology_sheets"]
    sheet_pairs = ", ".join(
        f"PDF page {pdf_page} (page_index={page_index})"
        for pdf_page, page_index in zip(
            sheets["pdf_page_numbers_1_based"], sheets["page_indices"], strict=True
        )
    )
    lines.extend(
        [
            "",
            "## Visual page evidence",
            "",
            (
                f"The layout diagram is on PDF page {diagram['pdf_page_number_1_based']} "
                f"(page_index={diagram['page_index']}). PDF page "
                f"{prose['pdf_page_number_1_based']} (page_index={prose['page_index']}) is "
                f"prose. The verified symbology sheets are: {sheet_pairs}."
            ),
            "",
            "## Normalization determinism",
            "",
            (
                f"All {manifest['_determinism']['available_pairs']} retained parser/document "
                "pairs produced byte-identical A/B Physical IR JSON. This statement is "
                f"generated from `{manifest['normalization_determinism']['path']}`."
            ),
            "",
            "## Reproducibility boundary",
            "",
            (
                "A clean clone can validate committed machine artifacts, regenerate these "
                "Markdown reports offline, and run unit tests. Raw parser outputs, source PDFs, "
                "run manifests, and Physical IR files remain intentionally untracked; restoring "
                "the paths and hashes listed in the benchmark manifest is required to recollect "
                "Stage A evidence."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_corpus_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Vietnamese Legal and Planning Reference Corpus v1",
        "",
        (
            f"The corpus registers {manifest['corpus_documents_count']} official documents and "
            f"{manifest['corpus_total_pages']} source pages. Source PDFs are identified by "
            "immutable SHA-256 values."
        ),
        "",
        "| Document | Official number | Pages | Issued | Effective | Signer | Source SHA-256 |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for document in manifest["documents"]:
        effective = (
            document["effective_on"] if document["effective_on"] is not None else "explicit null"
        )
        lines.append(
            f"| `{document['document_id']}` | `{document['document_number']}` | "
            f"{document['pages']} | {document['issued_on']} | {effective} | "
            f"{document['signer']} | `{document['source_artifact_sha256']}` |"
        )
    lines.extend(
        [
            "",
            "## Reference annotation policy",
            "",
            (
                f"Version {manifest['annotation_version']} uses "
                f"`{manifest['annotation_provenance']['annotation_method']}` AI visual reference "
                "annotations. For digital pages, regions target visually separable text groups, "
                "headings, coherent outer tables, and visible figures at the selected audit "
                "granularity. For scanned pages, modality is separate and full-page raster boxes "
                "are not treated as OCR text truth."
            ),
            "",
            (
                "Per-page re-audit evidence is committed at "
                f"`{manifest['annotation_provenance']['audit_evidence_path']}` with SHA-256 "
                f"`{manifest['annotation_provenance']['audit_evidence_sha256']}`."
            ),
            "",
            (
                "These are reference annotations, not human ground truth. Prior parser-output "
                "exposure existed, but parser output was not the reference source for geometry "
                "or region type during the visual re-audit."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _render_gap_report(manifest: dict[str, Any]) -> str:
    table_totals: dict[str, Counter[str]] = {}
    raw_totals: Counter[str] = Counter()
    diagram_mappings: dict[str, dict[str, Any]] = {}
    for document in manifest["documents"]:
        for parser, run in document["runs"].items():
            mapping = run["table_object_mapping"]
            raw_totals[parser] += mapping["raw_table_object_count"]
            table_totals.setdefault(parser, Counter()).update(mapping["physical_ir_kind_counts"])
            if "layout_diagram_object_mapping" in run:
                diagram_mappings[parser] = run["layout_diagram_object_mapping"]
    scanned = manifest["scanned_case_evidence"]["runs"]
    diagram = manifest["map_figure_page_evidence"]["layout_diagram"]
    diagram_candidate_count = sum(
        sum(item["raw_object_type_counts"].values()) for item in diagram_mappings.values()
    )
    table_page = manifest["gap_example_pages"]["table_localization"]
    scanned_pages = manifest["gap_example_pages"]["scanned_modality"]
    scanned_page_pairs = ", ".join(
        f"PDF page {pdf_page} (page_index={page_index})"
        for pdf_page, page_index in zip(
            scanned_pages["pdf_page_numbers_1_based"],
            scanned_pages["page_indices"],
            strict=True,
        )
    )
    lines = [
        "# Physical Document IR v0: Evidence-Based Gaps",
        "",
        "This report records observed representational gaps only; it does not implement "
        "Physical IR v1.",
        "",
        "## TABLE representation",
        "",
        (
            f"- **Actual page evidence**: `{table_page['document_id']}`, PDF page "
            f"{table_page['pdf_page_number_1_based']} "
            f"(page_index={table_page['page_index']}) contains a visually audited norm table."
        ),
        "",
    ]
    for parser in sorted(raw_totals):
        mappings = (
            ", ".join(
                f"{count} as `{kind}`" for kind, count in sorted(table_totals[parser].items())
            )
            or "no corresponding normalized blocks"
        )
        lines.append(
            f"- **Raw `{parser}` representation and v0 mapping**: "
            f"{raw_totals[parser]} parser-native table objects map to {mappings}."
        )
    lines.extend(
        [
            "",
            (
                "- **Specific information loss**: The counts follow parser-native table "
                "objects through `source_raw_index`. "
                "Separately emitted nested text may become TEXT, but that does not preserve the "
                "table object, cells, spans, or header structure."
            ),
            "",
            (
                f"- **Machine-derived frequency**: {sum(raw_totals.values())} table objects "
                "were traced across the retained runs."
            ),
            "",
            "- **Candidate future requirement**: Represent TABLE objects and structured cells.",
            "",
            "## FIGURE / MAP representation",
            "",
            (
                "- **Actual page evidence**: TT04/2023 contains a verified layout diagram on "
                f"PDF page "
                f"{diagram['pdf_page_number_1_based']} (page_index={diagram['page_index']}) and "
                "later visually verified symbology sheets."
            ),
            "",
        ]
    )
    for parser in sorted(diagram_mappings):
        mapping = diagram_mappings[parser]
        raw = ", ".join(
            f"{count} `{kind}`" for kind, count in mapping["raw_object_type_counts"].items()
        )
        normalized = ", ".join(
            f"{count} `{kind}`" for kind, count in mapping["physical_ir_kind_counts"].items()
        )
        lines.append(
            f"- **Raw `{parser}` representation and v0 mapping**: {raw} map to {normalized}."
        )
    lines.extend(
        [
            "",
            (
                "- **Specific information loss**: Physical IR v0 has no FIGURE/MAP kind or "
                "retained visual-asset link, so graphical identity and asset provenance are lost."
            ),
            "",
            (
                "- **Machine-derived frequency**: the page-specific provenance trace covers "
                f"{diagram_candidate_count} "
                "parser-native diagram candidates across the retained parser runs; it is not a "
                "corpus-wide figure-frequency estimate."
            ),
            "",
            (
                "- **Candidate future requirement**: Add typed visual objects and relative "
                "asset references."
            ),
            "",
            "## OCR / modality distinction",
            "",
            (
                f"- **Actual page evidence**: `{scanned_pages['document_id']}` is raster-only "
                f"on {scanned_page_pairs}."
            ),
            "",
        ]
    )
    for parser, facts in scanned.items():
        raw_types = ", ".join(
            f"{count} `{kind}`" for kind, count in facts["raw_object_type_counts"].items()
        )
        physical_kinds = ", ".join(
            f"{count} `{kind}`" for kind, count in facts["physical_ir_kind_counts"].items()
        )
        lines.append(
            f"- **Raw `{parser}` representation and v0 mapping**: {raw_types}; normalized as "
            f"{physical_kinds}."
        )
    lines.extend(
        [
            "",
            (
                "- **Specific information loss**: Physical IR v0 does not distinguish native "
                "digital text from OCR-derived text "
                "or attach OCR confidence. No OCR recall, CER, or WER conclusion is possible "
                "without transcription reference data."
            ),
            "",
            (
                "- **Machine-derived frequency**: "
                + "; ".join(
                    f"{parser} produced {facts['total_structural_physical_blocks']} structural "
                    f"blocks, {facts['non_empty_text_blocks']} with non-empty text"
                    for parser, facts in scanned.items()
                )
                + "."
            ),
            "",
            (
                "- **Candidate future requirement**: Carry extraction modality, OCR confidence, "
                "and provenance without treating OCR output as transcription reference data."
            ),
            "",
            "## Clean-clone boundary",
            "",
            (
                "The quantitative statements above are rendered from the committed benchmark "
                "manifest. Recomputing them from parser-native evidence requires restoration of "
                "the external paths and hashes listed there."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_reports_from_committed_artifacts(root: Path, output_root: Path | None = None) -> None:
    """Stage B: validate committed machine evidence and render reports without raw artifacts."""
    manifest = _validated_machine_evidence(root)
    destination = root if output_root is None else output_root
    rendered = (
        _render_parser_report(manifest),
        _render_corpus_report(manifest),
        _render_gap_report(manifest),
    )
    for relative_path, report_text in zip(REPORT_PATHS, rendered, strict=True):
        path = destination / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((report_text.rstrip("\n") + "\n").encode("utf-8"))


def generate_all_benchmarks_and_reports(root: Path) -> dict[str, Any]:
    """Collect Stage A machine evidence, then render Stage B Markdown reports."""
    manifest = collect_benchmark_evidence(root)
    render_reports_from_committed_artifacts(root)
    return manifest


__all__ = [
    "collect_benchmark_evidence",
    "generate_all_benchmarks_and_reports",
    "get_corpus_config",
    "render_reports_from_committed_artifacts",
]
