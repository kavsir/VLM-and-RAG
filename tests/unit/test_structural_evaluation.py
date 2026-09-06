"""Committed Structural IR reference, benchmark, and offline reproduction tests."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.evaluation.structural import (
    StructuralReferenceAnnotation,
    load_structural_annotations,
    render_structural_ir_v1_validation,
    verify_committed_structural_evidence,
)
from vlm_rag.structural_ir import structural_document_from_json

ROOT = Path(__file__).parents[2]
EVIDENCE_PATH = ROOT / "data/benchmarks/structural_ir_v1_validation.v1.json"


def _evidence() -> dict[str, object]:
    value: object = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_reference_annotations_have_required_scope_and_independent_identity() -> None:
    annotations = load_structural_annotations(ROOT)
    assert len(annotations) == 6
    assert sum(len(annotation.audited_pages) for annotation in annotations.values()) == 46
    assert (
        sum(
            node.scored
            for annotation in annotations.values()
            for page in annotation.audited_pages
            for node in page.nodes
        )
        == 188
    )
    for annotation in annotations.values():
        assert annotation.prior_physical_ir_exposure is True
        assert annotation.physical_ir_used_as_reference is False


def test_reference_annotations_contain_no_parser_or_block_identity() -> None:
    for path in (ROOT / "data/structural_annotations").glob("*.json"):
        payload = path.read_text(encoding="utf-8")
        assert '"block_id"' not in payload
        assert '"parser"' not in payload


def test_reference_schema_forbids_unreviewed_fields() -> None:
    annotation = next(iter(load_structural_annotations(ROOT).values()))
    raw = annotation.model_dump(mode="json")
    raw["parser"] = "marker"
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_committed_outputs_are_strict_and_bound_to_issue_007_hashes() -> None:
    evidence = _evidence()
    physical = json.loads(
        (ROOT / "data/benchmarks/physical_ir_v1_validation.v1.json").read_text(encoding="utf-8")
    )
    expected = {
        (entry["document_id"], entry["parser"]): entry["sha256_a"] for entry in physical["entries"]
    }
    entries = evidence["entries"]
    assert isinstance(entries, list) and len(entries) == 10
    for entry in entries:
        assert isinstance(entry, dict)
        path = ROOT / entry["output_path"]
        document = structural_document_from_json(path.read_text(encoding="utf-8"))
        assert (
            document.source_physical_ir_sha256 == expected[(document.document_id, entry["parser"])]
        )
        assert entry["equal"] is True


def test_structural_evidence_and_report_reproduce_offline() -> None:
    evidence = _evidence()
    verify_committed_structural_evidence(ROOT, evidence)
    expected_report = (ROOT / "docs/research/structural-ir-v1-validation.md").read_text(
        encoding="utf-8"
    )
    assert render_structural_ir_v1_validation(evidence) == expected_report


def test_machine_evidence_records_exact_partition_and_metric_formulas() -> None:
    evidence = _evidence()
    assert evidence["available_pairs"] == 10
    assert evidence["all_deterministic"] is True
    aggregate = evidence["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["exact_text_partition_pairs"] == 10
    assert aggregate["single_object_ownership_pairs"] == 10
    formulas = evidence["metric_formulas"]
    assert isinstance(formulas, dict)
    assert formulas["node_precision"] == "TP / (TP + FP)"
