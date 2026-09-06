"""Committed Structural IR reference, benchmark, and offline reproduction tests."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vlm_rag.evaluation.structural import (
    StructuralReferenceAnnotation,
    evaluate_structural_document,
    load_structural_annotations,
    render_structural_ir_v1_validation,
    structural_set_jaccard,
    verify_committed_structural_evidence,
)
from vlm_rag.structural_ir import (
    AnchorRole,
    PhysicalAnchor,
    RecognitionEvidence,
    RecognitionMethod,
    StructuralDocument,
    StructuralNode,
    StructuralNodeKind,
    structural_document_from_json,
)

ROOT = Path(__file__).parents[2]
EVIDENCE_PATH = ROOT / "data/benchmarks/structural_ir_v1_validation.v1.json"


def _evidence() -> dict[str, object]:
    value: object = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


SHA = "a" * 64


def _segment(kind: StructuralNodeKind, key: str) -> str:
    name = "generic" if kind == StructuralNodeKind.GENERIC_SECTION else kind.value
    return f"{name}:{key}"


def _document(
    specs: list[tuple[StructuralNodeKind, str, str]],
) -> StructuralDocument:
    nodes = [
        StructuralNode(
            id="root",
            kind=StructuralNodeKind.DOCUMENT,
            parent_id=None,
            depth=0,
            recognition_evidence=RecognitionEvidence(
                rule_id="root",
                recognition_method=RecognitionMethod.DOCUMENT_ROOT,
                matched_text="document",
            ),
            canonical_path="document",
        )
    ]
    ids = {"document": "root"}
    for index, (kind, key, parent_path) in enumerate(specs, start=1):
        path = (
            f"{parent_path}/{_segment(kind, key)}"
            if parent_path != "document"
            else _segment(kind, key)
        )
        node_id = f"n{index}"
        nodes.append(
            StructuralNode(
                id=node_id,
                kind=kind,
                parent_id=ids[parent_path],
                depth=path.count("/") + 1,
                ordinal_raw=key,
                ordinal_key=key,
                marker_text=f"{key}.",
                heading_anchors=(
                    PhysicalAnchor(
                        block_id=f"b{index}",
                        page_index=0,
                        role=AnchorRole.MARKER,
                        char_start=0,
                        char_end=1,
                    ),
                ),
                recognition_evidence=RecognitionEvidence(
                    rule_id="test",
                    recognition_method=RecognitionMethod.EXPLICIT_LEGAL_MARKER,
                    matched_text=f"{key}.",
                ),
                canonical_path=path,
            )
        )
        ids[path] = node_id
    return StructuralDocument(
        source_physical_ir_sha256="b" * 64,
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SHA,
        nodes=tuple(nodes),
    )


def _annotation(
    specs: list[tuple[StructuralNodeKind, str, str]],
    *,
    context_paths: set[str] | None = None,
) -> StructuralReferenceAnnotation:
    context_paths = context_paths or set()
    nodes: list[dict[str, Any]] = []
    for kind, key, parent_path in specs:
        path = (
            f"{parent_path}/{_segment(kind, key)}"
            if parent_path != "document"
            else _segment(kind, key)
        )
        nodes.append(
            {
                "page_index": 0,
                "pdf_page_number_1_based": 1,
                "kind": kind.value,
                "ordinal_raw": key,
                "ordinal_key": key,
                "marker_text": f"{key}.",
                "title": None,
                "canonical_path": path,
                "parent_canonical_path": parent_path,
                "source_excerpt": f"{key}.",
                "visual_evidence_note": "Test reference.",
                "scored": path not in context_paths,
            }
        )
    return StructuralReferenceAnnotation.model_validate(
        {
            "annotation_schema_version": 2,
            "annotation_version": "v2",
            "annotator": "codex",
            "annotator_type": "ai_visual_audit",
            "annotation_method": "visual_pdf_structural_reaudit",
            "created_at": "2026-09-06T00:00:00+07:00",
            "document_id": "test-document",
            "version_id": "v1",
            "source_sha256": SHA,
            "prior_physical_ir_exposure": True,
            "prior_structural_extractor_exposure": True,
            "physical_ir_used_as_reference": False,
            "structural_extractor_output_used_as_reference": False,
            "selection_rationale": "Test reference.",
            "audited_pages": [
                {
                    "page_index": 0,
                    "pdf_page_number_1_based": 1,
                    "selection_rationale": "Test page.",
                    "nodes": nodes,
                }
            ],
        }
    )


def test_reference_annotations_have_required_scope_and_disclosed_identity() -> None:
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
        == 186
    )
    for annotation in annotations.values():
        assert annotation.prior_physical_ir_exposure is True
        assert annotation.prior_structural_extractor_exposure is True
        assert annotation.physical_ir_used_as_reference is False
        assert annotation.structural_extractor_output_used_as_reference is False
        assert annotation.annotation_schema_version == 2
        assert annotation.annotation_version == "v2"


def test_reference_reaudit_log_covers_all_selected_pages() -> None:
    audit = json.loads(
        (ROOT / "data/structural_annotations/reference_structural_audit.v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["page_count"] == 46 == len(audit["records"])
    assert sum(item["scored_node_count"] for item in audit["records"]) == 186
    assert sum(item["context_node_count"] for item in audit["records"]) == 40
    assert len({(item["document_id"], item["page_index"]) for item in audit["records"]}) == 46
    assert all(len(item["render_sha256"]) == 64 for item in audit["records"])
    corrected = sum(
        change.startswith("Corrected POINT")
        for item in audit["records"]
        for change in item["changes_from_v1"]
    )
    assert corrected == 30
    assert audit["summary"] == {
        "scored_node_count": 186,
        "context_node_count": 40,
        "changed_or_removed_v1_node_records": 47,
        "corrected_point_ordinal_records": 30,
        "pages_with_node_field_changes": 13,
    }


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


def _first_reference_payload() -> dict[str, Any]:
    annotation = next(iter(load_structural_annotations(ROOT).values()))
    return annotation.model_dump(mode="json")


def test_reference_schema_rejects_duplicate_scored_identity() -> None:
    raw = _first_reference_payload()
    scored = next(node for page in raw["audited_pages"] for node in page["nodes"] if node["scored"])
    raw["audited_pages"][0]["nodes"].append(deepcopy(scored))
    raw["audited_pages"][0]["nodes"][-1]["page_index"] = 0
    raw["audited_pages"][0]["nodes"][-1]["pdf_page_number_1_based"] = 1
    with pytest.raises(ValidationError, match="duplicate scored canonical path"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_kind_path_disagreement() -> None:
    raw = _first_reference_payload()
    node = next(
        node for page in raw["audited_pages"] for node in page["nodes"] if node["kind"] == "article"
    )
    node["canonical_path"] = "clause:1"
    node["parent_canonical_path"] = "document"
    with pytest.raises(ValidationError, match="canonical path segment"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_ordinal_path_disagreement() -> None:
    raw = _first_reference_payload()
    node = next(
        node
        for page in raw["audited_pages"]
        for node in page["nodes"]
        if node["ordinal_key"] is not None
    )
    node["ordinal_key"] = "999"
    with pytest.raises(ValidationError, match="ordinal_key disagrees"):
        StructuralReferenceAnnotation.model_validate(raw)


@pytest.mark.parametrize(
    ("child_kind", "invalid_parent_kind"),
    [
        (StructuralNodeKind.POINT, StructuralNodeKind.CHAPTER),
        (StructuralNodeKind.CLAUSE, StructuralNodeKind.SECTION),
    ],
)
def test_reference_schema_rejects_invalid_parent_relationship(
    child_kind: StructuralNodeKind, invalid_parent_kind: StructuralNodeKind
) -> None:
    chapter = (StructuralNodeKind.CHAPTER, "1", "document")
    section = (StructuralNodeKind.SECTION, "1", "chapter:1")
    article = (StructuralNodeKind.ARTICLE, "1", "chapter:1/section:1")
    clause = (StructuralNodeKind.CLAUSE, "1", "chapter:1/section:1/article:1")
    point = (StructuralNodeKind.POINT, "1", "chapter:1/section:1/article:1/clause:1")
    raw = _annotation([chapter, section, article, clause, point]).model_dump(mode="json")
    child = next(
        node for node in raw["audited_pages"][0]["nodes"] if node["kind"] == child_kind.value
    )
    parent_path = (
        "chapter:1" if invalid_parent_kind == StructuralNodeKind.CHAPTER else "chapter:1/section:1"
    )
    child["parent_canonical_path"] = parent_path
    child["canonical_path"] = f"{parent_path}/{_segment(child_kind, '1')}"
    with pytest.raises(ValidationError, match="invalid reference hierarchy"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_phantom_parent() -> None:
    raw = _first_reference_payload()
    node = next(
        node for page in raw["audited_pages"] for node in page["nodes"] if node["kind"] == "article"
    )
    node["parent_canonical_path"] = "chapter:999"
    node["canonical_path"] = f"chapter:999/{node['canonical_path'].rsplit('/', 1)[-1]}"
    with pytest.raises(ValidationError, match="phantom parent_canonical_path"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_contradictory_context_definition() -> None:
    raw = _first_reference_payload()
    node = deepcopy(
        next(
            node
            for page in raw["audited_pages"]
            for node in page["nodes"]
            if node["ordinal_key"] == "1"
        )
    )
    node["ordinal_raw"] = "01"
    node["scored"] = False
    node["page_index"] = 0
    node["pdf_page_number_1_based"] = 1
    raw["audited_pages"][0]["nodes"].append(node)
    with pytest.raises(ValidationError, match="contradictory repeated reference context"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_hierarchy_aware_matching_does_not_occurrence_shift() -> None:
    specs = [
        (StructuralNodeKind.ARTICLE, "1", "document"),
        (StructuralNodeKind.CLAUSE, "1", "article:1"),
        (StructuralNodeKind.ARTICLE, "2", "document"),
        (StructuralNodeKind.CLAUSE, "1", "article:2"),
    ]
    prediction = _document([specs[0], specs[2], specs[3]])
    result = evaluate_structural_document(prediction, _annotation(specs))
    assert result["node_metrics_by_kind"]["clause"] == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
    }
    assert result["false_negative_trace"][0]["canonical_path"] == "article:1/clause:1"


def test_ambiguous_duplicate_group_remains_unmatched_one_to_one() -> None:
    reference_specs = [
        (StructuralNodeKind.ARTICLE, "1", "document"),
        (StructuralNodeKind.ARTICLE, "2", "document"),
        (StructuralNodeKind.CLAUSE, "1", "article:1"),
        (StructuralNodeKind.CLAUSE, "1", "article:2"),
    ]
    prediction_specs = [
        (StructuralNodeKind.ARTICLE, "3", "document"),
        (StructuralNodeKind.ARTICLE, "4", "document"),
        (StructuralNodeKind.CLAUSE, "1", "article:3"),
        (StructuralNodeKind.CLAUSE, "1", "article:4"),
    ]
    result = evaluate_structural_document(_document(prediction_specs), _annotation(reference_specs))
    assert result["node_metrics_by_kind"]["clause"]["tp"] == 0
    assert result["node_metrics_by_kind"]["clause"]["fp"] == 2
    assert result["node_metrics_by_kind"]["clause"]["fn"] == 2
    assert result["matching_diagnostics"] == [
        {
            "page_index": 0,
            "kind": "clause",
            "ordinal_key": "1",
            "reference_candidates": 2,
            "prediction_candidates": 2,
            "reason": "ambiguous_duplicate_group_unmatched",
        }
    ]


def test_full_parent_edge_metric_counts_root_and_unmatched_edges() -> None:
    article = (StructuralNodeKind.ARTICLE, "1", "document")
    clause = (StructuralNodeKind.CLAUSE, "1", "article:1")
    correct = evaluate_structural_document(
        _document([article, clause]), _annotation([article, clause])
    )
    assert correct["parent_edge_metrics"] == {
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    extra = evaluate_structural_document(_document([article, clause]), _annotation([article]))
    assert extra["parent_edge_metrics"]["fp"] == 1
    missing = evaluate_structural_document(_document([article]), _annotation([article, clause]))
    assert missing["parent_edge_metrics"]["fn"] == 1


def test_full_parent_edge_metric_counts_wrong_parent_as_fp_and_fn() -> None:
    article_one = (StructuralNodeKind.ARTICLE, "1", "document")
    article_two = (StructuralNodeKind.ARTICLE, "2", "document")
    clause_one = (StructuralNodeKind.CLAUSE, "1", "article:1")
    clause_two = (StructuralNodeKind.CLAUSE, "1", "article:2")
    reference = _annotation(
        [article_one, article_two, clause_one], context_paths={"article:1", "article:2"}
    )
    result = evaluate_structural_document(_document([article_two, clause_two]), reference)
    assert result["node_metrics"] == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert result["parent_edge_metrics"]["tp"] == 0
    assert result["parent_edge_metrics"]["fp"] == 1
    assert result["parent_edge_metrics"]["fn"] == 1


def test_structural_set_jaccard_null_and_fraction_semantics() -> None:
    item_a = {("article", "article:1")}
    item_b = {("article", "article:2")}
    assert structural_set_jaccard(set(), set()) is None
    assert structural_set_jaccard(item_a, set()) == 0.0
    assert structural_set_jaccard(item_a, item_a) == 1.0
    assert structural_set_jaccard(item_a | item_b, item_b) == 0.5


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
    assert evidence["validation_schema_version"] == 2
    assert evidence["reference_annotation_schema_version"] == 2
    assert evidence["reference_annotation_version"] == "v2"
    assert evidence["available_pairs"] == 10
    assert evidence["all_deterministic"] is True
    aggregate = evidence["aggregate"]
    assert isinstance(aggregate, dict)
    assert aggregate["exact_text_partition_pairs"] == 10
    assert aggregate["single_object_ownership_pairs"] == 10
    formulas = evidence["metric_formulas"]
    assert isinstance(formulas, dict)
    assert formulas["node_precision"] == "TP / (TP + FP)"
    assert "wrong parent contributes one FP and one FN" in formulas["parent_edge"]
    assert "null" in formulas["undefined_metric"]
    assert "null for an empty union" in formulas["cross_parser_jaccard"]
    assert evidence["benchmark"]["by_parser"]["marker"]["documents"] == 6
    assert evidence["benchmark"]["by_parser"]["mineru"]["documents"] == 4
    weighted = evidence["benchmark"]["parser_representation_weighted_aggregate"]
    assert len(evidence["false_positive_trace"]) == weighted["node_metrics"]["fp"]
    assert len(evidence["false_negative_trace"]) == weighted["node_metrics"]["fn"]
    assert weighted["node_metrics_by_kind"]["part"]["recall"] is None
    assert evidence["canonical_path_correction"]["regenerated_suffix_paths"] == 0
    qd = next(
        item
        for item in evidence["cross_parser_agreement"]
        if item["document_id"] == "qd-23-2008-ubnd-hanoi-vien-quy-hoach"
    )
    assert qd["legal"]["jaccard"] is None
