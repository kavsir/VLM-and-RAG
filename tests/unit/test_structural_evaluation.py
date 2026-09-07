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
    OrdinalSystem,
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
        method = (
            RecognitionMethod.GENERIC_DECIMAL_HEADING
            if kind == StructuralNodeKind.GENERIC_SECTION
            else RecognitionMethod.EXPLICIT_LEGAL_MARKER
        )
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
                    recognition_method=method,
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
    instance_by_path: dict[str, str] = {}
    for kind, key, parent_path in specs:
        path = (
            f"{parent_path}/{_segment(kind, key)}"
            if parent_path != "document"
            else _segment(kind, key)
        )
        ordinal_system = {
            StructuralNodeKind.ARTICLE: OrdinalSystem.ARTICLE,
            StructuralNodeKind.CLAUSE: OrdinalSystem.CLAUSE,
            StructuralNodeKind.POINT: OrdinalSystem.POINT,
            StructuralNodeKind.GENERIC_SECTION: OrdinalSystem.GENERIC_DECIMAL,
        }.get(kind, OrdinalSystem.FORMAL_CONTAINER)
        instance_id = f"ref-test-{len(nodes):04d}"
        nodes.append(
            {
                "reference_record_id": f"record-test-{len(nodes):04d}",
                "reference_instance_id": instance_id,
                "parent_reference_instance_id": instance_by_path.get(parent_path),
                "page_index": 0,
                "pdf_page_number_1_based": 1,
                "kind": kind.value,
                "ordinal_raw": key,
                "ordinal_key": key,
                "ordinal_system": ordinal_system.value,
                "marker_text": f"{key}.",
                "title": None,
                "canonical_path": path,
                "parent_canonical_path": parent_path,
                "source_excerpt": f"{key}.",
                "visual_evidence_note": "Test reference.",
                "scored": path not in context_paths,
            }
        )
        instance_by_path[path] = instance_id
    return StructuralReferenceAnnotation.model_validate(
        {
            "annotation_schema_version": 4,
            "annotation_version": "v4",
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


def _duplicate_canonical_parent_payload(
    *, second_parent_page: int, include_context_copy: bool = False
) -> tuple[dict[str, Any], str, str]:
    raw = _annotation(
        [
            (StructuralNodeKind.ARTICLE, "2", "document"),
            (StructuralNodeKind.CLAUSE, "1", "article:2"),
        ]
    ).model_dump(mode="json")
    parent_a, child = raw["audited_pages"][0]["nodes"]
    parent_a_id = parent_a["reference_instance_id"]
    parent_b = deepcopy(parent_a)
    parent_b["reference_record_id"] = "record-test-parent-b"
    parent_b["reference_instance_id"] = "ref-test-parent-b"
    parent_b["page_index"] = second_parent_page
    parent_b["pdf_page_number_1_based"] = second_parent_page + 1
    parent_b_id = parent_b["reference_instance_id"]
    child["page_index"] = 1
    child["pdf_page_number_1_based"] = 2

    page_zero_nodes = [parent_a]
    later_pages: list[dict[str, Any]] = []
    if second_parent_page == 0:
        page_zero_nodes.append(parent_b)
    else:
        later_pages.append(
            {
                "page_index": second_parent_page,
                "pdf_page_number_1_based": second_parent_page + 1,
                "selection_rationale": "Second duplicate parent marker.",
                "nodes": [parent_b],
            }
        )
    child_page_nodes = [child]
    if include_context_copy:
        context = deepcopy(parent_a)
        context["reference_record_id"] = "record-test-parent-a-context"
        context["scored"] = False
        child_page_nodes.insert(0, context)
    raw["audited_pages"] = [
        {
            "page_index": 0,
            "pdf_page_number_1_based": 1,
            "selection_rationale": "First duplicate parent marker.",
            "nodes": page_zero_nodes,
        },
        {
            "page_index": 1,
            "pdf_page_number_1_based": 2,
            "selection_rationale": "Child marker.",
            "nodes": child_page_nodes,
        },
        *later_pages,
    ]
    return raw, parent_a_id, parent_b_id


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
        == 187
    )
    for annotation in annotations.values():
        assert annotation.prior_physical_ir_exposure is True
        assert annotation.prior_structural_extractor_exposure is True
        assert annotation.physical_ir_used_as_reference is False
        assert annotation.structural_extractor_output_used_as_reference is False
        assert annotation.annotation_schema_version == 4
        assert annotation.annotation_version == "v4"


def test_all_reference_parent_instances_begin_on_or_before_children() -> None:
    checked_assignments = 0
    for annotation in load_structural_annotations(ROOT).values():
        instances = {
            node.reference_instance_id: node
            for page in annotation.audited_pages
            for node in page.nodes
        }
        for child in instances.values():
            if child.parent_reference_instance_id is None:
                continue
            parent = instances[child.parent_reference_instance_id]
            assert parent.page_index <= child.page_index
            checked_assignments += 1
    assert checked_assignments > 0
    assert (
        _evidence()["reference_scope"]["unique_parent_instance_assignments"] == checked_assignments
    )


def test_reference_reaudit_log_covers_all_selected_pages() -> None:
    audit = json.loads(
        (ROOT / "data/structural_annotations/reference_structural_audit.v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["page_count"] == 46 == len(audit["records"])
    assert sum(item["scored_node_count"] for item in audit["records"]) == 187
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
        "scored_node_count": 187,
        "context_node_count": 40,
        "changed_or_removed_v1_node_records": 47,
        "corrected_point_ordinal_records": 30,
        "pages_with_node_field_changes": 13,
        "restored_genuine_node_records": 1,
        "v2_to_v3_changed_node_records": 1,
        "pages_reverified_for_v3": 46,
        "v3_to_v4_visual_changes": 0,
        "pages_reverified_for_v4": 46,
        "annotation_record_count": 227,
        "unique_structural_instance_ids": 211,
        "parent_instance_assignments": 192,
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


def test_reference_schema_accepts_duplicate_scored_path_with_unique_instance_identity() -> None:
    raw = _first_reference_payload()
    scored = next(node for page in raw["audited_pages"] for node in page["nodes"] if node["scored"])
    raw["audited_pages"][0]["nodes"].append(deepcopy(scored))
    raw["audited_pages"][0]["nodes"][-1]["page_index"] = 0
    raw["audited_pages"][0]["nodes"][-1]["pdf_page_number_1_based"] = 1
    raw["audited_pages"][0]["nodes"][-1]["reference_record_id"] += "-duplicate"
    raw["audited_pages"][0]["nodes"][-1]["reference_instance_id"] += "-duplicate"
    StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_duplicate_reference_instance_identity() -> None:
    raw = _annotation([(StructuralNodeKind.ARTICLE, "1", "document")]).model_dump(mode="json")
    scored = raw["audited_pages"][0]["nodes"][0]
    raw["audited_pages"][0]["nodes"].append(deepcopy(scored))
    raw["audited_pages"][0]["nodes"][-1]["reference_record_id"] += "-duplicate"
    with pytest.raises(ValidationError, match="duplicate scored reference_instance_id"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_duplicate_record_identity() -> None:
    raw = _first_reference_payload()
    original = next(node for page in raw["audited_pages"] for node in page["nodes"])
    node = deepcopy(original)
    node["reference_instance_id"] += "-second-instance"
    next(page for page in raw["audited_pages"] if page["nodes"])["nodes"].append(node)
    with pytest.raises(ValidationError, match="duplicate reference_record_id"):
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


def test_reference_unnumbered_node_requires_literal_unnumbered_segment() -> None:
    raw = _annotation([(StructuralNodeKind.APPENDIX, "1", "document")]).model_dump(mode="json")
    node = raw["audited_pages"][0]["nodes"][0]
    node["ordinal_raw"] = None
    node["ordinal_key"] = None
    node["canonical_path"] = "appendix:unnumbered"
    StructuralReferenceAnnotation.model_validate(raw)
    node["canonical_path"] = "appendix:999"
    with pytest.raises(ValidationError, match="ordinal_key disagrees with canonical path"):
        StructuralReferenceAnnotation.model_validate(raw)


@pytest.mark.parametrize(
    ("kind", "source_key", "raw_ordinal", "bad_key", "ordinal_system"),
    [
        ("point", "a", "c", "100", "point"),
        ("chapter", "1", "IV", "8", "formal_container"),
        ("article", "1", "10a", "10", "article"),
        ("generic_section", "1", "IX", "11", "generic_roman"),
    ],
)
def test_reference_schema_rejects_kind_aware_ordinal_corruption(
    kind: str,
    source_key: str,
    raw_ordinal: str,
    bad_key: str,
    ordinal_system: str,
) -> None:
    kind_enum = StructuralNodeKind(kind)
    if kind_enum == StructuralNodeKind.POINT:
        specs = [
            (StructuralNodeKind.ARTICLE, "1", "document"),
            (StructuralNodeKind.CLAUSE, "1", "article:1"),
            (StructuralNodeKind.POINT, source_key, "article:1/clause:1"),
        ]
    else:
        specs = [(kind_enum, source_key, "document")]
    raw = _annotation(specs).model_dump(mode="json")
    node = raw["audited_pages"][0]["nodes"][-1]
    node["ordinal_raw"] = raw_ordinal
    node["ordinal_key"] = bad_key
    node["ordinal_system"] = ordinal_system
    node["canonical_path"] = (
        _segment(kind_enum, bad_key)
        if node["parent_canonical_path"] == "document"
        else f"{node['parent_canonical_path']}/{_segment(kind_enum, bad_key)}"
    )
    with pytest.raises(ValidationError, match="kind-aware ordinal semantics"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_duplicate_reference_instances_are_counted_one_to_one() -> None:
    article = (StructuralNodeKind.ARTICLE, "2", "document")
    result = evaluate_structural_document(_document([article]), _annotation([article, article]))
    assert result["node_metrics"] == {
        "tp": 1,
        "fp": 0,
        "fn": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": pytest.approx(2 / 3),
    }
    assert len(result["false_negative_trace"]) == 1
    assert result["false_negative_trace"][0]["reference_instance_id"].startswith("ref-test-")


def test_ambiguous_duplicate_title_pairing_is_id_order_independent() -> None:
    article = (StructuralNodeKind.ARTICLE, "2", "document")
    prediction_raw = _document([article]).model_dump(mode="json")
    prediction_raw["nodes"][1]["title"] = "First visible title"
    prediction = StructuralDocument.model_validate(prediction_raw)
    annotation_raw = _annotation([article, article]).model_dump(mode="json")
    references = annotation_raw["audited_pages"][0]["nodes"]
    references[0]["title"] = "First visible title"
    references[1]["title"] = "Second visible title"
    annotation_a = StructuralReferenceAnnotation.model_validate(annotation_raw)
    result_a = evaluate_structural_document(prediction, annotation_a)
    references[0]["reference_instance_id"], references[1]["reference_instance_id"] = (
        references[1]["reference_instance_id"],
        references[0]["reference_instance_id"],
    )
    annotation_b = StructuralReferenceAnnotation.model_validate(annotation_raw)
    result_b = evaluate_structural_document(prediction, annotation_b)
    for metric_field in (
        "node_metrics",
        "node_metrics_by_kind",
        "parent_edge_metrics",
        "canonical_path_exact",
    ):
        assert result_a[metric_field] == result_b[metric_field]
    assert (
        result_a["title_normalized_exact"]
        == result_b["title_normalized_exact"]
        == {
            "eligible": 0,
            "exact": 0,
            "rate": None,
            "ambiguous_title_pairing_count": 1,
        }
    )
    assert result_a["matching_diagnostics"][0]["reason"] == (
        "ambiguous_duplicate_identity_title_excluded"
    )


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
    point = (StructuralNodeKind.POINT, "a", "chapter:1/section:1/article:1/clause:1")
    raw = _annotation([chapter, section, article, clause, point]).model_dump(mode="json")
    child = next(
        node for node in raw["audited_pages"][0]["nodes"] if node["kind"] == child_kind.value
    )
    parent_path = (
        "chapter:1" if invalid_parent_kind == StructuralNodeKind.CHAPTER else "chapter:1/section:1"
    )
    child["parent_canonical_path"] = parent_path
    child["parent_reference_instance_id"] = next(
        node["reference_instance_id"]
        for node in raw["audited_pages"][0]["nodes"]
        if node["canonical_path"] == parent_path
    )
    child["canonical_path"] = f"{parent_path}/{_segment(child_kind, child['ordinal_key'])}"
    with pytest.raises(ValidationError, match="invalid reference hierarchy"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_phantom_parent() -> None:
    raw = _annotation([(StructuralNodeKind.ARTICLE, "1", "document")]).model_dump(mode="json")
    node = raw["audited_pages"][0]["nodes"][0]
    node["parent_canonical_path"] = "chapter:999"
    node["canonical_path"] = f"chapter:999/{node['canonical_path'].rsplit('/', 1)[-1]}"
    node["parent_reference_instance_id"] = "ref-missing-parent"
    with pytest.raises(ValidationError, match="phantom parent_reference_instance_id"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_wrong_parent_instance() -> None:
    specs = [
        (StructuralNodeKind.ARTICLE, "1", "document"),
        (StructuralNodeKind.ARTICLE, "2", "document"),
        (StructuralNodeKind.CLAUSE, "1", "article:2"),
    ]
    raw = _annotation(specs).model_dump(mode="json")
    article_one, _, clause = raw["audited_pages"][0]["nodes"]
    clause["parent_reference_instance_id"] = article_one["reference_instance_id"]
    with pytest.raises(
        ValidationError, match="parent reference instance and canonical path disagree"
    ):
        StructuralReferenceAnnotation.model_validate(raw)


def test_duplicate_canonical_parent_requires_parent_not_to_begin_after_child() -> None:
    raw, earlier_parent_id, future_parent_id = _duplicate_canonical_parent_payload(
        second_parent_page=3
    )
    child = raw["audited_pages"][1]["nodes"][0]
    child["parent_reference_instance_id"] = earlier_parent_id
    StructuralReferenceAnnotation.model_validate(raw)
    child["parent_reference_instance_id"] = future_parent_id
    with pytest.raises(
        ValidationError, match="parent structural instance cannot begin after child"
    ):
        StructuralReferenceAnnotation.model_validate(raw)


def test_same_page_duplicate_canonical_parents_remain_explicitly_assignable() -> None:
    raw, _, same_page_parent_id = _duplicate_canonical_parent_payload(second_parent_page=0)
    raw["audited_pages"][1]["nodes"][0]["parent_reference_instance_id"] = same_page_parent_id
    StructuralReferenceAnnotation.model_validate(raw)


def test_context_copy_retains_structural_instance_marker_page() -> None:
    raw, earlier_parent_id, _ = _duplicate_canonical_parent_payload(
        second_parent_page=3, include_context_copy=True
    )
    context, child = raw["audited_pages"][1]["nodes"]
    assert context["page_index"] == 0
    assert context["reference_instance_id"] == earlier_parent_id
    assert child["page_index"] == 1
    StructuralReferenceAnnotation.model_validate(raw)
    context["page_index"] = 1
    context["pdf_page_number_1_based"] = 2
    with pytest.raises(ValidationError, match="contradictory repeated reference instance"):
        StructuralReferenceAnnotation.model_validate(raw)


def test_reference_schema_rejects_contradictory_context_definition() -> None:
    raw = _annotation([(StructuralNodeKind.CHAPTER, "1", "document")]).model_dump(mode="json")
    node = deepcopy(raw["audited_pages"][0]["nodes"][0])
    node["reference_record_id"] += "-context"
    node["title"] = "Conflicting title"
    node["scored"] = False
    raw["audited_pages"][0]["nodes"].append(node)
    with pytest.raises(ValidationError, match="contradictory repeated reference instance"):
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


def test_v4_preserves_qd23_repeated_article_as_false_negative() -> None:
    annotations = load_structural_annotations(ROOT)
    qd = annotations["qd-23-2008-ubnd-hanoi-vien-quy-hoach"]
    repeated = [
        node
        for page in qd.audited_pages
        for node in page.nodes
        if node.scored and node.kind == StructuralNodeKind.ARTICLE and node.ordinal_key == "2"
    ]
    assert [(node.page_index, node.canonical_path) for node in repeated] == [
        (0, "article:2"),
        (3, "article:2"),
    ]
    assert len({node.reference_instance_id for node in repeated}) == 2
    first_article_id = repeated[0].reference_instance_id
    clauses = [
        node
        for page in qd.audited_pages
        for node in page.nodes
        if node.kind == StructuralNodeKind.CLAUSE and node.canonical_path.startswith("article:2/")
    ]
    assert clauses
    assert {node.parent_reference_instance_id for node in clauses} == {first_article_id}
    instances = {
        node.reference_instance_id: node for page in qd.audited_pages for node in page.nodes
    }
    assert all(instances[first_article_id].page_index <= clause.page_index for clause in clauses)
    second_article_id = repeated[1].reference_instance_id
    raw = qd.model_dump(mode="json")
    earlier_clause = next(
        node
        for page in raw["audited_pages"]
        for node in page["nodes"]
        if node["kind"] == "clause" and node["parent_reference_instance_id"] == first_article_id
    )
    earlier_clause["parent_reference_instance_id"] = second_article_id
    with pytest.raises(
        ValidationError, match="parent structural instance cannot begin after child"
    ):
        StructuralReferenceAnnotation.model_validate(raw)
    evidence = _evidence()
    restored_false_negatives = [
        item
        for item in evidence["false_negative_trace"]
        if item["document_id"] == qd.document_id
        and item["page_index"] == 3
        and item["kind"] == "article"
        and item["ordinal_key"] == "2"
    ]
    assert {item["parser"] for item in restored_false_negatives} == {"marker", "mineru"}


def test_retained_tt04_stack_and_toc_corrections_are_measured() -> None:
    evidence = _evidence()
    duplicate_audit = evidence["duplicate_rejection_audit"]
    assert duplicate_audit["after_fix"]["total"] < duplicate_audit["before_fix"]["total"]
    assert "tt-04-2023-bkhdt-so-do-ban-do" not in duplicate_audit["after_fix"]["by_document"]
    assert "tt-04-2026-bxd-pl2-dinh-muc" not in duplicate_audit["after_fix"]["by_document"]
    assert duplicate_audit["after_fix"]["total"] == 8
    assert duplicate_audit["remaining_classification"] == {
        "QUOTED_NESTED_LEGISLATION_LIMITATION": 8,
        "GENUINE_STRUCTURE_RECOVERED": 0,
    }
    assert evidence["toc_corpus_audit"]["rejected_entries"] == 15
    for parser in ("marker", "mineru"):
        path = ROOT / "data/structural_ir/tt_04_2023_bkhdt_so_do_ban_do" / f"{parser}.v1.json"
        document = structural_document_from_json(path.read_text(encoding="utf-8"))
        appendix = next(
            node
            for node in document.nodes
            if node.kind == StructuralNodeKind.APPENDIX and node.ordinal_key == "1"
        )
        assert appendix.heading_anchors[0].page_index == 8
        assert not any(
            node.heading_anchors
            and node.heading_anchors[0].page_index >= 8
            and node.canonical_path.startswith("chapter:4/article:14")
            for node in document.nodes
        )


def test_historical_duplicate_audit_is_complete_and_current_remainder_is_classified() -> None:
    audit = json.loads(
        (ROOT / "data/benchmarks/structural_duplicate_audit.38d1042.v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["summary"] == {
        "total": 43,
        "by_classification": {
            "GENUINE_STRUCTURE_RECOVERED": 35,
            "QUOTED_NESTED_LEGISLATION_LIMITATION": 8,
        },
    }
    recovered = [
        item
        for item in audit["records"]
        if item["visual_classification"] == "GENUINE_STRUCTURE_RECOVERED"
    ]
    assert all(item["post_correction_outcome"]["status"] == "represented" for item in recovered)
    evidence = _evidence()
    assert evidence["duplicate_rejection_audit"]["after_fix"]["total"] == 8


def test_structural_evidence_and_report_reproduce_offline() -> None:
    evidence = _evidence()
    verify_committed_structural_evidence(ROOT, evidence)
    expected_report = (ROOT / "docs/research/structural-ir-v1-validation.md").read_text(
        encoding="utf-8"
    )
    assert render_structural_ir_v1_validation(evidence) == expected_report


def test_machine_evidence_records_exact_partition_and_metric_formulas() -> None:
    evidence = _evidence()
    assert evidence["validation_schema_version"] == 4
    assert evidence["reference_annotation_schema_version"] == 4
    assert evidence["reference_annotation_version"] == "v4"
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
    assert (
        evidence["canonical_path_correction"]["reviewed_head_suffix_paths_independent_recount"]
        == 67
    )
    assert evidence["duplicate_rejection_audit"]["before_fix"]["total"] == 237
    qd = next(
        item
        for item in evidence["cross_parser_agreement"]
        if item["document_id"] == "qd-23-2008-ubnd-hanoi-vien-quy-hoach"
    )
    assert qd["legal"]["jaccard"] is None
