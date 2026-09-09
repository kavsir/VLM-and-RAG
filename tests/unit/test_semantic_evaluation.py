"""Offline committed Semantic IR v1 research-evidence tests."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.evaluation.semantic import (
    SemanticReferenceAnnotation,
    _evaluate_document,
    load_semantic_reference_annotations,
    normalized_edit_distance,
)
from vlm_rag.semantic_ir import SemanticMentionKind, semantic_document_from_json


def test_reference_is_disclosed_fixed_46_page_visual_audit() -> None:
    root = Path.cwd()
    annotations = load_semantic_reference_annotations(root)
    assert len(annotations) == 6
    assert sum(len(annotation.pages) for annotation in annotations.values()) == 46
    assert all(annotation.prior_semantic_extractor_exposure for annotation in annotations.values())
    assert all(
        not annotation.independent_or_blind_ground_truth for annotation in annotations.values()
    )
    assert all(annotation.annotation_schema_version == 2 for annotation in annotations.values())
    assert all(
        not annotation.semantic_extractor_output_used_as_reference_truth
        for annotation in annotations.values()
    )
    observed = {
        mention.kind
        for annotation in annotations.values()
        for page in annotation.pages
        for mention in page.mentions
    }
    assert observed == set(SemanticMentionKind)
    raw = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((root / "data/semantic_annotations").glob("*.json"))
    )
    assert '"block_id"' not in raw
    assert '"structural_node_id"' not in raw
    assert '"statement-000' not in raw


def test_committed_outputs_and_machine_evidence_are_self_consistent_offline() -> None:
    root = Path.cwd()
    evidence = json.loads(
        (root / "data/benchmarks/semantic_ir_v1_validation.v1.json").read_text(encoding="utf-8")
    )
    assert evidence["validation_schema_version"] == 2
    assert evidence["reference"]["version"] == "v2"
    assert evidence["reference"]["schema"] == 2
    assert evidence["reference"]["pages"] == 46
    assert evidence["representations"] == 10
    assert evidence["semantic_determinism"] == {
        "byte_identical": 10,
        "required": 10,
        "all_passed": True,
    }
    assert evidence["real_vlm_experiment"]["completed"] is False
    assert evidence["real_vlm_experiment"]["statement"] == (
        "REAL VLM QUALITY EXPERIMENT NOT COMPLETED"
    )
    assert evidence["selector"]["matching_protocol"]["unit"].startswith("one-to-one")
    for entry in evidence["entries"]:
        payload = (root / entry["output_path"]).read_bytes()
        document = semantic_document_from_json(payload.decode("utf-8"))
        assert document.semantic_ir_version == 1
        assert hashlib.sha256(payload).hexdigest() == entry["sha256_a"]
        assert entry["sha256_a"] == entry["sha256_b"]


def test_reference_v2_rejects_duplicate_occurrences() -> None:
    source = sorted(Path("data/semantic_annotations").glob("*.v2.json"))[0]
    payload = json.loads(source.read_text(encoding="utf-8"))
    duplicate = deepcopy(payload)
    first_mention = deepcopy(duplicate["pages"][0]["mentions"][0])
    duplicate["pages"][0]["mentions"].append(first_mention)
    with pytest.raises(ValidationError, match="duplicate reference mention ID"):
        SemanticReferenceAnnotation.model_validate(duplicate)

    wrong_span = deepcopy(payload)
    wrong_span["pages"][0]["mentions"][0]["char_end"] -= 1
    with pytest.raises(ValidationError, match="raw_text differs"):
        SemanticReferenceAnnotation.model_validate(wrong_span)

    wrong_page = deepcopy(payload)
    wrong_page["pages"][0]["mentions"][0]["page_index"] += 1
    with pytest.raises(ValidationError, match="wrong page"):
        SemanticReferenceAnnotation.model_validate(wrong_page)

    missing_legal = deepcopy(payload)
    legal = next(
        mention
        for page in missing_legal["pages"]
        for mention in page["mentions"]
        if mention["kind"] == "legal_reference"
    )
    legal["legal_reference"] = None
    with pytest.raises(ValidationError, match="legal details are required"):
        SemanticReferenceAnnotation.model_validate(missing_legal)

    wrong_quantity = deepcopy(payload)
    quantity = next(
        mention
        for page in wrong_quantity["pages"]
        for mention in page["mentions"]
        if mention["kind"] == "quantity"
    )
    quantity["quantity"]["raw_value"] = "not-in-evidence"
    with pytest.raises(ValidationError, match="quantity components differ"):
        SemanticReferenceAnnotation.model_validate(wrong_quantity)


def test_occurrence_matching_does_not_reuse_one_prediction() -> None:
    root = Path.cwd()
    annotations = load_semantic_reference_annotations(root)
    evidence = json.loads(
        (root / "data/benchmarks/semantic_ir_v1_validation.v1.json").read_text(encoding="utf-8")
    )
    entry = evidence["entries"][0]
    document = semantic_document_from_json(
        (root / entry["output_path"]).read_text(encoding="utf-8")
    )
    annotation = annotations[document.document_id]
    prediction_counts = Counter(
        (mention.evidence_anchors[0].page_index, mention.kind, mention.raw_text)
        for mention in document.mentions
    )
    reference = next(
        mention
        for page in annotation.pages
        for mention in page.mentions
        if prediction_counts[(mention.page_index, mention.kind, mention.raw_text)] == 1
    )
    source_page = next(page for page in annotation.pages if page.page_index == reference.page_index)
    source_statement = next(
        statement
        for statement in source_page.statements
        if statement.reference_statement_id == reference.reference_statement_id
    )
    duplicate = reference.model_copy(
        update={"reference_mention_id": f"{reference.reference_mention_id}-duplicate"}
    )
    focused_page = source_page.model_copy(
        update={"statements": (source_statement,), "mentions": (reference, duplicate)}
    )
    focused = annotation.model_copy(update={"pages": (focused_page,)})
    result = _evaluate_document(document, focused)
    assert result["mention_metrics"]["tp"] == 1
    assert result["mention_metrics"]["fn"] == 1


def test_normalized_edit_distance_is_local_and_deterministic() -> None:
    assert normalized_edit_distance("Quốc hội", "Quốc hội") == 0.0
    assert normalized_edit_distance("abc", "axc") == 1 / 3
    assert normalized_edit_distance("", "text") == 1.0
