"""Offline committed Semantic IR v1 research-evidence tests."""

import hashlib
import json
from pathlib import Path

from vlm_rag.evaluation.semantic import (
    load_semantic_reference_annotations,
    normalized_edit_distance,
)
from vlm_rag.semantic_ir import SemanticMentionKind, semantic_document_from_json


def test_reference_is_disclosed_fixed_46_page_candidate_audit() -> None:
    root = Path.cwd()
    annotations = load_semantic_reference_annotations(root)
    assert len(annotations) == 6
    assert sum(len(annotation.pages) for annotation in annotations.values()) == 46
    assert all(annotation.prior_semantic_extractor_exposure for annotation in annotations.values())
    assert all(
        not annotation.independent_or_blind_ground_truth for annotation in annotations.values()
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
    for entry in evidence["entries"]:
        payload = (root / entry["output_path"]).read_bytes()
        document = semantic_document_from_json(payload.decode("utf-8"))
        assert document.semantic_ir_version == 1
        assert hashlib.sha256(payload).hexdigest() == entry["sha256_a"]
        assert entry["sha256_a"] == entry["sha256_b"]


def test_normalized_edit_distance_is_local_and_deterministic() -> None:
    assert normalized_edit_distance("Quốc hội", "Quốc hội") == 0.0
    assert normalized_edit_distance("abc", "axc") == 1 / 3
    assert normalized_edit_distance("", "text") == 1.0
