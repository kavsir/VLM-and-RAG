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
    SemanticReferenceMention,
    SemanticReferencePage,
    SemanticReferenceStatement,
    VisualAssistanceReference,
    _evaluate_document,
    _evaluate_selector,
    load_semantic_reference_annotations,
    normalized_edit_distance,
)
from vlm_rag.physical_ir import BoundingBox
from vlm_rag.semantic_ir import (
    LegalReferenceComponents,
    LegalReferenceScope,
    SemanticDocument,
    SemanticMention,
    SemanticMentionKind,
    SemanticProvenanceKind,
    SemanticStatement,
    TextSemanticAnchor,
    semantic_document_from_json,
)
from vlm_rag.vlm import (
    SelectionReason,
    VisualEvidenceRequest,
    VLMSelectionResult,
    VLMTaskType,
)


def _semantic_fixture(
    *,
    raw_text: str,
    kind: SemanticMentionKind,
    occurrences: tuple[tuple[str, str, str | None, LegalReferenceComponents | None], ...],
) -> SemanticDocument:
    statements: list[SemanticStatement] = []
    mentions: list[SemanticMention] = []
    for index, (identifier, text, normalized, legal) in enumerate(occurrences):
        char_start = text.index(raw_text)
        char_end = char_start + len(raw_text)
        statement_id = f"runtime-statement-{identifier}"
        mention_id = f"runtime-mention-{identifier}"
        block_id = f"runtime-block-{index}"
        statements.append(
            SemanticStatement(
                id=statement_id,
                structural_node_id="node-document",
                structural_canonical_path="document",
                text=text,
                evidence_anchors=(
                    TextSemanticAnchor(
                        block_id=block_id,
                        page_index=0,
                        char_start=0,
                        char_end=len(text),
                    ),
                ),
                mention_ids=(mention_id,),
                provenance=SemanticProvenanceKind.RULE_BASED_TEXT,
            )
        )
        mentions.append(
            SemanticMention(
                id=mention_id,
                kind=kind,
                statement_id=statement_id,
                raw_text=raw_text,
                normalized_value=normalized,
                evidence_anchors=(
                    TextSemanticAnchor(
                        block_id=block_id,
                        page_index=0,
                        char_start=char_start,
                        char_end=char_end,
                    ),
                ),
                provenance=SemanticProvenanceKind.RULE_BASED_TEXT,
                legal_reference=legal,
            )
        )
    return SemanticDocument(
        document_id="synthetic-document",
        version_id="v1",
        source_artifact_sha256="1" * 64,
        source_physical_ir_sha256="2" * 64,
        source_structural_ir_sha256="3" * 64,
        statements=tuple(statements),
        mentions=tuple(mentions),
    )


def _reference_fixture(
    *,
    raw_text: str,
    kind: SemanticMentionKind,
    occurrences: tuple[tuple[str, str, str | None, LegalReferenceComponents | None], ...],
    visual_assistance: tuple[VisualAssistanceReference, ...] = (),
) -> SemanticReferenceAnnotation:
    statements: list[SemanticReferenceStatement] = []
    mentions: list[SemanticReferenceMention] = []
    for identifier, text, normalized, legal in occurrences:
        statement_id = f"reference-statement-{identifier}"
        char_start = text.index(raw_text)
        statements.append(
            SemanticReferenceStatement(
                reference_statement_id=statement_id,
                page_index=0,
                exact_evidence_excerpt=text,
                audit_note="synthetic adversarial occurrence fixture",
            )
        )
        mentions.append(
            SemanticReferenceMention(
                reference_mention_id=f"reference-mention-{identifier}",
                reference_statement_id=statement_id,
                page_index=0,
                kind=kind,
                raw_text=raw_text,
                normalized_value=normalized,
                char_start=char_start,
                char_end=char_start + len(raw_text),
                audit_note="synthetic adversarial occurrence fixture",
                legal_reference=legal,
            )
        )
    return SemanticReferenceAnnotation(
        annotator_type="ai_visual_audit",
        annotation_method="visual_pdf_semantic_reaudit",
        document_id="synthetic-document",
        version_id="v1",
        source_artifact_sha256="1" * 64,
        prior_physical_ir_exposure=True,
        prior_structural_ir_exposure=True,
        prior_semantic_extractor_exposure=True,
        semantic_extractor_candidates_seen=True,
        semantic_extractor_output_used_as_reference_truth=False,
        independent_or_blind_ground_truth=False,
        pages=(
            SemanticReferencePage(
                page_index=0,
                pdf_page_number_1_based=1,
                render_sha256="4" * 64,
                statements=tuple(statements),
                mentions=tuple(mentions),
                visual_assistance=visual_assistance,
            ),
        ),
    )


def _visual_request(request_id: str, bbox: BoundingBox) -> VisualEvidenceRequest:
    return VisualEvidenceRequest(
        request_id=request_id,
        document_id="synthetic-document",
        version_id="v1",
        source_artifact_sha256="1" * 64,
        page_index=0,
        bbox=bbox,
        source_block_ids=(f"block-{request_id}",),
        task_type=VLMTaskType.OCR_RECOVERY,
        selection_reason=SelectionReason.OCR_CORRUPTION_SIGNAL,
        priority=0,
    )


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
    assert evidence["selector"]["matching_protocol"]["optimization"].startswith(
        "maximum cardinality"
    )
    assert evidence["selector"]["matched_total_overlap_quality"] >= 0.0
    assert evidence["text_only"]["ambiguous_occurrence_pairing_count"] >= 0
    for metric in (
        "exact_evidence_span_match",
        "normalized_value_exact_match",
        "legal_reference_component_exact_match",
    ):
        assert evidence["text_only"][metric]["ambiguous_excluded"] >= 0
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


def test_occurrence_pairing_uses_exact_statement_and_span_before_order() -> None:
    reference = _reference_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(
            ("a", "A ... UBND ...", None, None),
            ("b", "B ... UBND ...", None, None),
        ),
    )
    prediction = _semantic_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(("b", "B ... UBND ...", None, None),),
    )
    result = _evaluate_document(prediction, reference)
    assert result["mention_metrics"]["tp"] == 1
    assert result["mention_metrics"]["fn"] == 1
    assert result["ambiguous_occurrence_pairing_count"] == 0
    assert result["exact_evidence_span_match"] == {
        "exact": 1,
        "eligible": 1,
        "ambiguous_excluded": 0,
        "rate": 1.0,
    }
    assert result["false_negative_trace"][0]["reference_mention_id"].endswith("-a")


def test_ambiguous_normalization_pairing_is_excluded_without_target_leakage() -> None:
    reference = _reference_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(
            ("a", "X UBND", "NORMALIZED-A", None),
            ("b", "X UBND", "NORMALIZED-B", None),
        ),
    )
    prediction = _semantic_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(("prediction", "X UBND", "NORMALIZED-B", None),),
    )
    result = _evaluate_document(prediction, reference)
    assert result["mention_metrics"]["tp"] == 1
    assert result["mention_metrics"]["fn"] == 1
    assert result["ambiguous_occurrence_pairing_count"] == 1
    assert result["normalized_value_exact_match"] == {
        "exact": 0,
        "eligible": 0,
        "ambiguous_excluded": 1,
        "rate": None,
    }
    assert result["exact_evidence_span_match"]["exact"] == 1
    assert result["exact_evidence_span_match"]["ambiguous_excluded"] == 0


def test_duplicate_legal_surface_pairs_by_independent_statement_context() -> None:
    self_document = LegalReferenceComponents(article="3", scope=LegalReferenceScope.SELF_DOCUMENT)
    external_document = LegalReferenceComponents(
        article="3", scope=LegalReferenceScope.EXTERNAL_DOCUMENT
    )
    reference = _reference_fixture(
        raw_text="Điều 3",
        kind=SemanticMentionKind.LEGAL_REFERENCE,
        occurrences=(
            ("a", "A viện dẫn Điều 3", None, self_document),
            ("b", "B viện dẫn Điều 3", None, external_document),
        ),
    )
    prediction = _semantic_fixture(
        raw_text="Điều 3",
        kind=SemanticMentionKind.LEGAL_REFERENCE,
        occurrences=(("b", "B viện dẫn Điều 3", None, external_document),),
    )
    result = _evaluate_document(prediction, reference)
    assert result["mention_metrics"]["tp"] == 1
    assert result["mention_metrics"]["fn"] == 1
    assert result["ambiguous_occurrence_pairing_count"] == 0
    assert result["legal_reference_component_exact_match"]["exact"] == 1
    assert result["legal_reference_component_exact_match"]["eligible"] == 1
    assert result["legal_reference_component_exact_match"]["ambiguous_excluded"] == 0


def test_normalization_is_scored_after_context_pairing_without_target_leakage() -> None:
    reference = _reference_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(
            ("a", "A UBND", "VALUE-A", None),
            ("b", "B UBND", "VALUE-B", None),
        ),
    )
    prediction = _semantic_fixture(
        raw_text="UBND",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(("b", "B UBND", "VALUE-A", None),),
    )
    result = _evaluate_document(prediction, reference)
    assert result["mention_metrics"]["tp"] == 1
    assert result["exact_evidence_span_match"]["exact"] == 1
    assert result["normalized_value_exact_match"] == {
        "exact": 0,
        "eligible": 1,
        "ambiguous_excluded": 0,
        "rate": 0.0,
    }


def test_ambiguous_legal_components_are_excluded_when_outcomes_differ() -> None:
    self_document = LegalReferenceComponents(article="3", scope=LegalReferenceScope.SELF_DOCUMENT)
    external_document = LegalReferenceComponents(
        article="3", scope=LegalReferenceScope.EXTERNAL_DOCUMENT
    )
    reference = _reference_fixture(
        raw_text="Article 3",
        kind=SemanticMentionKind.LEGAL_REFERENCE,
        occurrences=(
            ("a", "Same Article 3", None, self_document),
            ("b", "Same Article 3", None, external_document),
        ),
    )
    prediction = _semantic_fixture(
        raw_text="Article 3",
        kind=SemanticMentionKind.LEGAL_REFERENCE,
        occurrences=(("prediction", "Same Article 3", None, external_document),),
    )
    result = _evaluate_document(prediction, reference)
    legal = result["legal_reference_component_exact_match"]
    assert result["mention_metrics"]["tp"] == 1
    assert result["ambiguous_occurrence_pairing_count"] == 1
    assert legal["eligible"] == 0
    assert legal["ambiguous_excluded"] == 1
    assert legal["by_component"]["scope"]["ambiguous_excluded"] == 1
    assert legal["by_component"]["article"]["exact"] == 1


def test_selector_matching_is_maximum_cardinality_and_id_invariant() -> None:
    references = (
        VisualAssistanceReference(
            reference_id="reference-one",
            page_index=0,
            bbox_normalized_1000=(0, 0, 100, 100),
            task_type=VLMTaskType.OCR_RECOVERY,
            need_reason="synthetic maximum-cardinality fixture",
            evidence_note="R1 can match P1 or P2",
        ),
        VisualAssistanceReference(
            reference_id="reference-two",
            page_index=0,
            bbox_normalized_1000=(0, 0, 40, 100),
            task_type=VLMTaskType.OCR_RECOVERY,
            need_reason="synthetic maximum-cardinality fixture",
            evidence_note="R2 can match only P1",
        ),
    )
    requests = (
        _visual_request("z-request-one", BoundingBox(x0=0.0, y0=0.0, x1=40.0, y1=100.0)),
        _visual_request("a-request-two", BoundingBox(x0=60.0, y0=0.0, x1=100.0, y1=100.0)),
    )
    annotation = _reference_fixture(
        raw_text="unused",
        kind=SemanticMentionKind.ORGANIZATION_OR_AUTHORITY,
        occurrences=(),
        visual_assistance=references,
    )
    selection = VLMSelectionResult(requests=requests, non_selections=())
    baseline = _evaluate_selector(selection, annotation)
    assert baseline["tp"] == 2
    assert baseline["fp"] == 0
    assert baseline["fn"] == 0

    swapped_references = (
        references[0].model_copy(update={"reference_id": references[1].reference_id}),
        references[1].model_copy(update={"reference_id": references[0].reference_id}),
    )
    swapped_reference_page = annotation.pages[0].model_copy(
        update={"visual_assistance": swapped_references}
    )
    swapped_annotation = annotation.model_copy(update={"pages": (swapped_reference_page,)})
    swapped_requests = (
        requests[0].model_copy(update={"request_id": requests[1].request_id}),
        requests[1].model_copy(update={"request_id": requests[0].request_id}),
    )
    swapped_selection = selection.model_copy(update={"requests": swapped_requests})
    for result in (
        _evaluate_selector(selection, swapped_annotation),
        _evaluate_selector(swapped_selection, annotation),
    ):
        for field in (
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "matched_total_iou",
            "matched_total_reference_containment",
            "matched_total_overlap_quality",
        ):
            assert result[field] == baseline[field]


def test_normalized_edit_distance_is_local_and_deterministic() -> None:
    assert normalized_edit_distance("Quốc hội", "Quốc hội") == 0.0
    assert normalized_edit_distance("abc", "axc") == 1 / 3
    assert normalized_edit_distance("", "text") == 1.0
