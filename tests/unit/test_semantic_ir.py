"""Domain Semantic IR v1 contracts and deterministic extraction tests."""

import hashlib
from copy import deepcopy

import pytest
from pydantic import ValidationError

from vlm_rag.physical_ir import (
    BlockDisposition,
    BlockKindV1,
    BlockProvenanceV1,
    BoundingBox,
    PhysicalBlockV1,
    PhysicalDocumentV1,
    PhysicalPageV1,
    TextExtractionEvidence,
    TextExtractionMethod,
    physical_document_v1_to_json,
)
from vlm_rag.semantic_ir import (
    SemanticDocument,
    SemanticIRSerializationError,
    SemanticMentionKind,
    SemanticProfile,
    SemanticSourceIntegrityError,
    build_semantic_document,
    extract_mention_candidates,
    semantic_document_from_json,
    semantic_document_to_json,
    validate_semantic_sources,
)
from vlm_rag.structural_ir import (
    VietnameseStructuralExtractor,
    structural_document_to_json,
    validate_against_physical,
)

SOURCE_SHA = hashlib.sha256(b"source-pdf").hexdigest()


def _physical(text: str) -> PhysicalDocumentV1:
    block = PhysicalBlockV1(
        id="test-document_v1_p0000_b0000",
        page_index=0,
        reading_order=0,
        kind=BlockKindV1.TEXT,
        disposition=BlockDisposition.CONTENT,
        text=text,
        bbox=BoundingBox(x0=10.0, y0=20.0, x1=900.0, y1=200.0),
        provenance=BlockProvenanceV1(
            parser="fixture",
            parser_version="1",
            parser_backend="test",
            source_raw_artifact="fixture.json",
            source_raw_index=0,
            source_raw_type="text",
        ),
        text_extraction=TextExtractionEvidence(method=TextExtractionMethod.NATIVE_TEXT),
    )
    return PhysicalDocumentV1(
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="fixture",
        parser_version="1",
        parser_backend="test",
        page_count=1,
        pages=(PhysicalPageV1(page_index=0, width=100.0, height=200.0, blocks=(block,)),),
    )


def _semantic(text: str) -> tuple[PhysicalDocumentV1, object, SemanticDocument]:
    physical = _physical(text)
    structural = VietnameseStructuralExtractor().extract(physical)
    validate_against_physical(structural, physical)
    semantic = build_semantic_document(physical, structural)
    validate_semantic_sources(semantic, physical, structural)
    return physical, structural, semantic


def test_semantic_identity_hashes_exact_spans_and_serialization() -> None:
    text = (
        "Điều 1. Phạm vi\nTheo Điều 3 Nghị định số 12/2024/NĐ-CP, Bộ Xây dựng "
        "ban hành ngày 13/05/2026 cho 25 ha."
    )
    physical, structural, semantic = _semantic(text)
    assert semantic.semantic_ir_version == 1
    assert semantic.profile == SemanticProfile.VI_LEGAL_PLANNING_SEMANTIC_V1
    assert (
        semantic.source_physical_ir_sha256
        == hashlib.sha256(physical_document_v1_to_json(physical).encode()).hexdigest()
    )
    assert (
        semantic.source_structural_ir_sha256
        == hashlib.sha256(structural_document_to_json(structural).encode()).hexdigest()
    )
    block = physical.pages[0].blocks[0]
    for mention in semantic.mentions:
        anchor = mention.evidence_anchors[0]
        assert anchor.anchor_type == "text"
        assert block.text[anchor.char_start : anchor.char_end] == mention.raw_text
    payload = semantic_document_to_json(semantic)
    assert payload.endswith("\n") and "\r\n" not in payload
    assert semantic_document_to_json(semantic_document_from_json(payload)) == payload


def test_controlled_mentions_preserve_raw_and_normalize_only_when_safe() -> None:
    text = (
        "2512/QĐ-UBND 112/2025/QH15 04/2026/BXD 103/VBHN-VPQH; "
        "ngày 13 tháng 5 năm 2026; 12,5 km; 1.000 ha; Quốc hội; Chính phủ."
    )
    candidates = extract_mention_candidates(text)
    identifiers = [
        item for item in candidates if item.kind == SemanticMentionKind.DOCUMENT_IDENTIFIER
    ]
    assert [item.raw_text for item in identifiers] == [
        "2512/QĐ-UBND",
        "112/2025/QH15",
        "04/2026/BXD",
        "103/VBHN-VPQH",
    ]
    assert all(item.normalized_value == item.raw_text for item in identifiers)
    temporal = next(
        item for item in candidates if item.kind == SemanticMentionKind.TEMPORAL_EXPRESSION
    )
    assert temporal.raw_text == "ngày 13 tháng 5 năm 2026"
    assert temporal.normalized_value == "2026-05-13"
    quantities = [item for item in candidates if item.kind == SemanticMentionKind.QUANTITY]
    assert quantities[0].normalized_value == "12.5 km"
    assert quantities[1].normalized_value is None
    authorities = [
        item.raw_text
        for item in candidates
        if item.kind == SemanticMentionKind.ORGANIZATION_OR_AUTHORITY
    ]
    assert authorities == ["Quốc hội", "Chính phủ"]


def test_legal_reference_components_and_no_external_resolution() -> None:
    candidate = next(
        item
        for item in extract_mention_candidates(
            "theo điểm a khoản 2 Điều 3 Nghị định số 12/2024/NĐ-CP"
        )
        if item.kind == SemanticMentionKind.LEGAL_REFERENCE
    )
    assert candidate.legal_reference is not None
    assert candidate.legal_reference.point == "a"
    assert candidate.legal_reference.clause == "2"
    assert candidate.legal_reference.article == "3"
    assert candidate.legal_reference.instrument_number_normalized == "12/2024/NĐ-CP"
    assert candidate.legal_reference.scope.value == "external_document"


def test_authority_extraction_is_not_capitalization_ner() -> None:
    candidates = extract_mention_candidates("Công Ty Mặt Trời và viện nghiên cứu địa phương")
    assert not any(
        item.kind == SemanticMentionKind.ORGANIZATION_OR_AUTHORITY for item in candidates
    )


def test_semantic_source_mismatch_and_duplicate_ids_fail() -> None:
    physical, structural, semantic = _semantic("Điều 1. A\nQuốc hội ban hành 112/2025/QH15")
    raw = semantic.model_dump(mode="json")
    raw["source_structural_ir_sha256"] = "0" * 64
    mismatched = SemanticDocument.model_validate(raw)
    with pytest.raises(SemanticSourceIntegrityError, match="source_structural_ir_sha256 mismatch"):
        validate_semantic_sources(mismatched, physical, structural)
    duplicate = semantic.model_dump(mode="json")
    duplicate["statements"].append(deepcopy(duplicate["statements"][0]))
    with pytest.raises(ValidationError, match="duplicate semantic statement id"):
        SemanticDocument.model_validate(duplicate)
    with pytest.raises(SemanticIRSerializationError, match="unsupported semantic_ir_version"):
        semantic_document_from_json('{"semantic_ir_version":2}')


def test_semantic_generation_does_not_mutate_physical_or_structural_ir() -> None:
    physical = _physical("Điều 1. A\nChính phủ sử dụng 12 km")
    structural = VietnameseStructuralExtractor().extract(physical)
    physical_before = physical_document_v1_to_json(physical)
    structural_before = structural_document_to_json(structural)
    first = build_semantic_document(physical, structural)
    second = build_semantic_document(physical, structural)
    assert semantic_document_to_json(first) == semantic_document_to_json(second)
    assert physical_document_v1_to_json(physical) == physical_before
    assert structural_document_to_json(structural) == structural_before
