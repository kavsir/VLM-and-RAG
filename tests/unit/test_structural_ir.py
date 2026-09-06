"""Structural IR v1 schema, extraction, anchoring, and serialization tests."""

import hashlib
import json
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
)
from vlm_rag.structural_ir import (
    AnchorRole,
    PhysicalAnchor,
    StructuralDiagnostic,
    StructuralDocument,
    StructuralIRSerializationError,
    StructuralNode,
    StructuralNodeKind,
    StructuralPhysicalIntegrityError,
    VietnameseStructuralExtractor,
    article_ordinal_key,
    clause_ordinal_key,
    formal_container_ordinal_key,
    generic_decimal_ordinal_key,
    point_ordinal_key,
    reconstruction_by_block,
    roman_ordinal_key,
    structural_document_from_json,
    structural_document_to_json,
    validate_against_physical,
)

SOURCE_SHA = hashlib.sha256(b"source").hexdigest()


def _block(
    block_id: str,
    text: str,
    *,
    page_index: int = 0,
    reading_order: int = 0,
    kind: BlockKindV1 = BlockKindV1.TEXT,
    disposition: BlockDisposition = BlockDisposition.CONTENT,
) -> PhysicalBlockV1:
    return PhysicalBlockV1(
        id=block_id,
        page_index=page_index,
        reading_order=reading_order,
        kind=kind,
        disposition=disposition,
        text=text,
        bbox=BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
        heading_level=1 if kind == BlockKindV1.TITLE else None,
        provenance=BlockProvenanceV1(
            parser="test",
            parser_version="1",
            parser_backend="fixture",
            source_raw_artifact="fixture.json",
            source_raw_index=reading_order,
            source_raw_type=kind.value,
        ),
    )


def _physical(*blocks: PhysicalBlockV1) -> PhysicalDocumentV1:
    return PhysicalDocumentV1(
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="test",
        parser_version="1",
        parser_backend="fixture",
        page_count=1,
        pages=(PhysicalPageV1(page_index=0, width=100.0, height=200.0, blocks=blocks),),
    )


def _extract(text: str, *, kind: BlockKindV1 = BlockKindV1.TEXT) -> StructuralDocument:
    physical = _physical(_block("b0", text, kind=kind))
    result = VietnameseStructuralExtractor().extract(physical)
    validate_against_physical(result, physical)
    return result


def test_kind_aware_ordinal_contracts() -> None:
    assert article_ordinal_key("10A") == "10a"
    assert article_ordinal_key("3Đ") == "3đ"
    assert clause_ordinal_key("2") == "2"
    assert generic_decimal_ordinal_key("1.2.3") == "1.2.3"
    assert formal_container_ordinal_key("IV") == "4"
    for letter in ("a", "c", "d", "i", "l", "m", "đ"):
        assert point_ordinal_key(letter) == letter


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("I", "1"),
        ("II", "2"),
        ("III", "3"),
        ("IV", "4"),
        ("V", "5"),
        ("IX", "9"),
        ("X", "10"),
        ("XIV", "14"),
        ("XL", "40"),
        ("XC", "90"),
        ("C", "100"),
        ("CD", "400"),
        ("D", "500"),
        ("CM", "900"),
        ("M", "1000"),
        ("MMMCMXCIX", "3999"),
    ],
)
def test_strict_valid_roman_ordinals(raw: str, expected: str) -> None:
    assert roman_ordinal_key(raw) == expected


@pytest.mark.parametrize("raw", ["IIII", "IC", "VX", "IIV", "MMMM", "IL", "XM", "VV"])
def test_invalid_roman_ordinals_are_rejected(raw: str) -> None:
    assert roman_ordinal_key(raw) is None
    assert len(_extract(f"CHƯƠNG {raw}").nodes) == 1


@pytest.mark.parametrize(
    ("source", "bad_key"),
    [
        ("Điều 1.\n1. Khoản\nc) Điểm", "100"),
        ("Điều 1.\n1. Khoản\nd) Điểm", "500"),
        ("Điều 1.\n1. Khoản\ni) Điểm", "1"),
        ("Điều 10a. Bổ sung", "999"),
        ("CHƯƠNG IV", "8"),
    ],
)
def test_domain_model_rejects_kind_aware_ordinal_corruption(source: str, bad_key: str) -> None:
    node = _extract(source).nodes[-1]
    raw = node.model_dump(mode="json")
    raw["ordinal_key"] = bad_key
    raw["canonical_path"] = raw["canonical_path"].rsplit(":", maxsplit=1)[0] + f":{bad_key}"
    with pytest.raises(ValidationError, match="kind-aware semantics"):
        StructuralNode.model_validate(raw)


def test_toc_suppression_is_event_scoped_on_mixed_page() -> None:
    result = _extract(
        "MỤC LỤC\n"
        "CHƯƠNG I ........ 2\n"
        "CHƯƠNG II ....... 8\n\n"
        "CHƯƠNG I\n"
        "QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi điều chỉnh"
    )
    assert [(node.kind, node.ordinal_key) for node in result.nodes[1:]] == [
        (StructuralNodeKind.CHAPTER, "1"),
        (StructuralNodeKind.ARTICLE, "1"),
    ]
    assert {node.canonical_path for node in result.nodes} >= {"chapter:1", "chapter:1/article:1"}


def test_two_leader_lines_do_not_suppress_real_hierarchy() -> None:
    result = _extract(
        "Hạng mục A ........ 10\n"
        "Hạng mục B ........ 20\n\n"
        "CHƯƠNG I\n"
        "QUY ĐỊNH CHUNG\n"
        "Điều 1. Phạm vi"
    )
    assert [(node.kind, node.ordinal_key) for node in result.nodes[1:]] == [
        (StructuralNodeKind.CHAPTER, "1"),
        (StructuralNodeKind.ARTICLE, "1"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Điều 3 này được áp dụng...",
        "Điều 4 được sửa đổi...",
        "Chương II quy định về...",
        "Phụ lục II hướng dẫn...",
        "Điều 3 của Luật này...",
        "Điều 4 nêu trên...",
        "Chương II của Luật...",
        "Phụ lục II kèm theo...",
        "Mục 2 của kế hoạch...",
    ],
)
def test_title_block_does_not_override_formal_prose_grammar(text: str) -> None:
    assert len(_extract(text, kind=BlockKindV1.TITLE).nodes) == 1


def test_appendix_generic_stack_closes_stale_legal_state_and_supports_letters() -> None:
    result = _extract(
        "Điều 14. Hiệu lực\n"
        "1. Khoản cũ\n"
        "Phụ lục I NỘI DUNG\n"
        "I. Nhóm A\n"
        "1. Mục A\n"
        "a) Tiểu mục A\n"
        "II. Nhóm B\n"
        "2. Mục B\n"
        "a) Tiểu mục B",
        kind=BlockKindV1.TITLE,
    )
    appendix_nodes = result.nodes[3:]
    assert appendix_nodes[0].kind == StructuralNodeKind.APPENDIX
    assert all(node.kind == StructuralNodeKind.GENERIC_SECTION for node in appendix_nodes[1:])
    assert appendix_nodes[3].recognition_evidence.recognition_method.value == (
        "generic_letter_heading"
    )
    assert appendix_nodes[-1].canonical_path == "appendix:1/generic:2/generic:2/generic:a"
    assert all("~" not in node.canonical_path for node in result.nodes)


@pytest.mark.parametrize(
    ("text", "kind", "key"),
    [
        ("Chương I", StructuralNodeKind.CHAPTER, "1"),
        ("Điều 1. Phạm vi điều chỉnh", StructuralNodeKind.ARTICLE, "1"),
        ("ĐIỀU 10a. Bổ sung", StructuralNodeKind.ARTICLE, "10a"),
        ("PHẦN II", StructuralNodeKind.PART, "2"),
        ("Mục 3", StructuralNodeKind.SECTION, "3"),
        ("Tiểu mục 2", StructuralNodeKind.SUBSECTION, "2"),
        ("PHỤ LỤC II", StructuralNodeKind.APPENDIX, "2"),
        ("CHUONG IV", StructuralNodeKind.CHAPTER, "4"),
        ("DIEU 23. Quy định", StructuralNodeKind.ARTICLE, "23"),
    ],
)
def test_formal_legal_markers(text: str, kind: StructuralNodeKind, key: str) -> None:
    node = _extract(text).nodes[1]
    assert (node.kind, node.ordinal_key) == (kind, key)


def test_article_clause_and_vietnamese_points_require_context() -> None:
    result = _extract("Điều 1. Phạm vi\n1. Nội dung\na) Điểm a\nđ) Điểm đ")
    assert [node.kind for node in result.nodes[1:]] == [
        StructuralNodeKind.ARTICLE,
        StructuralNodeKind.CLAUSE,
        StructuralNodeKind.POINT,
        StructuralNodeKind.POINT,
    ]
    article, clause, point_a, point_d = result.nodes[1:]
    assert clause.parent_id == article.id
    assert point_a.parent_id == clause.id == point_d.parent_id
    assert point_d.ordinal_key == "đ"


@pytest.mark.parametrize("letter", ["c", "d", "i", "l", "m", "đ"])
def test_point_letters_are_never_roman_normalized(letter: str) -> None:
    result = _extract(f"Điều 1. A\n1. B\n{letter}) C")
    point = result.nodes[-1]
    assert point.kind == StructuralNodeKind.POINT
    assert point.ordinal_key == letter
    assert point.canonical_path == f"article:1/clause:1/point:{letter}"


@pytest.mark.parametrize(
    "text",
    [
        "Theo Điều 3 của Luật này...",
        "quy định tại khoản 2 Điều 4",
        "tại điểm a khoản 1 Điều 5",
        "thực hiện Chương II",
        "theo Phụ lục II",
        "MỤC LỤC",
    ],
)
def test_cross_references_in_body_do_not_create_nodes(text: str) -> None:
    assert len(_extract(text).nodes) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Điều 3 của Luật này được áp dụng...",
        "Điều 4 nêu trên được sửa đổi...",
        "Chương II của Luật quy định...",
        "Phụ lục II kèm theo Thông tư này hướng dẫn...",
        "Mục 2 của kế hoạch này quy định...",
    ],
)
def test_line_start_prose_references_remain_body(text: str) -> None:
    physical = _physical(_block("b0", text))
    diagnostics: list[StructuralDiagnostic] = []
    structural = VietnameseStructuralExtractor().extract(physical, diagnostics=diagnostics)
    assert len(structural.nodes) == 1
    assert diagnostics[0].category == "line_start_prose_reference_rejected"
    assert reconstruction_by_block(structural, physical)["b0"] == text


def test_toc_pages_do_not_reserve_real_structural_paths() -> None:
    toc_text = "MỤC LỤC\nPHẦN I ........ 4\nCHƯƠNG I ...... 5\nĐIỀU 1 ........ 6\n"
    body_text = "PHẦN I\nCHƯƠNG I\nĐiều 1. Nội dung\n"
    toc_block = _block("toc", toc_text, page_index=0)
    body_block = _block("body", body_text, page_index=1)
    physical = PhysicalDocumentV1(
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="test",
        parser_version="1",
        parser_backend="fixture",
        page_count=2,
        pages=(
            PhysicalPageV1(page_index=0, width=100.0, height=200.0, blocks=(toc_block,)),
            PhysicalPageV1(page_index=1, width=100.0, height=200.0, blocks=(body_block,)),
        ),
    )
    diagnostics: list[StructuralDiagnostic] = []
    result = VietnameseStructuralExtractor().extract(physical, diagnostics=diagnostics)
    assert [node.canonical_path for node in result.nodes[1:]] == [
        "part:1",
        "part:1/chapter:1",
        "part:1/chapter:1/article:1",
    ]
    assert sum(item.category == "table_of_contents_entry_rejected" for item in diagnostics) == 3
    validate_against_physical(result, physical)
    assert reconstruction_by_block(result, physical) == {
        "toc": toc_block.text,
        "body": body_block.text,
    }


def test_plain_muc_luc_phrase_does_not_suppress_real_heading() -> None:
    result = _extract("mục lục\nCHƯƠNG I")
    assert [node.canonical_path for node in result.nodes[1:]] == ["chapter:1"]


def test_clause_and_point_markers_outside_legal_context_remain_body() -> None:
    assert len(_extract("1. Mục tiêu").nodes) == 1
    assert len(_extract("a) abc").nodes) == 1


def test_hierarchy_state_resets_at_article_chapter_and_clause() -> None:
    result = _extract(
        "Chương I\nĐiều 1. A\n1. body\na) point\n"
        "2. sibling\na) point two\nĐiều 3. B\n1. body B\n"
        "Chương II\nĐiều 10. C\n1. body C"
    )
    nodes = result.nodes
    articles = [node for node in nodes if node.kind == StructuralNodeKind.ARTICLE]
    clauses = [node for node in nodes if node.kind == StructuralNodeKind.CLAUSE]
    points = [node for node in nodes if node.kind == StructuralNodeKind.POINT]
    chapters = [node for node in nodes if node.kind == StructuralNodeKind.CHAPTER]
    assert [node.parent_id for node in articles] == [chapters[0].id, chapters[0].id, chapters[1].id]
    assert clauses[1].parent_id == articles[0].id
    assert clauses[2].parent_id == articles[1].id
    assert clauses[3].parent_id == articles[2].id
    assert points[1].parent_id == clauses[1].id


def test_one_physical_block_can_create_multiple_nodes_with_exact_partition() -> None:
    text = "Điều 1. Title\n\n1. Clause\n   a) Point\n   b) Point\n"
    physical = _physical(_block("b0", text))
    structural = VietnameseStructuralExtractor().extract(physical)
    assert [node.kind for node in structural.nodes[1:]] == [
        StructuralNodeKind.ARTICLE,
        StructuralNodeKind.CLAUSE,
        StructuralNodeKind.POINT,
        StructuralNodeKind.POINT,
    ]
    validate_against_physical(structural, physical)
    assert reconstruction_by_block(structural, physical)["b0"] == physical.pages[0].blocks[0].text


def test_cross_block_heading_title_continuation_is_conservative() -> None:
    physical = _physical(
        _block("b0", "Chương I\n", reading_order=0),
        _block("b1", "NHỮNG QUY ĐỊNH CHUNG", reading_order=1, kind=BlockKindV1.TITLE),
        _block("b2", "Đây là đoạn nội dung bình thường.", reading_order=2),
    )
    structural = VietnameseStructuralExtractor().extract(physical)
    chapter = structural.nodes[1]
    assert chapter.title == "NHỮNG QUY ĐỊNH CHUNG"
    assert [anchor.block_id for anchor in chapter.heading_anchors] == ["b0", "b1"]
    assert chapter.direct_content_anchors[0].block_id == "b2"
    validate_against_physical(structural, physical)


def test_object_is_owned_by_deepest_active_structural_node() -> None:
    physical = _physical(
        _block("b0", "Điều 5. Bảng\n", reading_order=0),
        _block("b1", "2. Nội dung", reading_order=1),
        _block("b2", "", reading_order=2, kind=BlockKindV1.TABLE),
    )
    structural = VietnameseStructuralExtractor().extract(physical)
    clause = next(node for node in structural.nodes if node.kind == StructuralNodeKind.CLAUSE)
    assert clause.direct_content_anchors[-1] == PhysicalAnchor(
        block_id="b2", page_index=0, role=AnchorRole.OBJECT
    )
    validate_against_physical(structural, physical)


def test_discarded_block_does_not_create_structure_or_content() -> None:
    physical = _physical(
        _block(
            "b0",
            "Điều 5",
            kind=BlockKindV1.PAGE_NUMBER,
            disposition=BlockDisposition.DISCARDED,
        )
    )
    structural = VietnameseStructuralExtractor().extract(physical)
    assert len(structural.nodes) == 1
    assert not structural.nodes[0].direct_content_anchors
    validate_against_physical(structural, physical)


def test_generic_planning_outline_nests_without_legal_kinds() -> None:
    result = _extract("I. QUAN ĐIỂM\n1. MỤC TIÊU\n1.1. Mục tiêu tổng quát\n1.1.1. Chi tiết")
    nodes = result.nodes[1:]
    assert all(node.kind == StructuralNodeKind.GENERIC_SECTION for node in nodes)
    assert [node.ordinal_key for node in nodes] == ["1", "1", "1.1", "1.1.1"]
    assert nodes[1].parent_id == nodes[0].id
    assert nodes[2].parent_id == nodes[1].id
    assert nodes[3].parent_id == nodes[2].id


def test_appendix_contains_generic_outline_and_title_continuation() -> None:
    physical = _physical(
        _block("b0", "PHỤ LỤC II\n", reading_order=0),
        _block("b1", "KẾ HOẠCH THỰC HIỆN", reading_order=1, kind=BlockKindV1.TITLE),
        _block("b2", "I. NỘI DUNG\n1. MỤC TIÊU\n1.1. Chi tiết", reading_order=2),
    )
    structural = VietnameseStructuralExtractor().extract(physical)
    appendix = structural.nodes[1]
    generic = structural.nodes[2:]
    assert appendix.kind == StructuralNodeKind.APPENDIX
    assert appendix.title == "KẾ HOẠCH THỰC HIỆN"
    assert generic[0].parent_id == appendix.id
    assert generic[1].parent_id == generic[0].id
    assert generic[2].parent_id == generic[1].id
    validate_against_physical(structural, physical)


def test_appendix_can_contain_part_chapter_and_section() -> None:
    result = _extract("PHỤ LỤC II\nPHẦN I. Phần\nCHƯƠNG I. Chương\nMỤC 1. Mục")
    appendix, part, chapter, section = result.nodes[1:]
    assert part.parent_id == appendix.id
    assert chapter.parent_id == part.id
    assert section.parent_id == chapter.id
    assert section.canonical_path == "appendix:2/part:1/chapter:1/section:1"


def test_ambiguous_generic_decimal_inside_clause_remains_body_and_preserves_context() -> None:
    result = _extract("Điều 1. Quy hoạch\n1. Nội dung\n1.1. Chi tiết\n2. Khoản tiếp")
    article, clause_one, clause_two = result.nodes[1:]
    assert clause_two.kind == StructuralNodeKind.CLAUSE
    assert clause_two.parent_id == article.id
    assert any(anchor.role == AnchorRole.BODY for anchor in clause_one.direct_content_anchors)


@pytest.mark.parametrize(
    "text",
    [
        "1.1. dự toán được tính như sau...",
        "2.3. số liệu được tổng hợp...",
        "1.2. trường hợp này...",
        "I. QUAN ĐIỂM\n1. đây là nội dung liệt kê thông thường",
    ],
)
def test_generic_numbering_requires_strong_heading_or_numeric_parent(text: str) -> None:
    nodes = _extract(text).nodes[1:]
    assert all(node.ordinal_key not in {"1.1", "2.3", "1.2"} for node in nodes)
    assert all(
        node.ordinal_key != "1" or node.title != "đây là nội dung liệt kê thông thường"
        for node in nodes
    )


def test_duplicate_structural_key_is_body_with_diagnostic() -> None:
    text = "Điều 1. A\nĐiều 1. B"
    physical = _physical(_block("b0", text))
    diagnostics: list[StructuralDiagnostic] = []
    result = VietnameseStructuralExtractor().extract(physical, diagnostics=diagnostics)
    assert [node.canonical_path for node in result.nodes] == ["document", "article:1"]
    duplicate = [
        item for item in diagnostics if item.category == "duplicate_structural_key_rejected"
    ]
    assert len(duplicate) == 1
    assert duplicate[0].canonical_key == "article:1"
    assert "~" not in structural_document_to_json(result)
    assert reconstruction_by_block(result, physical)["b0"] == text


def test_appendix_is_terminal_for_profile_v1() -> None:
    result = _extract("PHỤ LỤC II\nPHẦN I\nCHƯƠNG I\nĐiều 1. Nội dung")
    appendix, part, chapter, article = result.nodes[1:]
    assert part.parent_id == appendix.id
    assert chapter.parent_id == part.id
    assert article.parent_id == chapter.id


def test_non_contiguous_article_numbering_does_not_invent_nodes() -> None:
    result = _extract("Điều 1. A\nĐiều 3. B\nĐiều 3a. C\nĐiều 10. D")
    articles = [node for node in result.nodes if node.kind == StructuralNodeKind.ARTICLE]
    assert [node.ordinal_key for node in articles] == ["1", "3", "3a", "10"]
    assert len(articles) == 4


def test_model_rejects_duplicate_id_path_missing_parent_and_wrong_depth() -> None:
    document = _extract("Điều 1. A\n1. B")
    raw = document.model_dump(mode="json")
    duplicate_id = deepcopy(raw)
    duplicate_id["nodes"][2]["id"] = duplicate_id["nodes"][1]["id"]
    with pytest.raises(ValidationError, match="duplicate structural node id"):
        StructuralDocument.model_validate(duplicate_id)
    duplicate_path = _extract("Điều 1. A\nĐiều 2. B").model_dump(mode="json")
    duplicate_path["nodes"][2]["canonical_path"] = duplicate_path["nodes"][1]["canonical_path"]
    duplicate_path["nodes"][2]["ordinal_raw"] = "1"
    duplicate_path["nodes"][2]["ordinal_key"] = "1"
    with pytest.raises(ValidationError, match="duplicate canonical path"):
        StructuralDocument.model_validate(duplicate_path)
    missing_parent = deepcopy(raw)
    missing_parent["nodes"][2]["parent_id"] = "missing"
    with pytest.raises(ValidationError, match="parent must exist"):
        StructuralDocument.model_validate(missing_parent)
    wrong_depth = deepcopy(raw)
    wrong_depth["nodes"][2]["depth"] = 99
    with pytest.raises(ValidationError, match="wrong depth"):
        StructuralDocument.model_validate(wrong_depth)
    wrong_root_path = deepcopy(raw)
    wrong_root_path["nodes"][0]["canonical_path"] = "root"
    with pytest.raises(ValidationError, match="root canonical_path"):
        StructuralDocument.model_validate(wrong_root_path)
    wrong_kind_segment = deepcopy(raw)
    wrong_kind_segment["nodes"][1]["canonical_path"] = "chapter:1"
    with pytest.raises(ValidationError, match="does not match node kind"):
        StructuralDocument.model_validate(wrong_kind_segment)
    hidden_parent_segment = deepcopy(raw)
    hidden_parent_segment["nodes"][1]["canonical_path"] = "chapter:99/article:1"
    with pytest.raises(ValidationError, match="does not exactly extend parent"):
        StructuralDocument.model_validate(hidden_parent_segment)
    occurrence_suffix = deepcopy(raw)
    occurrence_suffix["nodes"][1]["canonical_path"] = "article:1~2"
    with pytest.raises(ValidationError, match="does not match node kind"):
        StructuralDocument.model_validate(occurrence_suffix)


@pytest.mark.parametrize(
    ("text", "child_index", "parent_index", "message"),
    [
        ("Chương I\nĐiều 1. A\n1. B", 3, 1, "clause cannot"),
        ("Điều 1. A\n1. B\na) C", 3, 1, "point cannot"),
    ],
)
def test_model_rejects_illegal_clause_and_point_parents(
    text: str, child_index: int, parent_index: int, message: str
) -> None:
    raw = _extract(text).model_dump(mode="json")
    raw["nodes"][child_index]["parent_id"] = raw["nodes"][parent_index]["id"]
    raw["nodes"][child_index]["depth"] = raw["nodes"][parent_index]["depth"] + 1
    with pytest.raises(ValidationError, match=message):
        StructuralDocument.model_validate(raw)


def test_physical_integrity_rejects_identity_sha_and_anchor_errors() -> None:
    physical = _physical(_block("b0", "Điều 1. A"))
    document = VietnameseStructuralExtractor().extract(physical)
    raw = document.model_dump(mode="json")
    wrong_identity = deepcopy(raw)
    wrong_identity["document_id"] = "other-document"
    with pytest.raises(StructuralPhysicalIntegrityError, match="document_id mismatch"):
        validate_against_physical(StructuralDocument.model_validate(wrong_identity), physical)
    wrong_sha = deepcopy(raw)
    wrong_sha["source_physical_ir_sha256"] = "0" * 64
    with pytest.raises(StructuralPhysicalIntegrityError, match="Physical IR SHA"):
        validate_against_physical(StructuralDocument.model_validate(wrong_sha), physical)
    missing_block = deepcopy(raw)
    missing_block["nodes"][1]["heading_anchors"][0]["block_id"] = "missing"
    with pytest.raises(StructuralPhysicalIntegrityError, match="missing block"):
        validate_against_physical(StructuralDocument.model_validate(missing_block), physical)
    wrong_page = deepcopy(raw)
    wrong_page["nodes"][1]["heading_anchors"][0]["page_index"] = 1
    with pytest.raises(StructuralPhysicalIntegrityError, match="page disagrees"):
        validate_against_physical(StructuralDocument.model_validate(wrong_page), physical)


def test_physical_integrity_rejects_out_of_bounds_and_overlapping_spans() -> None:
    physical = _physical(_block("b0", "Điều 1. A"))
    document = VietnameseStructuralExtractor().extract(physical)
    raw = document.model_dump(mode="json")
    out_of_bounds = deepcopy(raw)
    out_of_bounds["nodes"][1]["heading_anchors"][1]["char_end"] = 999
    with pytest.raises(StructuralPhysicalIntegrityError, match="exceeds text bounds"):
        validate_against_physical(StructuralDocument.model_validate(out_of_bounds), physical)
    overlap = deepcopy(raw)
    overlap["nodes"][1]["heading_anchors"][1]["char_start"] = 0
    with pytest.raises(StructuralPhysicalIntegrityError, match="overlap"):
        validate_against_physical(StructuralDocument.model_validate(overlap), physical)


def test_physical_integrity_rejects_forged_marker_title_and_recognition_evidence() -> None:
    physical = _physical(_block("b0", "Điều 1. Tiêu đề"))
    raw = VietnameseStructuralExtractor().extract(physical).model_dump(mode="json")
    forged_marker = deepcopy(raw)
    forged_marker["nodes"][1]["marker_text"] = "Điều"
    with pytest.raises(StructuralPhysicalIntegrityError, match="marker_text"):
        validate_against_physical(StructuralDocument.model_validate(forged_marker), physical)
    forged_title = deepcopy(raw)
    forged_title["nodes"][1]["title"] = "Tiêu đề khác"
    with pytest.raises(StructuralPhysicalIntegrityError, match="TITLE anchors"):
        validate_against_physical(StructuralDocument.model_validate(forged_title), physical)
    forged_evidence = deepcopy(raw)
    forged_evidence["nodes"][1]["recognition_evidence"]["matched_text"] = "Điều"
    with pytest.raises(StructuralPhysicalIntegrityError, match="recognition evidence"):
        validate_against_physical(StructuralDocument.model_validate(forged_evidence), physical)


def test_anchor_shape_is_strict() -> None:
    with pytest.raises(ValidationError, match="null character offsets"):
        PhysicalAnchor(block_id="b", page_index=0, role="object", char_start=0, char_end=1)
    with pytest.raises(ValidationError, match="require char_start"):
        PhysicalAnchor(block_id="b", page_index=0, role="body")
    with pytest.raises(ValidationError, match="char_start < char_end"):
        PhysicalAnchor(block_id="b", page_index=0, role="body", char_start=1, char_end=1)


def test_serialization_is_deterministic_unicode_lf_and_version_strict() -> None:
    document = _extract("Điều 1. Tiếng Việt\n1. Nội dung")
    payload_a = structural_document_to_json(document)
    payload_b = structural_document_to_json(document)
    assert payload_a == payload_b
    assert "Tiếng Việt" in payload_a
    assert payload_a.endswith("\n") and "\r" not in payload_a
    assert structural_document_from_json(payload_a) == document
    wrong_version = json.loads(payload_a)
    wrong_version["structural_ir_version"] = 2
    with pytest.raises(StructuralIRSerializationError, match="unsupported"):
        structural_document_from_json(json.dumps(wrong_version))


def test_structural_models_are_frozen_strict_and_forbid_extra_fields() -> None:
    document = _extract("Điều 1. A")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StructuralDocument.model_validate({**document.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        document.nodes[0].depth = 2  # type: ignore[misc]
