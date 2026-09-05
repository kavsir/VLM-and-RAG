"""Physical IR v1 schema, serialization, and evidence-mapping tests."""

import base64
import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.normalizers.marker import MarkerPhysicalNormalizer
from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1
from vlm_rag.physical_ir import (
    BlockDisposition,
    BlockKindV1,
    BlockProvenanceV1,
    BoundingBox,
    PhysicalBlockV1,
    PhysicalDocument,
    PhysicalDocumentV1,
    PhysicalIRSerializationError,
    PhysicalIRV1SerializationError,
    PhysicalIRVersionError,
    PhysicalPageV1,
    TableCell,
    TableStructure,
    TextExtractionEvidence,
    TextExtractionMethod,
    VisualAssetEvidence,
    VisualAssetStorage,
    load_physical_document,
    physical_document_from_json,
    physical_document_to_json,
    physical_document_v1_from_json,
    physical_document_v1_to_json,
)
from vlm_rag.physical_ir.table_html import TableHTMLStructureError, table_structure_from_html
from vlm_rag.physical_ir.validation_v1 import render_physical_ir_v1_validation

MARKER_FIXTURE = Path("tests/fixtures/marker/sample.json")
MINERU_CONTENT_FIXTURE = Path("tests/fixtures/mineru/sample_content_list.json")
MINERU_MIDDLE_FIXTURE = Path("tests/fixtures/mineru/sample_middle.json")
SOURCE_SHA = hashlib.sha256(b"source").hexdigest()


def _provenance(raw_type: str = "Text") -> BlockProvenanceV1:
    return BlockProvenanceV1(
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        source_raw_artifact="source/source.json",
        source_raw_index=0,
        source_raw_type=raw_type,
    )


def _block(kind: BlockKindV1 = BlockKindV1.TEXT, **overrides: object) -> PhysicalBlockV1:
    values: dict[str, object] = {
        "id": "test-document_v1_p0000_b0000",
        "page_index": 0,
        "reading_order": 0,
        "kind": kind,
        "disposition": BlockDisposition.CONTENT,
        "text": "evidence",
        "bbox": BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
        "provenance": _provenance(),
        "text_extraction": TextExtractionEvidence(method=TextExtractionMethod.UNKNOWN),
    }
    values.update(overrides)
    return PhysicalBlockV1(**values)


def _document(block: PhysicalBlockV1 | None = None) -> PhysicalDocumentV1:
    blocks = () if block is None else (block,)
    return PhysicalDocumentV1(
        physical_ir_version=2,
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=1,
        pages=(PhysicalPageV1(page_index=0, width=100.0, height=200.0, blocks=blocks),),
    )


def _write_marker_raw(tmp_path: Path, transform: object | None = None) -> Path:
    raw = tmp_path / "marker-run" / "raw"
    source = raw / "source"
    source.mkdir(parents=True)
    tree = json.loads(MARKER_FIXTURE.read_text(encoding="utf-8"))
    if callable(transform):
        transform(tree)
    (source / "source.json").write_text(json.dumps(tree), encoding="utf-8")
    (source / "source_meta.json").write_text(
        json.dumps({"page_stats": [{"page_id": 0, "text_extraction_method": "pdftext"}]}),
        encoding="utf-8",
    )
    run = {
        "parser": "marker",
        "parser_version": "2.0.0",
        "backend": "fast-no-ocr",
        "mode": "fast",
        "disable_ocr": True,
        "output_format": "json",
        "document_id": "test-document",
        "version_id": "v1",
        "input_sha256": SOURCE_SHA,
    }
    (raw.parent / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return raw


def _write_mineru_raw(tmp_path: Path, items: list[dict[str, object]]) -> Path:
    raw = tmp_path / "mineru-run" / "raw"
    output = raw / "source" / "auto"
    output.mkdir(parents=True)
    (output / "source_content_list.json").write_text(json.dumps(items), encoding="utf-8")
    (output / "source_middle.json").write_text(
        json.dumps(
            {
                "_version_name": "3.4.5",
                "_backend": "pipeline",
                "pdf_info": [{"page_idx": 0, "page_size": [100.0, 200.0]}],
            }
        ),
        encoding="utf-8",
    )
    run = {
        "parser": "mineru",
        "parser_version": "3.4.5",
        "backend": "pipeline",
        "document_id": "test-document",
        "version_id": "v1",
        "input_sha256": SOURCE_SHA,
    }
    (raw.parent / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return raw


def test_v1_wire_version_and_physical_kinds() -> None:
    assert _document().physical_ir_version == 2
    assert {kind.value for kind in BlockKindV1} == {
        "text",
        "title",
        "header",
        "page_number",
        "table",
        "figure",
        "image",
        "unknown",
    }
    for kind in (BlockKindV1.TABLE, BlockKindV1.FIGURE, BlockKindV1.IMAGE):
        assert _block(kind).kind == kind


def test_v1_models_are_strict_and_conditional_fields_are_constrained() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _document().model_copy(update={"unexpected": True}).model_validate(
            {**_document().model_dump(), "unexpected": True}
        )
    structure = TableStructure(row_count=0, column_count=0, cells=())
    with pytest.raises(ValidationError, match="only permitted for TABLE"):
        _block(BlockKindV1.TEXT, table_structure=structure)
    asset = VisualAssetEvidence(storage_kind=VisualAssetStorage.UNAVAILABLE)
    with pytest.raises(ValidationError, match="only permitted for FIGURE or IMAGE"):
        _block(BlockKindV1.TABLE, visual_asset=asset)
    with pytest.raises(ValidationError, match="only permitted for TITLE"):
        _block(BlockKindV1.TEXT, heading_level=1)
    with pytest.raises(ValidationError, match="requires non-empty"):
        _block(text="", text_extraction=TextExtractionEvidence(method="unknown"))


def test_v1_page_and_document_invariants() -> None:
    with pytest.raises(ValidationError, match="does not belong"):
        PhysicalPageV1(page_index=0, blocks=(_block(page_index=1),))
    with pytest.raises(ValidationError, match="reading_order"):
        PhysicalPageV1(page_index=0, blocks=(_block(reading_order=1),))
    with pytest.raises(ValidationError, match="page_count"):
        PhysicalDocumentV1(
            document_id="test-document",
            version_id="v1",
            source_artifact_sha256=SOURCE_SHA,
            parser="marker",
            parser_version="2.0.0",
            parser_backend="fast-no-ocr",
            page_count=0,
            pages=(PhysicalPageV1(page_index=0),),
        )
    duplicate = _block()
    duplicate_page_one = _block(page_index=1)
    with pytest.raises(ValidationError, match="duplicate block id"):
        PhysicalDocumentV1(
            document_id="test-document",
            version_id="v1",
            source_artifact_sha256=SOURCE_SHA,
            parser="marker",
            parser_version="2.0.0",
            parser_backend="fast-no-ocr",
            page_count=2,
            pages=(
                PhysicalPageV1(page_index=0, blocks=(duplicate,)),
                PhysicalPageV1(page_index=1, blocks=(duplicate_page_one,)),
            ),
        )


def test_table_grid_spans_overflow_and_overlap() -> None:
    spanning = TableCell(
        row_start=0,
        column_start=0,
        row_span=2,
        column_span=2,
        text="span",
        is_header=True,
    )
    assert TableStructure(row_count=2, column_count=2, cells=(spanning,)).cells == (spanning,)
    with pytest.raises(ValidationError, match="exceeds declared grid"):
        TableStructure(row_count=1, column_count=2, cells=(spanning,))
    overlapping = TableCell(
        row_start=1,
        column_start=1,
        row_span=1,
        column_span=1,
        text="overlap",
        is_header=False,
    )
    with pytest.raises(ValidationError, match="overlap"):
        TableStructure(row_count=2, column_count=2, cells=(spanning, overlapping))
    for field in ("row_span", "column_span"):
        with pytest.raises(ValidationError):
            TableCell(
                row_start=0,
                column_start=0,
                row_span=0 if field == "row_span" else 1,
                column_span=0 if field == "column_span" else 1,
                text="invalid",
                is_header=None,
            )


def test_table_html_simple_headers_rowspan_colspan_and_empty() -> None:
    structure = table_structure_from_html(
        "<table><tr><th rowspan='2'>H</th><th colspan='2'>Group</th></tr>"
        "<tr><td>A</td><td>B</td></tr></table>"
    )
    assert structure is not None
    assert (structure.row_count, structure.column_count, len(structure.cells)) == (2, 3, 4)
    assert structure.cells[0].is_header is True
    assert structure.cells[0].row_span == 2
    assert structure.cells[1].column_span == 2
    assert structure.cells[2].is_header is False
    assert table_structure_from_html("<table></table>") == TableStructure(
        row_count=0, column_count=0, cells=()
    )
    assert table_structure_from_html("plain text") is None
    with pytest.raises(TableHTMLStructureError):
        table_structure_from_html("<table><tr><td rowspan='0'>bad</td></tr></table>")
    with pytest.raises(TableHTMLStructureError, match="multiple"):
        table_structure_from_html("<table></table><table></table>")


@pytest.mark.parametrize(
    ("html", "valid", "expected_rows"),
    [
        ("<table><tr><td rowspan='10'>A</td></tr></table>", False, None),
        ("<table><tr><td rowspan='2'>A</td></tr><tr></tr></table>", True, 2),
        ("<table><tr><td rowspan='3'>A</td></tr><tr></tr></table>", False, None),
        (
            "<table><tr><th rowspan='2' colspan='2'>A</th><td>B</td></tr>"
            "<tr><td>C</td></tr></table>",
            True,
            2,
        ),
    ],
)
def test_table_html_rowspan_requires_explicit_rows(
    html: str, valid: bool, expected_rows: int | None
) -> None:
    if not valid:
        with pytest.raises(TableHTMLStructureError, match="explicit tr rows"):
            table_structure_from_html(html)
        return
    structure = table_structure_from_html(html)
    assert structure is not None
    assert structure.row_count == expected_rows


def test_table_cell_br_preserves_lines_and_vietnamese() -> None:
    structure = table_structure_from_html(
        "<table><tr><td>  Hà   Nội <br> dòng\t hai<br><br> cuối  </td></tr></table>"
    )
    assert structure is not None
    assert structure.cells[0].text == "Hà Nội\ndòng hai\n\ncuối"


@pytest.mark.parametrize(
    ("method", "confidence"),
    [
        (TextExtractionMethod.NATIVE_TEXT, None),
        (TextExtractionMethod.OCR, 0.0),
        (TextExtractionMethod.UNKNOWN, 1.0),
    ],
)
def test_text_extraction_methods_and_confidence_bounds(
    method: TextExtractionMethod, confidence: float | None
) -> None:
    evidence = TextExtractionEvidence(method=method, confidence=confidence)
    assert evidence.method == method
    assert evidence.confidence == confidence


def test_invalid_text_extraction_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TextExtractionEvidence(method="guessed")
    for confidence in (-0.01, 1.01, float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValidationError):
            TextExtractionEvidence(method="ocr", confidence=confidence)


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_v1_page_dimensions_must_be_finite(field: str, value: float) -> None:
    values: dict[str, object] = {"page_index": 0, "width": 1.0, "height": 1.0}
    values[field] = value
    with pytest.raises(ValidationError):
        PhysicalPageV1(**values)


@pytest.mark.parametrize("field", ["x0", "y0", "x1", "y1"])
def test_v1_rejects_forged_non_finite_v0_bbox(field: str) -> None:
    values = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}
    values[field] = float("nan")
    forged = BoundingBox.model_construct(**values, coordinate_system="normalized_1000")
    with pytest.raises(ValidationError, match="finite"):
        _block(bbox=forged)


def test_visual_asset_storage_contract_and_paths() -> None:
    relative = VisualAssetEvidence(
        storage_kind="relative_file",
        relative_path="source/auto/images/a.jpg",
        media_type="image/jpeg",
        sha256="a" * 64,
        byte_size=4,
    )
    embedded = VisualAssetEvidence(
        storage_kind="embedded_raw",
        media_type="image/jpeg",
        sha256="b" * 64,
        byte_size=4,
    )
    unavailable = VisualAssetEvidence(storage_kind="unavailable")
    assert relative.relative_path and embedded.relative_path is None
    assert unavailable.sha256 is None
    for path in ("/absolute.jpg", "../escape.jpg", "C:/absolute.jpg", "a\\b.jpg"):
        with pytest.raises(ValidationError, match="relative"):
            VisualAssetEvidence(storage_kind="relative_file", relative_path=path)
    with pytest.raises(ValidationError):
        VisualAssetEvidence(
            storage_kind="embedded_raw", media_type="image/jpeg", sha256="bad", byte_size=4
        )
    with pytest.raises(ValidationError, match="requires"):
        VisualAssetEvidence(storage_kind="embedded_raw", media_type="image/jpeg")


@pytest.mark.parametrize(
    "overrides",
    [
        {"relative_path": "images/a.png"},
        {"relative_path": "images/a.png", "sha256": "a" * 64},
        {"relative_path": "images/a.png", "byte_size": 1},
        {"sha256": "a" * 64, "byte_size": 1},
    ],
)
def test_relative_file_requires_path_hash_and_size(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match=r"relative_file|both be present"):
        VisualAssetEvidence(storage_kind="relative_file", **overrides)


def test_relative_file_media_type_may_be_null() -> None:
    evidence = VisualAssetEvidence(
        storage_kind="relative_file",
        relative_path="images/unknown.bin",
        sha256="a" * 64,
        byte_size=1,
    )
    assert evidence.media_type is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"relative_path": "images/a.png"},
        {"media_type": "image/png"},
        {"sha256": "a" * 64, "byte_size": 1},
    ],
)
def test_unavailable_visual_forbids_path_and_byte_metadata(
    metadata: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="unavailable"):
        VisualAssetEvidence(storage_kind="unavailable", **metadata)


def test_embedded_visual_forbids_relative_path() -> None:
    with pytest.raises(ValidationError, match="cannot carry relative_path"):
        VisualAssetEvidence(
            storage_kind="embedded_raw",
            relative_path="images/a.png",
            media_type="image/png",
            sha256="a" * 64,
            byte_size=1,
        )


def test_v1_serialization_roundtrip_is_deterministic_utf8_lf() -> None:
    document = _document(_block())
    payload_a = physical_document_v1_to_json(document)
    payload_b = physical_document_v1_to_json(document)
    assert payload_a == payload_b
    assert payload_a.endswith("\n") and "\r" not in payload_a
    assert physical_document_v1_from_json(payload_a) == document
    assert physical_document_v1_to_json(physical_document_v1_from_json(payload_a)) == payload_a


def test_v1_serializer_defensively_rejects_non_finite_numbers() -> None:
    forged_page = PhysicalPageV1.model_construct(
        page_index=0, width=float("nan"), height=1.0, blocks=()
    )
    forged_document = PhysicalDocumentV1.model_construct(
        physical_ir_version=2,
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=1,
        pages=(forged_page,),
    )
    with pytest.raises(PhysicalIRV1SerializationError, match="non-finite"):
        physical_document_v1_to_json(forged_document)


def test_version_dispatch_is_strict_and_does_not_upgrade(tmp_path: Path) -> None:
    marker_raw = _write_marker_raw(tmp_path)
    v0 = MarkerPhysicalNormalizer().normalize(marker_raw)
    v1 = MarkerPhysicalNormalizerV1().normalize(marker_raw)
    parsed_v0 = load_physical_document(physical_document_to_json(v0))
    parsed_v1 = load_physical_document(physical_document_v1_to_json(v1))
    assert isinstance(parsed_v0, PhysicalDocument)
    assert isinstance(parsed_v1, PhysicalDocumentV1)
    with pytest.raises(PhysicalIRV1SerializationError):
        physical_document_v1_from_json(physical_document_to_json(v0))
    with pytest.raises(PhysicalIRSerializationError):
        physical_document_from_json(physical_document_v1_to_json(v1))
    with pytest.raises(PhysicalIRVersionError, match="unsupported"):
        load_physical_document('{"physical_ir_version": 99}')


def test_marker_v1_maps_native_types_table_and_extraction(tmp_path: Path) -> None:
    raw = _write_marker_raw(tmp_path)
    document = MarkerPhysicalNormalizerV1().normalize(raw)
    kinds = [block.kind for block in document.pages[0].blocks]
    assert kinds[0:5] == [
        BlockKindV1.HEADER,
        BlockKindV1.TEXT,
        BlockKindV1.TITLE,
        BlockKindV1.UNKNOWN,
        BlockKindV1.TABLE,
    ]
    table = document.pages[0].blocks[4]
    assert table.provenance.source_raw_type == "Table"
    assert table.table_structure == TableStructure(
        row_count=1,
        column_count=1,
        cells=(
            TableCell(
                row_start=0,
                column_start=0,
                row_span=1,
                column_span=1,
                text="Cell",
                is_header=False,
            ),
        ),
    )
    assert table.text_extraction == TextExtractionEvidence(method="native_text")
    assert table.text_extraction.confidence is None
    assert document.pages[0].blocks[5].kind == BlockKindV1.UNKNOWN


@pytest.mark.parametrize("metadata_state", ["missing", "unknown", "empty"])
def test_marker_disable_ocr_alone_does_not_prove_native_text(
    tmp_path: Path, metadata_state: str
) -> None:
    raw = _write_marker_raw(tmp_path)
    metadata = raw / "source" / "source_meta.json"
    if metadata_state == "missing":
        metadata.unlink()
    else:
        method = "ocr" if metadata_state == "unknown" else ""
        metadata.write_text(
            json.dumps({"page_stats": [{"page_id": 0, "text_extraction_method": method}]}),
            encoding="utf-8",
        )
    nonempty = [
        block for block in MarkerPhysicalNormalizerV1().normalize(raw).pages[0].blocks if block.text
    ]
    assert nonempty
    assert all(block.text_extraction is not None for block in nonempty)
    assert all(block.text_extraction.method == TextExtractionMethod.UNKNOWN for block in nonempty)  # type: ignore[union-attr]


def test_marker_present_malformed_extraction_metadata_fails(tmp_path: Path) -> None:
    raw = _write_marker_raw(tmp_path)
    (raw / "source" / "source_meta.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="extraction metadata"):
        MarkerPhysicalNormalizerV1().normalize(raw)


def test_v1_normalizers_preserve_v0_provenance_conflict_checks(tmp_path: Path) -> None:
    marker_raw = _write_marker_raw(tmp_path)
    with pytest.raises(ValueError, match="provenance conflict"):
        MarkerPhysicalNormalizerV1().normalize(marker_raw, document_id="different-document")
    mineru_raw = _write_mineru_raw(
        tmp_path,
        [{"type": "text", "text": "x", "bbox": [0, 0, 10, 10], "page_idx": 0}],
    )
    with pytest.raises(ValueError, match="provenance conflict"):
        MinerUPhysicalNormalizerV1().normalize(mineru_raw, version_id="v2")


@pytest.mark.parametrize(
    ("raw_type", "expected"), [("Figure", BlockKindV1.FIGURE), ("Picture", BlockKindV1.IMAGE)]
)
def test_marker_visual_identity_and_embedded_asset_hash(
    tmp_path: Path, raw_type: str, expected: BlockKindV1
) -> None:
    image_bytes = b"\xff\xd8\xffretained-jpeg-bytes"

    def transform(tree: dict[str, object]) -> None:
        block = tree["children"][0]["children"][4]  # type: ignore[index]
        block["block_type"] = raw_type
        block["id"] = f"/page/0/{raw_type}/4"
        block["html"] = ""
        block["images"] = {f"/page/0/{raw_type}/4": base64.b64encode(image_bytes).decode("ascii")}

    document = MarkerPhysicalNormalizerV1().normalize(_write_marker_raw(tmp_path, transform))
    block = document.pages[0].blocks[4]
    assert block.kind == expected
    assert block.provenance.source_raw_type == raw_type
    assert block.visual_asset == VisualAssetEvidence(
        storage_kind="embedded_raw",
        media_type="image/jpeg",
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        byte_size=len(image_bytes),
    )


def test_mineru_v1_maps_table_image_unknown_and_keeps_unknown_extraction(tmp_path: Path) -> None:
    items: list[dict[str, object]] = [
        {
            "type": "table",
            "table_body": "<table><tr><th>H</th><td>A</td></tr></table>",
            "bbox": [0, 0, 80, 50],
            "page_idx": 0,
        },
        {
            "type": "image",
            "img_path": "images/asset.jpg",
            "bbox": [0, 60, 80, 100],
            "page_idx": 0,
        },
        {"type": "equation", "text": "x", "bbox": [0, 110, 80, 130], "page_idx": 0},
        {"type": "text", "text": "OCR or native?", "bbox": [0, 140, 80, 160], "page_idx": 0},
    ]
    raw = _write_mineru_raw(tmp_path, items)
    image = raw / "source" / "auto" / "images" / "asset.jpg"
    image.parent.mkdir()
    image_bytes = b"\xff\xd8\xffmineru-image"
    image.write_bytes(image_bytes)
    blocks = MinerUPhysicalNormalizerV1().normalize(raw).pages[0].blocks
    assert [block.kind for block in blocks] == [
        BlockKindV1.TABLE,
        BlockKindV1.IMAGE,
        BlockKindV1.UNKNOWN,
        BlockKindV1.TEXT,
    ]
    assert blocks[0].provenance.source_raw_type == "table"
    assert blocks[0].table_structure is not None
    assert blocks[1].provenance.source_raw_type == "image"
    assert blocks[1].visual_asset == VisualAssetEvidence(
        storage_kind="relative_file",
        relative_path="source/auto/images/asset.jpg",
        media_type="image/jpeg",
        sha256=hashlib.sha256(image_bytes).hexdigest(),
        byte_size=len(image_bytes),
    )
    assert blocks[3].text_extraction == TextExtractionEvidence(method="unknown")
    assert blocks[3].text_extraction.confidence is None
    assert blocks[0].disposition == BlockDisposition.CONTENT
    assert blocks[1].disposition == BlockDisposition.CONTENT
    assert blocks[2].disposition == BlockDisposition.UNKNOWN


def test_mineru_missing_referenced_visual_is_unavailable(tmp_path: Path) -> None:
    raw = _write_mineru_raw(
        tmp_path,
        [
            {
                "type": "image",
                "img_path": "images/not-retained.png",
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }
        ],
    )
    block = MinerUPhysicalNormalizerV1().normalize(raw).pages[0].blocks[0]
    assert block.kind == BlockKindV1.IMAGE
    assert block.disposition == BlockDisposition.CONTENT
    assert block.visual_asset == VisualAssetEvidence(storage_kind="unavailable")


@pytest.mark.parametrize("parser", ["marker", "mineru"])
def test_table_identity_survives_rowspan_structure_rejection(tmp_path: Path, parser: str) -> None:
    invalid_html = "<table><tr><td rowspan='10'>A</td></tr></table>"
    if parser == "marker":

        def transform(tree: dict[str, object]) -> None:
            block = tree["children"][0]["children"][4]  # type: ignore[index]
            block["html"] = invalid_html

        raw = _write_marker_raw(tmp_path, transform)
        block = MarkerPhysicalNormalizerV1().normalize(raw).pages[0].blocks[4]
    else:
        raw = _write_mineru_raw(
            tmp_path,
            [
                {
                    "type": "table",
                    "table_body": invalid_html,
                    "bbox": [0, 0, 100, 100],
                    "page_idx": 0,
                }
            ],
        )
        block = MinerUPhysicalNormalizerV1().normalize(raw).pages[0].blocks[0]
    assert block.kind == BlockKindV1.TABLE
    assert block.table_structure is None


def test_mineru_diagram_annotation_cannot_override_raw_table(tmp_path: Path) -> None:
    raw = _write_mineru_raw(
        tmp_path,
        [
            {
                "type": "table",
                "table_body": "",
                "bbox": [0, 0, 100, 100],
                "page_idx": 0,
            }
        ],
    )
    annotation = tmp_path / "data" / "annotations" / "reference.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text('{"expected_kind": "figure"}', encoding="utf-8")
    block = MinerUPhysicalNormalizerV1().normalize(raw).pages[0].blocks[0]
    assert block.kind == BlockKindV1.TABLE
    assert block.provenance.source_raw_type == "table"


def test_table_identity_survives_without_recoverable_structure(tmp_path: Path) -> None:
    raw = _write_mineru_raw(
        tmp_path,
        [{"type": "table", "bbox": [0, 0, 100, 100], "page_idx": 0}],
    )
    block = MinerUPhysicalNormalizerV1().normalize(raw).pages[0].blocks[0]
    assert block.kind == BlockKindV1.TABLE
    assert block.table_structure is None


@pytest.mark.parametrize("parser", ["marker", "mineru"])
def test_v1_rejected_evidence_writes_do_not_mutate_raw(tmp_path: Path, parser: str) -> None:
    if parser == "marker":
        raw = _write_marker_raw(tmp_path)
        normalizer: object = MarkerPhysicalNormalizerV1()
    else:
        raw = _write_mineru_raw(
            tmp_path,
            [{"type": "text", "text": "x", "bbox": [0, 0, 10, 10], "page_idx": 0}],
        )
        normalizer = MinerUPhysicalNormalizerV1()
    evidence = {path: path.read_bytes() for path in raw.parent.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="inside the raw"):
        normalizer.normalize_to_file(raw, raw / "derived" / "physical_ir_v1.json")  # type: ignore[attr-defined]
    assert {path: path.read_bytes() for path in evidence} == evidence


def test_v1_persistence_is_atomic_lf_and_protects_source(tmp_path: Path) -> None:
    raw = _write_marker_raw(tmp_path)
    normalizer = MarkerPhysicalNormalizerV1()
    output = tmp_path / "derived" / "physical_ir_v1.json"
    document = normalizer.normalize_to_file(raw, output)
    assert output.read_bytes() == physical_document_v1_to_json(document).encode("utf-8")
    assert b"\r\n" not in output.read_bytes()
    source = tmp_path / "source.pdf"
    source.write_bytes(b"authoritative")
    with pytest.raises(ValueError, match="source artifact"):
        normalizer.normalize_to_file(raw, source, source_artifact_path=source)
    assert source.read_bytes() == b"authoritative"


def test_frozen_v0_fixture_hashes_do_not_change(tmp_path: Path) -> None:
    marker_raw = _write_marker_raw(tmp_path)
    marker_payload = physical_document_to_json(MarkerPhysicalNormalizer().normalize(marker_raw))

    mineru_raw = tmp_path / "v0-mineru" / "raw"
    mineru_raw.mkdir(parents=True)
    (mineru_raw / "sample_content_list.json").write_bytes(MINERU_CONTENT_FIXTURE.read_bytes())
    (mineru_raw / "sample_middle.json").write_bytes(MINERU_MIDDLE_FIXTURE.read_bytes())
    mineru_document = MinerUPhysicalNormalizer().normalize(
        mineru_raw,
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=SOURCE_SHA,
        parser="mineru",
        parser_version="3.4.5",
        parser_backend="pipeline",
    )
    mineru_payload = physical_document_to_json(mineru_document)
    assert hashlib.sha256(marker_payload.encode()).hexdigest() == (
        "24c5a315e9eaa5a42c9f958c1e1e870091205c7db6fee496f7d452079094375c"
    )
    assert hashlib.sha256(mineru_payload.encode()).hexdigest() == (
        "9dccdb5baf6a918619fca79645b23d67bd749a85fe84c9cd609d3656a457efd0"
    )


def test_committed_v1_corpus_evidence_and_generated_report_are_consistent() -> None:
    evidence_path = Path("data/benchmarks/physical_ir_v1_validation.v1.json")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["validation_schema_version"] == 2
    assert evidence["validation_protocol_revision"] == 2
    assert evidence["available_pairs"] == 10
    assert evidence["all_v0_hashes_preserved"] is True
    assert evidence["all_v1_equal"] is True
    assert all(entry["equal"] and entry["v0_preserved"] for entry in evidence["entries"])
    aggregate = evidence["aggregate"]
    assert aggregate["table_count"] == 107
    assert aggregate["tables_with_structure"] == 82
    assert aggregate["figure_count"] == 1
    assert aggregate["image_count"] == 5
    assert aggregate["ocr_confidence_count"] == 0
    assert aggregate["visual_asset_hash_count"] == 6
    assert aggregate["verified_visual_asset_count"] == 6
    assert aggregate["rowspan_markup_table_count"] == 39
    assert aggregate["rowspan_explicit_row_violation_count"] == 0
    assert aggregate["v0_to_v1_kind_transitions"] == {
        "unknown->figure": 1,
        "unknown->image": 5,
        "unknown->table": 107,
    }
    anti_leakage = evidence["anti_evaluation_leakage_case"]["observations"]
    assert (
        anti_leakage["marker"]["source_raw_type"],
        anti_leakage["marker"]["normalized_kind"],
    ) == ("Figure", "figure")
    assert (
        anti_leakage["mineru"]["source_raw_type"],
        anti_leakage["mineru"]["normalized_kind"],
    ) == ("table", "table")
    marker = [entry for entry in evidence["entries"] if entry["parser"] == "marker"]
    mineru = [entry for entry in evidence["entries"] if entry["parser"] == "mineru"]
    assert sum(entry["table_count"] for entry in marker) == 55
    assert sum(entry["tables_with_structure"] for entry in marker) == 39
    assert sum(entry["table_count"] for entry in mineru) == 52
    assert sum(entry["tables_with_structure"] for entry in mineru) == 43
    assert aggregate["by_parser"]["marker"]["table_cell_count"] == 3205
    assert aggregate["by_parser"]["mineru"]["table_cell_count"] == 3060
    mineru_dispositions = aggregate["by_parser"]["mineru"]["kind_disposition_matrix"]
    assert mineru_dispositions["table"] == {"content": 52}
    assert mineru_dispositions["image"] == {"content": 1}
    assert aggregate["text_extraction_method_histogram"] == {
        "native_text": 2739,
        "not_recorded": 493,
        "unknown": 1737,
    }
    assert aggregate["unknown_kind_by_parser_source_raw_type"]["marker"] == {
        "Footnote": 40,
        "ListGroup": 46,
        "TableGroup": 2,
        "TableOfContents": 1,
    }
    assert aggregate["unknown_kind_by_parser_source_raw_type"]["mineru"] == {}
    assert aggregate["unknown_text_extraction_by_parser_source_raw_type"]["marker"] == {}
    assert aggregate["unknown_text_extraction_by_parser_source_raw_type"]["mineru"] == {
        "header": 50,
        "page_number": 130,
        "text": 1557,
    }
    report = Path("docs/research/physical-ir-v1-validation.md").read_text(encoding="utf-8")
    assert render_physical_ir_v1_validation(evidence) == report
