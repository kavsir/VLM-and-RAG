"""Unit tests for Physical Document Intermediate Representation (v0) and normalizers."""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError
from vlm_rag.physical_ir import (
    BlockDisposition,
    BlockKind,
    BlockProvenance,
    BoundingBox,
    PhysicalBlock,
    PhysicalDocument,
    PhysicalIRSerializationError,
    PhysicalPage,
    physical_document_from_json,
    physical_document_to_json,
)
from vlm_rag.physical_ir.__main__ import main
from vlm_rag.registry.models import (
    DocumentIdentity,
    DocumentManifest,
    DocumentVersion,
    FileArtifact,
    SourceReference,
)

FIXTURES_DIR = Path("tests/fixtures/mineru")
AUTHORITATIVE_SHA256 = "ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028"
DERIVED_ORIGIN_SHA256 = "42d26544aeae963fbcd08ec71311c6a173de37966864edff4b69b9344f4eb1de"


def _sample_manifest() -> DocumentManifest:
    return DocumentManifest(
        manifest_schema_version=1,
        document=DocumentIdentity(
            id="hanoi-master-plan-100y",
            document_number="2512/QĐ-UBND",
            normalized_document_number="2512/QD-UBND",
            title="Quy hoạch tổng thể",
            document_type="Quyết định",
            issuer="UBND Hà Nội",
        ),
        version=DocumentVersion(
            id="v1",
            document_id="hanoi-master-plan-100y",
            issued_on=date(2026, 5, 13),
            effective_on=date(2026, 5, 13),
        ),
        source=SourceReference(
            source_type="official_government_portal",
            publisher="Cổng TTĐT Hà Nội",
            signer="Vũ Đại Thắng",
            landing_page_url="https://vanban.hanoi.gov.test/doc/2512",
            asset_url="https://datafiles.hanoi.gov.test/pdf/2512.pdf",
            retrieved_at=datetime(2026, 9, 3, 15, 35, 40, tzinfo=UTC),
        ),
        artifact=FileArtifact(
            filename="QD-2512-2026.pdf",
            media_type="application/pdf",
            byte_size=2218758,
            sha256=AUTHORITATIVE_SHA256,
        ),
    )


def _sample_block(
    block_id: str = "doc_v1_p0000_b0000",
    page_index: int = 0,
    reading_order: int = 0,
    kind: BlockKind = BlockKind.TEXT,
    disposition: BlockDisposition = BlockDisposition.CONTENT,
    heading_level: int | None = None,
) -> PhysicalBlock:
    return PhysicalBlock(
        id=block_id,
        page_index=page_index,
        reading_order=reading_order,
        kind=kind,
        disposition=disposition,
        text="Sample paragraph content",
        bbox=BoundingBox(x0=100.0, y0=200.0, x1=500.0, y1=300.0),
        heading_level=heading_level,
        provenance=BlockProvenance(
            parser="mineru",
            parser_version="3.4.5",
            parser_backend="pipeline",
            source_raw_artifact="source_content_list.json",
            source_raw_index=0,
        ),
    )


def test_physical_document_validation() -> None:
    block = _sample_block()
    page = PhysicalPage(page_index=0, width=595.0, height=841.0, blocks=(block,))
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="hanoi-master-plan-100y",
        version_id="v1",
        source_artifact_sha256=AUTHORITATIVE_SHA256,
        parser="mineru",
        parser_version="3.4.5",
        parser_backend="pipeline",
        page_count=1,
        pages=(page,),
    )
    assert doc.physical_ir_version == 1
    assert doc.page_count == 1
    assert doc.pages[0].blocks[0].kind == BlockKind.TEXT


def test_page_validation_non_contiguous_raises() -> None:
    page1 = PhysicalPage(page_index=1, blocks=())
    with pytest.raises(ValidationError, match="pages must be contiguous starting from 0"):
        PhysicalDocument(
            physical_ir_version=1,
            document_id="hanoi-master-plan-100y",
            version_id="v1",
            source_artifact_sha256=AUTHORITATIVE_SHA256,
            parser="mineru",
            parser_version="3.4.5",
            parser_backend="pipeline",
            page_count=1,
            pages=(page1,),
        )


def test_page_count_mismatch_raises() -> None:
    page0 = PhysicalPage(page_index=0, blocks=())
    with pytest.raises(ValidationError, match="page_count"):
        PhysicalDocument(
            physical_ir_version=1,
            document_id="hanoi-master-plan-100y",
            version_id="v1",
            source_artifact_sha256=AUTHORITATIVE_SHA256,
            parser="mineru",
            parser_version="3.4.5",
            parser_backend="pipeline",
            page_count=2,
            pages=(page0,),
        )


def test_bounding_box_validation() -> None:
    box = BoundingBox(x0=10.0, y0=20.0, x1=100.0, y1=200.0)
    assert box.x0 == 10.0
    assert box.y1 == 200.0
    assert box.coordinate_system == "normalized_1000"

    with pytest.raises(ValidationError, match=r"x0 .* cannot exceed x1"):
        BoundingBox(x0=150.0, y0=20.0, x1=100.0, y1=200.0)

    with pytest.raises(ValidationError, match=r"y0 .* cannot exceed y1"):
        BoundingBox(x0=10.0, y0=250.0, x1=100.0, y1=200.0)

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        BoundingBox(x0=-1.0, y0=20.0, x1=100.0, y1=200.0)


def test_unique_block_ids() -> None:
    b1 = _sample_block(block_id="dup_id", page_index=0, reading_order=0)
    b2 = _sample_block(block_id="dup_id", page_index=1, reading_order=0)
    p0 = PhysicalPage(page_index=0, blocks=(b1,))
    p1 = PhysicalPage(page_index=1, blocks=(b2,))
    with pytest.raises(ValidationError, match="duplicate block id"):
        PhysicalDocument(
            physical_ir_version=1,
            document_id="hanoi-master-plan-100y",
            version_id="v1",
            source_artifact_sha256=AUTHORITATIVE_SHA256,
            parser="mineru",
            parser_version="3.4.5",
            parser_backend="pipeline",
            page_count=2,
            pages=(p0, p1),
        )


def test_reading_order_stability() -> None:
    b0 = _sample_block(block_id="b0", page_index=0, reading_order=0)
    b1 = _sample_block(block_id="b1", page_index=0, reading_order=2)  # skipped 1
    with pytest.raises(ValidationError, match=r"reading_order .* must match sequence index"):
        PhysicalPage(page_index=0, blocks=(b0, b1))


def test_heading_level_validation() -> None:
    # TEXT block cannot have heading_level
    with pytest.raises(ValidationError, match="heading_level is only permitted for TITLE blocks"):
        _sample_block(kind=BlockKind.TEXT, heading_level=1)

    # TITLE block with valid heading level
    title_block = _sample_block(
        kind=BlockKind.TITLE,
        disposition=BlockDisposition.CONTENT,
        heading_level=2,
    )
    assert title_block.kind == BlockKind.TITLE
    assert title_block.heading_level == 2


def test_discarded_physical_blocks_remain_represented() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())

    page0_blocks = doc.pages[0].blocks
    header_blocks = [b for b in page0_blocks if b.kind == BlockKind.HEADER]
    page_num_blocks = [b for b in page0_blocks if b.kind == BlockKind.PAGE_NUMBER]

    assert len(header_blocks) == 1
    assert header_blocks[0].disposition == BlockDisposition.DISCARDED
    assert header_blocks[0].text == "ỦY BAN NHÂN DÂN THÀNH PHỐ HÀ NỘI"

    assert len(page_num_blocks) == 1
    assert page_num_blocks[0].disposition == BlockDisposition.DISCARDED
    assert page_num_blocks[0].text == "1"


def test_mineru_text_to_text_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    text_blocks = [b for b in doc.pages[0].blocks if b.kind == BlockKind.TEXT]
    assert len(text_blocks) == 1
    assert text_blocks[0].text == "Số: /QĐ-UBND"
    assert text_blocks[0].disposition == BlockDisposition.CONTENT
    assert text_blocks[0].heading_level is None


def test_mineru_title_to_title_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    title_blocks_p0 = [b for b in doc.pages[0].blocks if b.kind == BlockKind.TITLE]
    assert len(title_blocks_p0) == 1
    assert title_blocks_p0[0].text == "QUYẾT ĐỊNH"
    assert title_blocks_p0[0].heading_level == 1

    title_blocks_p1 = [b for b in doc.pages[1].blocks if b.kind == BlockKind.TITLE]
    assert len(title_blocks_p1) == 1
    assert title_blocks_p1[0].text == "Điều 1. Quy định chung"
    assert title_blocks_p1[0].heading_level == 2


def test_page_idx_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    assert doc.page_count == 2
    assert doc.pages[0].page_index == 0
    assert doc.pages[1].page_index == 1


def test_bbox_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    b0 = doc.pages[0].blocks[0]
    assert b0.bbox.x0 == 184.0
    assert b0.bbox.y0 == 64.0
    assert b0.bbox.x1 == 401.0
    assert b0.bbox.y1 == 102.0
    assert b0.bbox.coordinate_system == "normalized_1000"


def test_text_level_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    p0_title = doc.pages[0].blocks[1]
    p1_title = doc.pages[1].blocks[0]
    assert p0_title.heading_level == 1
    assert p1_title.heading_level == 2


def test_parser_provenance_mapping() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    prov = doc.pages[0].blocks[0].provenance
    assert prov.parser == "mineru"
    assert prov.parser_version == "3.4.5"
    assert prov.parser_backend == "pipeline"
    assert "content_list" in prov.source_raw_artifact
    assert prov.source_raw_index == 0


def test_deterministic_id_generation() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc1 = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    doc2 = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())

    ids1 = [b.id for p in doc1.pages for b in p.blocks]
    ids2 = [b.id for p in doc2.pages for b in p.blocks]
    assert ids1 == ids2
    assert ids1[0] == "hanoi-master-plan-100y_v1_p0000_b0000"
    assert ids1[1] == "hanoi-master-plan-100y_v1_p0000_b0001"


def test_deterministic_serialization() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())

    json1 = physical_document_to_json(doc)
    json2 = physical_document_to_json(doc)
    assert json1 == json2
    assert (
        hashlib.sha256(json1.encode("utf-8")).hexdigest()
        == hashlib.sha256(json2.encode("utf-8")).hexdigest()
    )

    reconstructed = physical_document_from_json(json1)
    assert reconstructed.document_id == doc.document_id
    assert reconstructed.page_count == doc.page_count
    assert len(reconstructed.pages[0].blocks) == len(doc.pages[0].blocks)


def test_malformed_raw_input_failure(tmp_path: Path) -> None:
    normalizer = MinerUPhysicalNormalizer()

    # Empty directory without content list
    with pytest.raises(NormalizationError, match="could not find MinerU content list"):
        normalizer.normalize(tmp_path, manifest=_sample_manifest())

    # Invalid JSON in content list
    bad_file = tmp_path / "source_content_list.json"
    bad_file.write_text("{not-valid-json}", encoding="utf-8")
    with pytest.raises(NormalizationError, match="cannot read MinerU content list"):
        normalizer.normalize(tmp_path, manifest=_sample_manifest())

    # Content list is not an array
    bad_file.write_text('{"type": "text"}', encoding="utf-8")
    with pytest.raises(NormalizationError, match="must be a JSON array"):
        normalizer.normalize(tmp_path, manifest=_sample_manifest())

    # Item with inverted bbox
    bad_file.write_text(
        json.dumps([{"type": "text", "page_idx": 0, "bbox": [500, 100, 100, 200]}]),
        encoding="utf-8",
    )
    with pytest.raises(NormalizationError, match="invalid bounding box"):
        normalizer.normalize(tmp_path, manifest=_sample_manifest())


def test_unsupported_ir_schema_version_failure() -> None:
    normalizer = MinerUPhysicalNormalizer()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=_sample_manifest())
    raw_dict = doc.model_dump(mode="json")
    raw_dict["physical_ir_version"] = 2

    with pytest.raises(PhysicalIRSerializationError, match="physical_ir_version"):
        physical_document_from_json(json.dumps(raw_dict))


def test_authoritative_source_checksum_retained() -> None:
    normalizer = MinerUPhysicalNormalizer()
    manifest = _sample_manifest()
    doc = normalizer.normalize(FIXTURES_DIR, manifest=manifest)
    assert doc.source_artifact_sha256 == AUTHORITATIVE_SHA256
    assert doc.source_artifact_sha256 != DERIVED_ORIGIN_SHA256


def test_no_source_origin_pdf_substitution() -> None:
    # Ensure normalizer rejects or does not default to source_origin.pdf
    normalizer = MinerUPhysicalNormalizer()
    # Omitting manifest and source checksum raises error rather than inventing one
    with pytest.raises(NormalizationError, match="missing source_artifact_sha256"):
        normalizer.normalize(
            FIXTURES_DIR,
            document_id="hanoi-doc",
            version_id="v1",
            source_artifact_sha256=None,
        )


def test_cli_normalize_and_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_file = tmp_path / "test_ir.json"
    manifest_file = tmp_path / "manifest.yaml"
    manifest_file.write_text(
        Path("data/manifests/hanoi_master_plan_100y.v1.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # CLI normalize
    exit_code = main(
        [
            "normalize",
            "--raw-dir",
            str(FIXTURES_DIR),
            "--manifest",
            str(manifest_file),
            "--output",
            str(out_file),
        ]
    )
    assert exit_code == 0
    assert out_file.exists()

    # CLI validate
    exit_code = main(["validate", str(out_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "valid PhysicalDocument" in captured.out
