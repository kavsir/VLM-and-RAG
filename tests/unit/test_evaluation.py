"""Unit tests for the layout evaluation and spatial benchmark module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.evaluation.evaluator import evaluate_physical_document
from vlm_rag.evaluation.matching import calculate_iou, match_page_regions
from vlm_rag.evaluation.metrics import calculate_metrics
from vlm_rag.evaluation.models import (
    AnnotationBoundingBox,
    AuditedPage,
    DocumentAnnotation,
    GroundTruthRegion,
    RegionKind,
)
from vlm_rag.evaluation.serialization import dump_evaluation_report, load_annotation_file
from vlm_rag.physical_ir.models import (
    BlockDisposition,
    BlockKind,
    BlockProvenance,
    BoundingBox,
    PhysicalBlock,
    PhysicalDocument,
    PhysicalPage,
)


def _dummy_provenance() -> BlockProvenance:
    return BlockProvenance(
        parser="mineru",
        parser_version="3.4.5",
        parser_backend="pipeline",
        source_raw_artifact="middle.json",
        source_raw_index=0,
    )


def test_calculate_iou_identical_and_disjoint() -> None:
    box1 = AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
    box2 = AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
    assert calculate_iou(box1, box2) == pytest.approx(1.0)

    # Disjoint box
    box_disjoint = AnnotationBoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0)
    assert calculate_iou(box1, box_disjoint) == pytest.approx(0.0)

    # Partial overlap: box1 is 100x100 (area 10000)
    # box3 overlaps by 50x100 = 5000; union = 10000 + 10000 - 5000 = 15000; iou = 5000/15000 = 1/3
    box3 = AnnotationBoundingBox(x0=150.0, y0=100.0, x1=250.0, y1=200.0)
    assert calculate_iou(box1, box3) == pytest.approx(1.0 / 3.0)


def test_bounding_box_validation_invariants() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        AnnotationBoundingBox(x0=200.0, y0=100.0, x1=100.0, y1=200.0)

    with pytest.raises(ValidationError):
        AnnotationBoundingBox(x0=-10.0, y0=0.0, x1=100.0, y1=100.0)

    with pytest.raises(ValidationError):
        AnnotationBoundingBox(x0=0.0, y0=0.0, x1=1050.0, y1=100.0)


def test_audited_page_rejects_duplicate_region_ids() -> None:
    r1 = GroundTruthRegion(
        id="reg-1",
        reading_order=0,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
    )
    r2 = GroundTruthRegion(
        id="reg-1",
        reading_order=1,
        kind=RegionKind.HEADING,
        bbox=AnnotationBoundingBox(x0=0.0, y0=110.0, x1=100.0, y1=200.0),
    )
    with pytest.raises(ValidationError, match="duplicate region id"):
        AuditedPage(page_index=0, regions=(r1, r2))


def test_match_page_regions_threshold_and_tie_breaking() -> None:
    gt1 = GroundTruthRegion(
        id="gt-1",
        reading_order=0,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0),
    )
    pred1 = PhysicalBlock(
        id="blk-1",
        page_index=0,
        reading_order=0,
        kind=BlockKind.TEXT,
        disposition=BlockDisposition.CONTENT,
        text="Sample",
        bbox=BoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0),
        provenance=_dummy_provenance(),
    )
    pred_far = PhysicalBlock(
        id="blk-2",
        page_index=0,
        reading_order=1,
        kind=BlockKind.TEXT,
        disposition=BlockDisposition.CONTENT,
        text="Far away",
        bbox=BoundingBox(x0=800.0, y0=800.0, x1=900.0, y1=900.0),
        provenance=_dummy_provenance(),
    )

    result = match_page_regions([gt1], [pred1, pred_far], iou_threshold=0.5)
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0].ground_truth.id == "gt-1"
    assert result.matched_pairs[0].predicted.id == "blk-1"
    assert len(result.unmatched_ground_truth) == 0
    assert len(result.unmatched_predicted) == 1
    assert result.unmatched_predicted[0].id == "blk-2"


def test_reading_order_concordance_and_metrics() -> None:
    gt_regions = [
        GroundTruthRegion(
            id=f"gt-{i}",
            reading_order=i,
            kind=RegionKind.TEXT,
            bbox=AnnotationBoundingBox(
                x0=100.0, y0=float(i * 100), x1=200.0, y1=float(i * 100 + 50)
            ),
        )
        for i in range(3)
    ]
    # Inverted reading order in predictions for block 1 and 2
    pred_blocks = [
        PhysicalBlock(
            id="blk-0",
            page_index=0,
            reading_order=0,
            kind=BlockKind.TEXT,
            disposition=BlockDisposition.CONTENT,
            text="0",
            bbox=BoundingBox(x0=100.0, y0=0.0, x1=200.0, y1=50.0),
            provenance=_dummy_provenance(),
        ),
        PhysicalBlock(
            id="blk-1",
            page_index=0,
            reading_order=2,
            kind=BlockKind.TEXT,
            disposition=BlockDisposition.CONTENT,
            text="1",
            bbox=BoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=150.0),
            provenance=_dummy_provenance(),
        ),
        PhysicalBlock(
            id="blk-2",
            page_index=0,
            reading_order=1,
            kind=BlockKind.TEXT,
            disposition=BlockDisposition.CONTENT,
            text="2",
            bbox=BoundingBox(x0=100.0, y0=200.0, x1=200.0, y1=250.0),
            provenance=_dummy_provenance(),
        ),
    ]

    res = match_page_regions(gt_regions, pred_blocks, iou_threshold=0.5)
    metrics = calculate_metrics([res])
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(1.0)
    assert metrics.f1 == pytest.approx(1.0)
    # 3 pairs total: (0,1) concordant, (0,2) concordant, (1,2) inverted -> 2/3 concordant
    assert metrics.reading_order_concordance == pytest.approx(2.0 / 3.0, abs=1e-3)


def test_evaluate_physical_document_end_to_end(tmp_path: Path) -> None:
    gt_page = AuditedPage(
        page_index=0,
        phenomena=("legal_hierarchy",),
        regions=(
            GroundTruthRegion(
                id="r-title",
                reading_order=0,
                kind=RegionKind.HEADING,
                bbox=AnnotationBoundingBox(x0=200.0, y0=50.0, x1=800.0, y1=100.0),
                heading_level=1,
            ),
            GroundTruthRegion(
                id="r-article",
                reading_order=1,
                kind=RegionKind.TEXT,
                bbox=AnnotationBoundingBox(x0=100.0, y0=120.0, x1=900.0, y1=300.0),
            ),
        ),
    )
    annotation = DocumentAnnotation(
        document_id="doc-test",
        version_id="v1",
        source_sha256="a" * 64,
        audited_pages=(gt_page,),
    )

    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-test",
        version_id="v1",
        source_artifact_sha256="a" * 64,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="pipeline",
        page_count=1,
        pages=(
            PhysicalPage(
                page_index=0,
                width=1000.0,
                height=1000.0,
                blocks=(
                    PhysicalBlock(
                        id="b-title",
                        page_index=0,
                        reading_order=0,
                        kind=BlockKind.TITLE,
                        disposition=BlockDisposition.CONTENT,
                        text="Title",
                        bbox=BoundingBox(x0=200.0, y0=50.0, x1=800.0, y1=100.0),
                        heading_level=1,
                        provenance=_dummy_provenance(),
                    ),
                    PhysicalBlock(
                        id="b-article",
                        page_index=0,
                        reading_order=1,
                        kind=BlockKind.TEXT,
                        disposition=BlockDisposition.CONTENT,
                        text="Article",
                        bbox=BoundingBox(x0=100.0, y0=120.0, x1=900.0, y1=300.0),
                        provenance=_dummy_provenance(),
                    ),
                ),
            ),
        ),
    )

    report = evaluate_physical_document(doc, annotation)
    assert report.pages_audited == 1
    assert report.metrics.f1 == pytest.approx(1.0)
    assert report.metrics.reading_order_concordance == pytest.approx(1.0)

    # Serialization roundtrip
    out_file = tmp_path / "report.json"
    dump_text = dump_evaluation_report(report.to_dict(), out_file)
    assert out_file.exists()
    assert dump_text.endswith("\n")

    # Annotation file load test
    ann_file = tmp_path / "annotation.json"
    ann_file.write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
    loaded_ann = load_annotation_file(ann_file)
    assert loaded_ann.document_id == "doc-test"
    assert len(loaded_ann.audited_pages) == 1
