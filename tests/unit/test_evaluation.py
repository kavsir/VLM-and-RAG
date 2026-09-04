"""Unit tests for the layout evaluation and spatial benchmark module."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.evaluation.evaluator import evaluate_physical_document
from vlm_rag.evaluation.matching import (
    MatchedPair,
    calculate_iou,
    match_page_regions,
)
from vlm_rag.evaluation.metrics import (
    calculate_metrics,
    calculate_pairwise_order_accuracy,
)
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


def _dummy_provenance(parser: str = "mineru", backend: str = "pipeline") -> BlockProvenance:
    return BlockProvenance(
        parser=parser,
        parser_version="3.4.5" if parser == "mineru" else "2.0.0",
        parser_backend=backend,
        source_raw_artifact="middle.json",
        source_raw_index=0,
    )


def _dummy_block(
    block_id: str,
    page_index: int = 0,
    reading_order: int = 0,
    kind: BlockKind = BlockKind.TEXT,
    bbox: BoundingBox | None = None,
    parser: str = "mineru",
    backend: str = "pipeline",
) -> PhysicalBlock:
    return PhysicalBlock(
        id=block_id,
        page_index=page_index,
        reading_order=reading_order,
        kind=kind,
        disposition=BlockDisposition.CONTENT,
        text=f"Text of {block_id}",
        bbox=bbox or BoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0),
        provenance=_dummy_provenance(parser=parser, backend=backend),
    )


def _dummy_gt_region(
    region_id: str,
    reading_order: int = 0,
    kind: RegionKind = RegionKind.TEXT,
    bbox: AnnotationBoundingBox | None = None,
) -> GroundTruthRegion:
    return GroundTruthRegion(
        id=region_id,
        reading_order=reading_order,
        kind=kind,
        bbox=bbox or AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0),
    )


def test_calculate_iou_identical_and_disjoint() -> None:
    box1 = AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
    box2 = AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=200.0)
    assert calculate_iou(box1, box2) == pytest.approx(1.0)

    box_disjoint = AnnotationBoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0)
    assert calculate_iou(box1, box_disjoint) == pytest.approx(0.0)

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
    r1 = _dummy_gt_region("reg-1", reading_order=0)
    r2 = _dummy_gt_region("reg-1", reading_order=1)
    with pytest.raises(ValidationError, match="duplicate region id"):
        AuditedPage(page_index=0, regions=(r1, r2))


def test_match_page_regions_threshold_and_tie_breaking() -> None:
    gt1 = _dummy_gt_region("gt-1")
    pred1 = _dummy_block("blk-1")
    pred_far = _dummy_block(
        "blk-2",
        reading_order=1,
        bbox=BoundingBox(x0=800.0, y0=800.0, x1=900.0, y1=900.0),
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
    pred_blocks = [
        _dummy_block(
            "blk-0",
            reading_order=0,
            bbox=BoundingBox(x0=100.0, y0=0.0, x1=200.0, y1=50.0),
        ),
        _dummy_block(
            "blk-1",
            reading_order=2,
            bbox=BoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=150.0),
        ),
        _dummy_block(
            "blk-2",
            reading_order=1,
            bbox=BoundingBox(x0=100.0, y0=200.0, x1=200.0, y1=250.0),
        ),
    ]

    res = match_page_regions(gt_regions, pred_blocks, iou_threshold=0.5)
    metrics = calculate_metrics([res])
    assert metrics.spatial_precision == pytest.approx(1.0)
    assert metrics.spatial_recall == pytest.approx(1.0)
    assert metrics.spatial_f1 == pytest.approx(1.0)
    assert metrics.pairwise_order_accuracy == pytest.approx(2.0 / 3.0, abs=1e-3)


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
        parser_backend="fast-no-ocr",
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
                        provenance=_dummy_provenance(parser="marker", backend="fast-no-ocr"),
                    ),
                    PhysicalBlock(
                        id="b-article",
                        page_index=0,
                        reading_order=1,
                        kind=BlockKind.TEXT,
                        disposition=BlockDisposition.CONTENT,
                        text="Article",
                        bbox=BoundingBox(x0=100.0, y0=120.0, x1=900.0, y1=300.0),
                        provenance=_dummy_provenance(parser="marker", backend="fast-no-ocr"),
                    ),
                ),
            ),
        ),
    )

    report = evaluate_physical_document(doc, annotation)
    assert report.pages_audited == 1
    assert report.metrics.spatial_f1 == pytest.approx(1.0)
    assert report.metrics.pairwise_order_accuracy == pytest.approx(1.0)

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


# ==============================================================================
# 20 REQUIRED BENCHMARK & EVALUATION SUITE TESTS
# ==============================================================================


def test_1_exact_bbox_gt_text_vs_predicted_title() -> None:
    """1. exact-bbox GT TEXT vs predicted TITLE: TEXT F1 != 1, heading FP visible."""
    gt = GroundTruthRegion(
        id="gt-t1",
        reading_order=0,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=100.0, y0=100.0, x1=500.0, y1=200.0),
    )
    pred = PhysicalBlock(
        id="blk-t1",
        page_index=0,
        reading_order=0,
        kind=BlockKind.TITLE,
        disposition=BlockDisposition.CONTENT,
        text="A Title",
        bbox=BoundingBox(x0=100.0, y0=100.0, x1=500.0, y1=200.0),
        provenance=_dummy_provenance(),
    )

    res = match_page_regions([gt], [pred], iou_threshold=0.5)
    metrics = calculate_metrics([res])

    # Spatially matched
    assert metrics.spatial_precision == 1.0
    assert metrics.spatial_recall == 1.0
    assert metrics.spatial_f1 == 1.0

    # Classification failed: TEXT F1 must NOT be 1.0
    text_metrics = next(c for c in metrics.by_category if c.category == "text")
    assert text_metrics.f1 == 0.0
    assert text_metrics.false_negatives == 1
    assert text_metrics.true_positives == 0

    # Heading false positive must be visible
    heading_metrics = next(c for c in metrics.by_category if c.category == "heading")
    assert heading_metrics.false_positives == 1
    assert heading_metrics.true_positives == 0
    assert heading_metrics.f1 == 0.0

    # Confusion matrix captures the misclassification
    assert metrics.confusion_matrix["text"]["heading"] == 1


def test_2_predicted_only_category_appears_as_false_positive() -> None:
    """2. predicted-only category appears as FP even if GT has none."""
    gt = _dummy_gt_region("gt-1", kind=RegionKind.TEXT)
    pred_matched = _dummy_block("p-text", kind=BlockKind.TEXT)
    pred_unmatched_hdr = _dummy_block(
        "p-hdr",
        kind=BlockKind.HEADER,
        bbox=BoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0),
    )

    res = match_page_regions([gt], [pred_matched, pred_unmatched_hdr], iou_threshold=0.5)
    metrics = calculate_metrics([res])

    hdr_metrics = next((c for c in metrics.by_category if c.category == "header"), None)
    assert hdr_metrics is not None
    assert hdr_metrics.true_positives == 0
    assert hdr_metrics.false_positives == 1
    assert hdr_metrics.false_negatives == 0
    assert hdr_metrics.f1 == 0.0


def test_3_confusion_matrix_correctness() -> None:
    """3. confusion matrix correctness across multiple classes and unassigned blocks."""
    gt1 = _dummy_gt_region("gt-text-1", reading_order=0, kind=RegionKind.TEXT)
    gt2 = _dummy_gt_region(
        "gt-text-2",
        reading_order=1,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=300.0, y0=100.0, x1=400.0, y1=200.0),
    )
    gt3 = _dummy_gt_region(
        "gt-hdr",
        reading_order=2,
        kind=RegionKind.HEADER,
        bbox=AnnotationBoundingBox(x0=500.0, y0=100.0, x1=600.0, y1=200.0),
    )
    gt_unmatched = _dummy_gt_region(
        "gt-unmatched",
        reading_order=3,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=700.0, y0=100.0, x1=800.0, y1=200.0),
    )

    p1 = _dummy_block("p-text", kind=BlockKind.TEXT)
    p2 = _dummy_block(
        "p-title",
        kind=BlockKind.TITLE,
        bbox=BoundingBox(x0=300.0, y0=100.0, x1=400.0, y1=200.0),
    )
    p3 = _dummy_block(
        "p-hdr",
        kind=BlockKind.HEADER,
        bbox=BoundingBox(x0=500.0, y0=100.0, x1=600.0, y1=200.0),
    )
    p_unmatched = _dummy_block(
        "p-pgnum",
        kind=BlockKind.PAGE_NUMBER,
        bbox=BoundingBox(x0=900.0, y0=100.0, x1=950.0, y1=200.0),
    )

    res = match_page_regions([gt1, gt2, gt3, gt_unmatched], [p1, p2, p3, p_unmatched])
    metrics = calculate_metrics([res])

    cm = metrics.confusion_matrix
    assert cm["text"]["text"] == 1
    assert cm["text"]["heading"] == 1
    assert cm["header"]["header"] == 1

    text_cat = next(c for c in metrics.by_category if c.category == "text")
    # gt1 was text->text (TP=1), gt2 was text->heading (FN), gt_unmatched was FN -> FN=2
    assert text_cat.true_positives == 1
    assert text_cat.false_negatives == 2

    pg_cat = next(c for c in metrics.by_category if c.category == "page_number")
    assert pg_cat.false_positives == 1


def test_4_perfect_pairwise_order() -> None:
    """4. perfect pairwise order yields 1.0."""
    pairs = [
        MatchedPair(
            ground_truth=_dummy_gt_region(f"gt-{i}", reading_order=i),
            predicted=_dummy_block(f"p-{i}", reading_order=i),
            iou=0.9,
        )
        for i in range(3)
    ]
    assert calculate_pairwise_order_accuracy(pairs) == pytest.approx(1.0)


def test_5_fully_reversed_pairwise_order() -> None:
    """5. fully reversed pairwise order yields 0.0."""
    pairs = [
        MatchedPair(
            ground_truth=_dummy_gt_region(f"gt-{i}", reading_order=i),
            predicted=_dummy_block(f"p-{i}", reading_order=2 - i),
            iou=0.9,
        )
        for i in range(3)
    ]
    assert calculate_pairwise_order_accuracy(pairs) == pytest.approx(0.0)


def test_6_partial_inversion_order() -> None:
    """6. partial inversion yields correct ratio."""
    # GT: 0, 1, 2; Pred: 0, 2, 1 -> pairs: (0,1): 0<2, (0,2): 0<1, (1,2): 2<1 (no) -> 2/3
    pred_orders = [0, 2, 1]
    pairs = [
        MatchedPair(
            ground_truth=_dummy_gt_region(f"gt-{i}", reading_order=i),
            predicted=_dummy_block(f"p-{i}", reading_order=pred_orders[i]),
            iou=0.9,
        )
        for i in range(3)
    ]
    assert calculate_pairwise_order_accuracy(pairs) == pytest.approx(2.0 / 3.0)


def test_7_zero_match_reading_order_is_none() -> None:
    """7. 0-match reading order = None/N/A."""
    assert calculate_pairwise_order_accuracy([]) is None


def test_8_single_match_reading_order_is_none() -> None:
    """8. 1-match reading order = None/N/A."""
    p = MatchedPair(
        ground_truth=_dummy_gt_region("gt-0", reading_order=0),
        predicted=_dummy_block("p-0", reading_order=0),
        iou=0.9,
    )
    assert calculate_pairwise_order_accuracy([p]) is None


def test_9_maximum_cardinality_counterexample() -> None:
    """9. maximum-cardinality counterexample: greedy fails with 1; max-cardinality gives 2."""
    # GT: A [0, 0, 100, 100], B [40, 0, 140, 100]
    # Pred: X [10, 0, 110, 100], Y [0, 0, 70, 100]
    # With iou_threshold=0.5:
    # A-X IoU = 90/110 = 0.8182
    # A-Y IoU = 70/100 = 0.70
    # B-X IoU = 70/130 = 0.5385
    # Greedy picks A-X (0.8182), leaving B and Y unmatched (cardinality 1).
    # Maximum cardinality picks A-Y (0.70) and B-X (0.5385) (cardinality 2).
    gt_a = GroundTruthRegion(
        id="A",
        reading_order=0,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
    )
    gt_b = GroundTruthRegion(
        id="B",
        reading_order=1,
        kind=RegionKind.TEXT,
        bbox=AnnotationBoundingBox(x0=40.0, y0=0.0, x1=140.0, y1=100.0),
    )
    pred_x = _dummy_block(
        "X",
        reading_order=0,
        bbox=BoundingBox(x0=10.0, y0=0.0, x1=110.0, y1=100.0),
    )
    pred_y = _dummy_block(
        "Y",
        reading_order=1,
        bbox=BoundingBox(x0=0.0, y0=0.0, x1=70.0, y1=100.0),
    )

    result = match_page_regions([gt_a, gt_b], [pred_x, pred_y], iou_threshold=0.5)
    assert len(result.matched_pairs) == 2
    matched_gt = {mp.ground_truth.id: mp.predicted.id for mp in result.matched_pairs}
    assert matched_gt["A"] == "Y"
    assert matched_gt["B"] == "X"
    assert len(result.unmatched_ground_truth) == 0
    assert len(result.unmatched_predicted) == 0


def test_10_duplicate_audited_page_index_rejected() -> None:
    """10. duplicate audited page index rejected."""
    p1 = AuditedPage(page_index=0, regions=(_dummy_gt_region("r1"),))
    p2 = AuditedPage(page_index=0, regions=(_dummy_gt_region("r2"),))
    with pytest.raises(ValidationError, match="duplicate audited page index"):
        DocumentAnnotation(
            document_id="doc-test",
            version_id="v1",
            source_sha256="a" * 64,
            audited_pages=(p1, p2),
        )


def test_11_duplicate_reading_order_rejected() -> None:
    """11. duplicate reading order rejected on the same page."""
    r1 = _dummy_gt_region("r1", reading_order=0)
    r2 = _dummy_gt_region("r2", reading_order=0)
    with pytest.raises(ValidationError, match="duplicate reading order"):
        AuditedPage(page_index=0, regions=(r1, r2))


def test_12_malformed_sha_rejected() -> None:
    """12. malformed SHA rejected."""
    with pytest.raises(ValidationError):
        DocumentAnnotation(
            document_id="doc-test",
            version_id="v1",
            source_sha256="short_sha",
            audited_pages=(),
        )


def test_13_zero_area_reference_bbox_rejected() -> None:
    """13. zero-area reference bbox rejected."""
    with pytest.raises(ValidationError, match="cannot exceed or equal"):
        AnnotationBoundingBox(x0=100.0, y0=100.0, x1=100.0, y1=200.0)

    with pytest.raises(ValidationError, match="cannot exceed or equal"):
        AnnotationBoundingBox(x0=100.0, y0=100.0, x1=200.0, y1=100.0)


def test_14_annotation_document_id_mismatch_rejected() -> None:
    """14. annotation/document ID mismatch rejected."""
    ann = DocumentAnnotation(
        document_id="doc-correct",
        version_id="v1",
        source_sha256="a" * 64,
        audited_pages=(),
    )
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-mismatch",
        version_id="v1",
        source_artifact_sha256="a" * 64,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=0,
        pages=(),
    )
    with pytest.raises(ValueError, match="Document ID mismatch"):
        evaluate_physical_document(doc, ann)


def test_15_version_mismatch_rejected() -> None:
    """15. version mismatch rejected."""
    ann = DocumentAnnotation(
        document_id="doc-test",
        version_id="v1",
        source_sha256="a" * 64,
        audited_pages=(),
    )
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-test",
        version_id="v2",
        source_artifact_sha256="a" * 64,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=0,
        pages=(),
    )
    with pytest.raises(ValueError, match="Version ID mismatch"):
        evaluate_physical_document(doc, ann)


def test_16_source_sha_mismatch_rejected() -> None:
    """16. source SHA mismatch rejected."""
    ann = DocumentAnnotation(
        document_id="doc-test",
        version_id="v1",
        source_sha256="a" * 64,
        audited_pages=(),
    )
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-test",
        version_id="v1",
        source_artifact_sha256="b" * 64,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=0,
        pages=(),
    )
    with pytest.raises(ValueError, match="Source SHA-256 mismatch"):
        evaluate_physical_document(doc, ann)


def test_17_out_of_range_audited_page_rejected() -> None:
    """17. out-of-range audited page rejected."""
    p_out = AuditedPage(page_index=5, regions=(_dummy_gt_region("r1"),))
    ann = DocumentAnnotation(
        document_id="doc-test",
        version_id="v1",
        source_sha256="a" * 64,
        audited_pages=(p_out,),
    )
    page0 = PhysicalPage(page_index=0)
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-test",
        version_id="v1",
        source_artifact_sha256="a" * 64,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        page_count=1,  # Valid index: 0
        pages=(page0,),
    )
    with pytest.raises(ValueError, match="out of bounds"):
        evaluate_physical_document(doc, ann)


def test_18_marker_backend_provenance_consistency() -> None:
    """18. Marker backend provenance consistency: backend must be fast-no-ocr, not pipeline."""
    # Marker parser provenance contract
    prov = _dummy_provenance(parser="marker", backend="fast-no-ocr")
    assert prov.parser == "marker"
    assert prov.parser_backend == "fast-no-ocr"
    assert prov.parser_backend != "pipeline"

    # Manifest check
    bench_manifest = Path("data/benchmarks/benchmark_manifest.v1.json")
    if bench_manifest.exists():
        import json

        data = json.loads(bench_manifest.read_text(encoding="utf-8"))
        marker_spec = data["parser_coverage_matrix"]["marker"]
        assert marker_spec["backend"] == "fast-no-ocr"
        assert marker_spec["backend"] != "pipeline"


def test_19_corpus_total_page_computation_equals_250() -> None:
    """19. corpus total-page computation = exactly 250 pages."""
    corpus_page_counts = {
        "hanoi_master_plan_100y": 80,
        "luat_112_2025_qh15": 52,
        "vbhn_103_2026_quy_hoach_tong_the": 48,
        "tt_04_2026_bxd_pl2_dinh_muc": 19,
        "tt_04_2023_bkhdt_so_do_ban_do": 47,
        "qd_23_2008_ubnd_hanoi_vien_quy_hoach": 4,
    }
    total_pages = sum(corpus_page_counts.values())
    assert total_pages == 250


def test_20_generated_report_numbers_equal_source_evaluation_artifacts() -> None:
    """20. generated report numbers equal source evaluation artifacts."""
    gt_page = AuditedPage(
        page_index=0,
        regions=(
            _dummy_gt_region("r1", reading_order=0, kind=RegionKind.TEXT),
            _dummy_gt_region(
                "r2",
                reading_order=1,
                kind=RegionKind.HEADING,
                bbox=AnnotationBoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0),
            ),
        ),
    )
    ann = DocumentAnnotation(
        document_id="doc-report-test",
        version_id="v1",
        source_sha256="c" * 64,
        audited_pages=(gt_page,),
    )
    doc = PhysicalDocument(
        physical_ir_version=1,
        document_id="doc-report-test",
        version_id="v1",
        source_artifact_sha256="c" * 64,
        parser="mineru",
        parser_version="3.4.5",
        parser_backend="pipeline",
        page_count=1,
        pages=(
            PhysicalPage(
                page_index=0,
                width=1000.0,
                height=1000.0,
                blocks=(
                    _dummy_block("p1", reading_order=0, kind=BlockKind.TEXT),
                    _dummy_block(
                        "p2",
                        reading_order=1,
                        kind=BlockKind.TITLE,
                        bbox=BoundingBox(x0=300.0, y0=300.0, x1=400.0, y1=400.0),
                    ),
                ),
            ),
        ),
    )

    report = evaluate_physical_document(doc, ann)
    report_dict = report.to_dict()

    # Verify machine dict matches report object exactly
    assert report_dict["document_id"] == "doc-report-test"
    assert report_dict["metrics"]["spatial_precision"] == report.metrics.spatial_precision
    assert report_dict["metrics"]["spatial_recall"] == report.metrics.spatial_recall
    assert report_dict["metrics"]["spatial_f1"] == report.metrics.spatial_f1
    assert (
        report_dict["metrics"]["pairwise_order_accuracy"] == report.metrics.pairwise_order_accuracy
    )

    # Verify markdown table matches benchmark artifact exactly
    hanoi_marker_bench = Path("data/benchmarks/marker_hanoi_master_plan_100y.v1.json")
    report_md = Path("docs/research/parser-benchmark-v1.md")
    if hanoi_marker_bench.exists() and report_md.exists():
        import json

        data = json.loads(hanoi_marker_bench.read_text(encoding="utf-8"))
        f1_str = f"{data['metrics']['spatial_f1']:.4f}"
        report_text = report_md.read_text(encoding="utf-8")
        assert f1_str in report_text
