"""Parser-neutral evaluation package for physical document extraction."""

from vlm_rag.evaluation.evaluator import (
    DocumentEvaluationReport,
    PageEvaluationReport,
    evaluate_physical_document,
)
from vlm_rag.evaluation.matching import MatchResult, match_page_regions
from vlm_rag.evaluation.metrics import (
    CategoryMetrics,
    OverallMetrics,
    calculate_metrics,
    calculate_pairwise_order_accuracy,
)
from vlm_rag.evaluation.models import (
    AnnotationBoundingBox,
    AuditedPage,
    DocumentAnnotation,
    GroundTruthRegion,
    ReferenceDocumentAnnotation,
    ReferencePage,
    ReferenceRegion,
    RegionKind,
)
from vlm_rag.evaluation.serialization import dump_evaluation_report, load_annotation_file

__all__ = [
    "AnnotationBoundingBox",
    "AuditedPage",
    "CategoryMetrics",
    "DocumentAnnotation",
    "DocumentEvaluationReport",
    "GroundTruthRegion",
    "MatchResult",
    "OverallMetrics",
    "PageEvaluationReport",
    "ReferenceDocumentAnnotation",
    "ReferencePage",
    "ReferenceRegion",
    "RegionKind",
    "calculate_metrics",
    "calculate_pairwise_order_accuracy",
    "dump_evaluation_report",
    "evaluate_physical_document",
    "load_annotation_file",
    "match_page_regions",
]
