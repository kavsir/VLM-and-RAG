"""High-level evaluator running benchmark assessments on PhysicalDocument instances."""

from dataclasses import dataclass
from typing import Any

from vlm_rag.evaluation.matching import match_page_regions
from vlm_rag.evaluation.metrics import OverallMetrics, calculate_metrics
from vlm_rag.evaluation.models import DocumentAnnotation
from vlm_rag.physical_ir.models import PhysicalDocument


@dataclass(frozen=True, slots=True)
class PageEvaluationReport:
    """Evaluation summary for one audited page."""

    page_index: int
    reference_count: int
    predicted_count: int
    matched_count: int
    unmatched_reference: int
    unmatched_predicted: int
    mean_iou: float


@dataclass(frozen=True, slots=True)
class DocumentEvaluationReport:
    """Overall evaluation summary for a document against audited annotations."""

    document_id: str
    parser: str
    parser_version: str
    parser_backend: str
    pages_audited: int
    metrics: OverallMetrics
    page_reports: tuple[PageEvaluationReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert report into a serializable dict."""
        return {
            "document_id": self.document_id,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "parser_backend": self.parser_backend,
            "pages_audited": self.pages_audited,
            "metrics": self.metrics.model_dump(),
            "pages": [
                {
                    "page_index": p.page_index,
                    "reference_count": p.reference_count,
                    "predicted_count": p.predicted_count,
                    "matched_count": p.matched_count,
                    "unmatched_reference": p.unmatched_reference,
                    "unmatched_predicted": p.unmatched_predicted,
                    "mean_iou": p.mean_iou,
                }
                for p in self.page_reports
            ],
        }


def evaluate_physical_document(
    document: PhysicalDocument,
    annotation: DocumentAnnotation,
    *,
    iou_threshold: float = 0.5,
) -> DocumentEvaluationReport:
    """Evaluate a parsed PhysicalDocument against reference annotations."""
    # Cross-artifact integrity validation
    if annotation.document_id != document.document_id:
        raise ValueError(
            f"Document ID mismatch: annotation has '{annotation.document_id}', "
            f"document has '{document.document_id}'"
        )
    if annotation.version_id != document.version_id:
        raise ValueError(
            f"Version ID mismatch: annotation has '{annotation.version_id}', "
            f"document has '{document.version_id}'"
        )
    if annotation.source_sha256 != document.source_artifact_sha256:
        raise ValueError(
            f"Source SHA-256 mismatch: annotation has '{annotation.source_sha256}', "
            f"document has '{document.source_artifact_sha256}'"
        )
    for audited_page in annotation.audited_pages:
        if audited_page.page_index >= document.page_count:
            raise ValueError(
                f"Audited page index {audited_page.page_index} is out of bounds for "
                f"document with {document.page_count} pages (valid 0..{document.page_count - 1})"
            )

    page_lookup = {page.page_index: page for page in document.pages}

    match_results = []
    page_reports = []

    for audited_page in annotation.audited_pages:
        idx = audited_page.page_index
        pred_page = page_lookup.get(idx)
        pred_blocks = pred_page.blocks if pred_page else ()

        res = match_page_regions(audited_page.regions, pred_blocks, iou_threshold=iou_threshold)
        match_results.append(res)

        matched_cnt = len(res.matched_pairs)
        mean_p_iou = sum(m.iou for m in res.matched_pairs) / matched_cnt if matched_cnt > 0 else 0.0

        page_reports.append(
            PageEvaluationReport(
                page_index=idx,
                reference_count=len(audited_page.regions),
                predicted_count=len(pred_blocks),
                matched_count=matched_cnt,
                unmatched_reference=len(res.unmatched_reference),
                unmatched_predicted=len(res.unmatched_predicted),
                mean_iou=round(mean_p_iou, 4),
            )
        )

    overall_metrics = calculate_metrics(match_results)

    return DocumentEvaluationReport(
        document_id=document.document_id,
        parser=document.parser,
        parser_version=document.parser_version,
        parser_backend=document.parser_backend,
        pages_audited=len(annotation.audited_pages),
        metrics=overall_metrics,
        page_reports=tuple(page_reports),
    )


__all__ = [
    "DocumentEvaluationReport",
    "PageEvaluationReport",
    "evaluate_physical_document",
]
