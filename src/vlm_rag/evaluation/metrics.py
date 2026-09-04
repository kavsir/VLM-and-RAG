"""Evaluation metrics for spatial region detection, classification, and reading order."""

from collections.abc import Sequence

from vlm_rag.evaluation.matching import MatchedPair, MatchResult
from vlm_rag.evaluation.models import EvaluationBaseModel
from vlm_rag.physical_ir.models import BlockKind


class CategoryMetrics(EvaluationBaseModel):
    """Precision, Recall, F1 for a specific block category."""

    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class OverallMetrics(EvaluationBaseModel):
    """Summary metrics across all evaluated pages."""

    total_ground_truth: int
    total_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float
    reading_order_concordance: float
    by_category: tuple[CategoryMetrics, ...]


def _map_pred_kind_to_eval_category(kind: BlockKind) -> str:
    """Map Physical IR BlockKind to canonical evaluation category."""
    if kind == BlockKind.TITLE:
        return "heading"
    if kind == BlockKind.TEXT:
        return "text"
    if kind == BlockKind.HEADER:
        return "header"
    if kind == BlockKind.PAGE_NUMBER:
        return "page_number"
    return "unknown"


def _calculate_reading_order_concordance(matched_pairs: Sequence[MatchedPair]) -> float:
    """Calculate pairwise concordance (normalized Kendall tau) between GT and prediction."""
    if len(matched_pairs) < 2:
        return 1.0

    total_pairs = 0
    concordant_pairs = 0

    for i in range(len(matched_pairs)):
        for j in range(i + 1, len(matched_pairs)):
            total_pairs += 1
            gt_diff = (
                matched_pairs[i].ground_truth.reading_order
                - matched_pairs[j].ground_truth.reading_order
            )
            pred_diff = (
                matched_pairs[i].predicted.reading_order - matched_pairs[j].predicted.reading_order
            )

            if (
                (gt_diff > 0 and pred_diff > 0)
                or (gt_diff < 0 and pred_diff < 0)
                or (gt_diff == 0 and pred_diff == 0)
            ):
                concordant_pairs += 1

    return concordant_pairs / total_pairs if total_pairs > 0 else 1.0


def calculate_metrics(match_results: Sequence[MatchResult]) -> OverallMetrics:
    """Aggregate matching results across pages into complete metrics."""
    all_matched: list[MatchedPair] = []
    total_unmatched_gt = 0
    total_unmatched_pred = 0

    for res in match_results:
        all_matched.extend(res.matched_pairs)
        total_unmatched_gt += len(res.unmatched_ground_truth)
        total_unmatched_pred += len(res.unmatched_predicted)

    tp = len(all_matched)
    fp = total_unmatched_pred
    fn = total_unmatched_gt

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    mean_iou = sum(m.iou for m in all_matched) / tp if tp > 0 else 0.0
    concordance = _calculate_reading_order_concordance(all_matched)

    # Per-category metrics
    # Collect categories present in GT or predictions
    gt_categories = {m.ground_truth.kind.value for m in all_matched}
    for res in match_results:
        for u in res.unmatched_ground_truth:
            gt_categories.add(u.kind.value)

    by_cat: list[CategoryMetrics] = []
    for cat in sorted(gt_categories):
        cat_tp = sum(1 for m in all_matched if m.ground_truth.kind.value == cat)
        cat_fn = sum(
            1 for res in match_results for u in res.unmatched_ground_truth if u.kind.value == cat
        )
        # False positives mapped to this category
        cat_fp = sum(
            1
            for res in match_results
            for p in res.unmatched_predicted
            if _map_pred_kind_to_eval_category(p.kind) == cat
        )
        c_prec = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0.0
        c_rec = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0.0
        c_f1 = (2 * c_prec * c_rec) / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0.0
        by_cat.append(
            CategoryMetrics(
                category=cat,
                true_positives=cat_tp,
                false_positives=cat_fp,
                false_negatives=cat_fn,
                precision=round(c_prec, 4),
                recall=round(c_rec, 4),
                f1=round(c_f1, 4),
            )
        )

    return OverallMetrics(
        total_ground_truth=tp + fn,
        total_predicted=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        reading_order_concordance=round(concordance, 4),
        by_category=tuple(by_cat),
    )
