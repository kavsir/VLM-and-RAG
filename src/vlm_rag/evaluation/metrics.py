"""Evaluation metrics for spatial region detection, classification, and reading order."""

from collections.abc import Sequence

from pydantic import Field

from vlm_rag.evaluation.matching import MatchedPair, MatchResult
from vlm_rag.evaluation.models import EvaluationBaseModel, RegionKind
from vlm_rag.physical_ir.models import BlockKind


class CategoryMetrics(EvaluationBaseModel):
    """Precision, Recall, F1 for a specific layout category."""

    category: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class OverallMetrics(EvaluationBaseModel):
    """Summary metrics across all evaluated pages."""

    total_reference: int
    total_predicted: int
    true_positives: int
    false_positives: int
    false_negatives: int
    spatial_precision: float
    spatial_recall: float
    spatial_f1: float
    mean_iou: float
    pairwise_order_accuracy: float | None = None
    classification_accuracy_on_matched: float = 0.0
    confusion_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    by_category: tuple[CategoryMetrics, ...]


def _map_pred_kind_to_eval_category(kind: BlockKind | str) -> str:
    """Map Physical IR BlockKind to canonical evaluation category."""
    kind_str = kind.value if isinstance(kind, BlockKind) else str(kind)
    if kind_str in (BlockKind.TITLE.value, "title"):
        return "heading"
    if kind_str in (BlockKind.TEXT.value, "text"):
        return "text"
    if kind_str in (BlockKind.HEADER.value, "header"):
        return "header"
    if kind_str in (BlockKind.PAGE_NUMBER.value, "page_number"):
        return "page_number"
    if kind_str in ("table", "figure", "list_item", "footer", "scanned_block"):
        return kind_str
    return "unknown"


def _map_reference_kind_to_eval_category(kind: RegionKind | str) -> str:
    """Map RegionKind to canonical evaluation category."""
    val = kind.value if isinstance(kind, RegionKind) else str(kind)
    if val == "title":
        return "heading"
    return val


def calculate_pairwise_order_accuracy(
    matched_pairs: Sequence[MatchedPair],
) -> float | None:
    """Calculate pairwise reading order accuracy between references and predictions.

    Returns the ratio of concordant comparable pairs to total comparable pairs in [0, 1].
    If fewer than 2 pairs are matched, returns None because ordering cannot be measured.
    Ties in prediction reading order are treated as non-concordant.
    """
    if len(matched_pairs) < 2:
        return None

    # Sort matched pairs by reference reading order
    ordered_pairs = sorted(matched_pairs, key=lambda mp: mp.reference.reading_order)

    total_comparable = 0
    concordant = 0

    for i in range(len(ordered_pairs)):
        for j in range(i + 1, len(ordered_pairs)):
            total_comparable += 1
            pred_i = ordered_pairs[i].predicted.reading_order
            pred_j = ordered_pairs[j].predicted.reading_order
            if pred_i < pred_j:
                concordant += 1

    if total_comparable == 0:
        return None
    return concordant / total_comparable


def calculate_metrics(match_results: Sequence[MatchResult]) -> OverallMetrics:
    """Aggregate matching results across pages into spatial and classification metrics."""
    all_matched: list[MatchedPair] = []
    total_unmatched_reference = 0
    total_unmatched_pred = 0

    for res in match_results:
        all_matched.extend(res.matched_pairs)
        total_unmatched_reference += len(res.unmatched_reference)
        total_unmatched_pred += len(res.unmatched_predicted)

    # Spatial region detection counts
    tp = len(all_matched)
    fp = total_unmatched_pred
    fn = total_unmatched_reference

    spatial_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spatial_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spatial_f1 = (
        (2 * spatial_prec * spatial_rec) / (spatial_prec + spatial_rec)
        if (spatial_prec + spatial_rec) > 0
        else 0.0
    )
    mean_iou = sum(m.iou for m in all_matched) / tp if tp > 0 else 0.0

    # Reading order accuracy
    order_accuracy = calculate_pairwise_order_accuracy(all_matched)

    # Classification metrics & Confusion Matrix over GT union Pred categories
    all_categories: set[str] = set()
    for m in all_matched:
        all_categories.add(_map_reference_kind_to_eval_category(m.reference.kind))
        all_categories.add(_map_pred_kind_to_eval_category(m.predicted.kind))
    for res in match_results:
        for u in res.unmatched_reference:
            all_categories.add(_map_reference_kind_to_eval_category(u.kind))
        for p in res.unmatched_predicted:
            all_categories.add(_map_pred_kind_to_eval_category(p.kind))

    sorted_categories = sorted(all_categories)
    confusion_matrix: dict[str, dict[str, int]] = {
        c1: dict.fromkeys(sorted_categories, 0) for c1 in sorted_categories
    }

    correct_classification_on_matched = 0
    for m in all_matched:
        gt_cat = _map_reference_kind_to_eval_category(m.reference.kind)
        pred_cat = _map_pred_kind_to_eval_category(m.predicted.kind)
        confusion_matrix[gt_cat][pred_cat] += 1
        if gt_cat == pred_cat:
            correct_classification_on_matched += 1

    classification_accuracy_on_matched = correct_classification_on_matched / tp if tp > 0 else 0.0

    by_cat: list[CategoryMetrics] = []
    for cat in sorted_categories:
        cat_tp = confusion_matrix[cat][cat]
        cat_fn = sum(confusion_matrix[cat][other] for other in sorted_categories if other != cat)
        cat_fn += sum(
            1
            for res in match_results
            for u in res.unmatched_reference
            if _map_reference_kind_to_eval_category(u.kind) == cat
        )
        cat_fp = sum(confusion_matrix[other][cat] for other in sorted_categories if other != cat)
        cat_fp += sum(
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
        total_reference=tp + fn,
        total_predicted=tp + fp,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        spatial_precision=round(spatial_prec, 4),
        spatial_recall=round(spatial_rec, 4),
        spatial_f1=round(spatial_f1, 4),
        mean_iou=round(mean_iou, 4),
        pairwise_order_accuracy=(round(order_accuracy, 4) if order_accuracy is not None else None),
        classification_accuracy_on_matched=round(classification_accuracy_on_matched, 4),
        confusion_matrix=confusion_matrix,
        by_category=tuple(by_cat),
    )


__all__ = [
    "CategoryMetrics",
    "OverallMetrics",
    "calculate_metrics",
    "calculate_pairwise_order_accuracy",
]
