"""Deterministic bipartite IoU region matching between predictions and annotations."""

from collections.abc import Sequence
from dataclasses import dataclass

from vlm_rag.evaluation.models import AnnotationBoundingBox, GroundTruthRegion
from vlm_rag.physical_ir.models import BoundingBox, PhysicalBlock


def calculate_iou(
    box1: AnnotationBoundingBox | BoundingBox,
    box2: AnnotationBoundingBox | BoundingBox,
) -> float:
    """Calculate Intersection over Union between two normalized_1000 bounding boxes."""
    ix0 = max(box1.x0, box2.x0)
    iy0 = max(box1.y0, box2.y0)
    ix1 = min(box1.x1, box2.x1)
    iy1 = min(box1.y1, box2.y1)

    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    intersection = iw * ih

    area1 = max(0.0, box1.x1 - box1.x0) * max(0.0, box1.y1 - box1.y0)
    area2 = max(0.0, box2.x1 - box2.x0) * max(0.0, box2.y1 - box2.y0)
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


@dataclass(frozen=True, slots=True)
class MatchedPair:
    """A matched ground truth region and predicted block with their overlap score."""

    ground_truth: GroundTruthRegion
    predicted: PhysicalBlock
    iou: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Results of greedy bipartite matching on a single page."""

    matched_pairs: tuple[MatchedPair, ...]
    unmatched_ground_truth: tuple[GroundTruthRegion, ...]
    unmatched_predicted: tuple[PhysicalBlock, ...]


def match_page_regions(
    ground_truth_regions: Sequence[GroundTruthRegion],
    predicted_blocks: Sequence[PhysicalBlock],
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Deterministically match ground truth regions with predicted physical blocks."""
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("iou_threshold must be between 0.0 and 1.0")

    # Compute candidate pairs with iou >= threshold
    candidates: list[tuple[float, str, str, GroundTruthRegion, PhysicalBlock]] = []
    for gt in ground_truth_regions:
        for pred in predicted_blocks:
            score = calculate_iou(gt.bbox, pred.bbox)
            if score >= iou_threshold:
                # Use negative score for descending sort; break ties deterministically by ids
                candidates.append((-score, gt.id, pred.id, gt, pred))

    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    matched: list[MatchedPair] = []
    used_gt_ids: set[str] = set()
    used_pred_ids: set[str] = set()

    for neg_score, _, _, gt, pred in candidates:
        if gt.id in used_gt_ids or pred.id in used_pred_ids:
            continue
        used_gt_ids.add(gt.id)
        used_pred_ids.add(pred.id)
        matched.append(MatchedPair(ground_truth=gt, predicted=pred, iou=-neg_score))

    unmatched_gt = tuple(gt for gt in ground_truth_regions if gt.id not in used_gt_ids)
    unmatched_pred = tuple(pred for pred in predicted_blocks if pred.id not in used_pred_ids)

    return MatchResult(
        matched_pairs=tuple(matched),
        unmatched_ground_truth=unmatched_gt,
        unmatched_predicted=unmatched_pred,
    )
