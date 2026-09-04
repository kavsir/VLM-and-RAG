"""Deterministic maximum-cardinality bipartite IoU region matching."""

from collections import deque
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
    """Results of maximum-cardinality bipartite matching on a single page."""

    matched_pairs: tuple[MatchedPair, ...]
    unmatched_ground_truth: tuple[GroundTruthRegion, ...]
    unmatched_predicted: tuple[PhysicalBlock, ...]


def _find_max_cardinality_max_iou_matching(
    m: int,
    n: int,
    candidate_edges: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Solve maximum-cardinality bipartite matching with secondary max-IoU objective.

    Uses successive shortest augmenting paths (Min-Cost Max-Flow) with large negative base
    cost to strictly prioritize maximum cardinality over individual IoU magnitudes.
    Deterministic tie-breaking is enforced by sorting candidate edges.
    """
    if m == 0 or n == 0 or not candidate_edges:
        return []

    V = m + n + 2
    S = 0
    T = V - 1

    adj: list[list[int]] = [[] for _ in range(V)]
    edges: list[dict[str, int]] = []

    def add_edge(frm: int, to: int, cap: int, cost: int, edge_idx: int = -1) -> None:
        e1 = {
            "frm": frm,
            "to": to,
            "cap": cap,
            "flow": 0,
            "cost": cost,
            "rev": len(edges) + 1,
            "edge_idx": edge_idx,
        }
        e2 = {
            "frm": to,
            "to": frm,
            "cap": 0,
            "flow": 0,
            "cost": -cost,
            "rev": len(edges),
            "edge_idx": -1,
        }
        adj[frm].append(len(edges))
        edges.append(e1)
        adj[to].append(len(edges))
        edges.append(e2)

    for u in range(m):
        add_edge(S, u + 1, 1, 0)
    for v in range(n):
        add_edge(m + 1 + v, T, 1, 0)

    # Sort deterministically: highest IoU first, then lower u, then lower v
    sorted_edges = sorted(candidate_edges, key=lambda x: (-x[2], x[0], x[1]))
    for idx, (u, v, iou) in enumerate(sorted_edges):
        cost = -int(1_000_000 + round(iou * 10_000))
        add_edge(u + 1, m + 1 + v, 1, cost, idx)

    while True:
        dist = [float("inf")] * V
        parent = [-1] * V
        in_queue = [False] * V
        dist[S] = 0.0
        q: deque[int] = deque([S])
        in_queue[S] = True

        while q:
            curr = q.popleft()
            in_queue[curr] = False
            for e_idx in adj[curr]:
                e = edges[e_idx]
                if e["cap"] - e["flow"] > 0 and dist[e["to"]] > dist[curr] + e["cost"]:
                    dist[e["to"]] = dist[curr] + e["cost"]
                    parent[e["to"]] = e_idx
                    if not in_queue[e["to"]]:
                        q.append(e["to"])
                        in_queue[e["to"]] = True

        if dist[T] >= 0:
            break

        curr = T
        while curr != S:
            e_idx = parent[curr]
            edges[e_idx]["flow"] += 1
            edges[edges[e_idx]["rev"]]["flow"] -= 1
            curr = edges[e_idx]["frm"]

    results: list[tuple[int, int, float]] = []
    for e in edges:
        if 1 <= e["frm"] <= m and m + 1 <= e["to"] <= m + n and e["flow"] > 0:
            u = e["frm"] - 1
            v = e["to"] - (m + 1)
            iou = sorted_edges[e["edge_idx"]][2]
            results.append((u, v, iou))

    return results


def match_page_regions(
    ground_truth_regions: Sequence[GroundTruthRegion],
    predicted_blocks: Sequence[PhysicalBlock],
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Deterministically match ground truth regions with predicted physical blocks."""
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("iou_threshold must be between 0.0 and 1.0")

    candidates: list[tuple[int, int, float]] = []
    for gt_idx, gt in enumerate(ground_truth_regions):
        for pred_idx, pred in enumerate(predicted_blocks):
            score = calculate_iou(gt.bbox, pred.bbox)
            if score >= iou_threshold:
                candidates.append((gt_idx, pred_idx, score))

    matched_indices = _find_max_cardinality_max_iou_matching(
        len(ground_truth_regions),
        len(predicted_blocks),
        candidates,
    )

    matched_pairs: list[MatchedPair] = []
    used_gt_indices: set[int] = set()
    used_pred_indices: set[int] = set()

    for u, v, iou in matched_indices:
        gt = ground_truth_regions[u]
        pred = predicted_blocks[v]
        used_gt_indices.add(u)
        used_pred_indices.add(v)
        matched_pairs.append(MatchedPair(ground_truth=gt, predicted=pred, iou=iou))

    matched_pairs.sort(key=lambda mp: (mp.ground_truth.reading_order, mp.ground_truth.id))

    unmatched_gt = tuple(
        gt for i, gt in enumerate(ground_truth_regions) if i not in used_gt_indices
    )
    unmatched_pred = tuple(
        pred for j, pred in enumerate(predicted_blocks) if j not in used_pred_indices
    )

    return MatchResult(
        matched_pairs=tuple(matched_pairs),
        unmatched_ground_truth=unmatched_gt,
        unmatched_predicted=unmatched_pred,
    )


__all__ = [
    "MatchResult",
    "MatchedPair",
    "calculate_iou",
    "match_page_regions",
]
