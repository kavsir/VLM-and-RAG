"""Deterministic maximum-cardinality bipartite IoU region matching."""

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from vlm_rag.evaluation.models import AnnotationBoundingBox, ReferenceRegion
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
    """A matched reference region and predicted block with their overlap score."""

    reference: ReferenceRegion
    predicted: PhysicalBlock
    iou: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Results of maximum-cardinality bipartite matching on a single page."""

    matched_pairs: tuple[MatchedPair, ...]
    unmatched_reference: tuple[ReferenceRegion, ...]
    unmatched_predicted: tuple[PhysicalBlock, ...]


def _find_max_cardinality_max_iou_matching(
    m: int,
    n: int,
    candidate_edges: list[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    """Solve maximum-cardinality bipartite matching with secondary max-IoU objective.

    Cardinality is computed independently. A min-cost flow of exactly that cardinality then
    maximizes the sum of the exact IEEE-754 IoU values represented by ``Fraction.from_float``.
    Stable graph ordering and path tie-breaking make equal-weight solutions deterministic.
    """
    if m == 0 or n == 0 or not candidate_edges:
        return []

    neighbors: list[list[int]] = [[] for _ in range(m)]
    for u, v, _ in sorted(candidate_edges, key=lambda edge: (edge[0], edge[1])):
        neighbors[u].append(v)
    matched_right = [-1] * n

    def augment_cardinality(u: int, seen: set[int]) -> bool:
        for v in neighbors[u]:
            if v in seen:
                continue
            seen.add(v)
            if matched_right[v] == -1 or augment_cardinality(matched_right[v], seen):
                matched_right[v] = u
                return True
        return False

    cardinality = sum(augment_cardinality(u, set()) for u in range(m))

    @dataclass(slots=True)
    class FlowEdge:
        to: int
        reverse_index: int
        capacity: int
        cost: Fraction

    vertex_count = m + n + 2
    source = 0
    sink = vertex_count - 1
    graph: list[list[FlowEdge]] = [[] for _ in range(vertex_count)]

    def add_edge(frm: int, to: int, capacity: int, cost: Fraction) -> FlowEdge:
        forward = FlowEdge(to, len(graph[to]), capacity, cost)
        reverse = FlowEdge(frm, len(graph[frm]), 0, -cost)
        graph[frm].append(forward)
        graph[to].append(reverse)
        return forward

    for u in range(m):
        add_edge(source, u + 1, 1, Fraction(0))
    for v in range(n):
        add_edge(m + 1 + v, sink, 1, Fraction(0))

    candidate_flow_edges: list[tuple[int, int, float, FlowEdge]] = []
    for u, v, iou in sorted(candidate_edges, key=lambda edge: (edge[0], edge[1])):
        edge = add_edge(u + 1, m + 1 + v, 1, -Fraction.from_float(iou))
        candidate_flow_edges.append((u, v, iou, edge))

    for _ in range(cardinality):
        distances: list[Fraction | None] = [None] * vertex_count
        path_keys: list[tuple[tuple[int, int], ...] | None] = [None] * vertex_count
        parents: list[tuple[int, int] | None] = [None] * vertex_count
        distances[source] = Fraction(0)
        path_keys[source] = ()

        for _relaxation in range(vertex_count - 1):
            changed = False
            for frm, outgoing in enumerate(graph):
                source_distance = distances[frm]
                source_path_key = path_keys[frm]
                if source_distance is None or source_path_key is None:
                    continue
                for edge_index, edge in enumerate(outgoing):
                    if edge.capacity == 0:
                        continue
                    candidate_distance = source_distance + edge.cost
                    candidate_key = (*source_path_key, (frm, edge_index))
                    current_distance = distances[edge.to]
                    current_key = path_keys[edge.to]
                    if (
                        current_distance is None
                        or candidate_distance < current_distance
                        or (
                            candidate_distance == current_distance
                            and (current_key is None or candidate_key < current_key)
                        )
                    ):
                        distances[edge.to] = candidate_distance
                        path_keys[edge.to] = candidate_key
                        parents[edge.to] = (frm, edge_index)
                        changed = True
            if not changed:
                break

        if parents[sink] is None:
            raise RuntimeError("maximum-cardinality matching could not be reconstructed")
        current = sink
        while current != source:
            parent = parents[current]
            if parent is None:
                raise RuntimeError("incomplete augmenting path")
            frm, edge_index = parent
            edge = graph[frm][edge_index]
            edge.capacity -= 1
            graph[edge.to][edge.reverse_index].capacity += 1
            current = frm

    return sorted(
        ((u, v, iou) for u, v, iou, edge in candidate_flow_edges if edge.capacity == 0),
        key=lambda match: (match[0], match[1]),
    )


def match_page_regions(
    reference_regions: Sequence[ReferenceRegion],
    predicted_blocks: Sequence[PhysicalBlock],
    *,
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Deterministically match reference regions with predicted physical blocks."""
    if iou_threshold < 0.0 or iou_threshold > 1.0:
        raise ValueError("iou_threshold must be between 0.0 and 1.0")

    candidates: list[tuple[int, int, float]] = []
    for reference_index, reference in enumerate(reference_regions):
        for pred_idx, pred in enumerate(predicted_blocks):
            score = calculate_iou(reference.bbox, pred.bbox)
            if score >= iou_threshold:
                candidates.append((reference_index, pred_idx, score))

    matched_indices = _find_max_cardinality_max_iou_matching(
        len(reference_regions),
        len(predicted_blocks),
        candidates,
    )

    matched_pairs: list[MatchedPair] = []
    used_reference_indices: set[int] = set()
    used_pred_indices: set[int] = set()

    for u, v, iou in matched_indices:
        reference = reference_regions[u]
        pred = predicted_blocks[v]
        used_reference_indices.add(u)
        used_pred_indices.add(v)
        matched_pairs.append(MatchedPair(reference=reference, predicted=pred, iou=iou))

    matched_pairs.sort(key=lambda mp: (mp.reference.reading_order, mp.reference.id))

    unmatched_reference = tuple(
        reference
        for i, reference in enumerate(reference_regions)
        if i not in used_reference_indices
    )
    unmatched_pred = tuple(
        pred for j, pred in enumerate(predicted_blocks) if j not in used_pred_indices
    )

    return MatchResult(
        matched_pairs=tuple(matched_pairs),
        unmatched_reference=unmatched_reference,
        unmatched_predicted=unmatched_pred,
    )


__all__ = [
    "MatchResult",
    "MatchedPair",
    "calculate_iou",
    "match_page_regions",
]
