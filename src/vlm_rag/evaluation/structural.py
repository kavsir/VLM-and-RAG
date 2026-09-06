# ruff: noqa: E501
"""Parser-neutral reference models and evaluation for Structural IR v1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
    model_validator,
)

from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1
from vlm_rag.physical_ir.models import BlockDisposition
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_to_json
from vlm_rag.structural_ir.anchors import physical_text_events
from vlm_rag.structural_ir.models import StructuralDocument, StructuralNode, StructuralNodeKind
from vlm_rag.structural_ir.serialization import (
    structural_document_from_json,
    structural_document_to_json,
)
from vlm_rag.structural_ir.validation import validate_against_physical
from vlm_rag.structural_ir.vietnamese import StructuralDiagnostic, VietnameseStructuralExtractor

if TYPE_CHECKING:
    from vlm_rag.physical_ir.v1 import PhysicalDocumentV1


class StructuralEvaluationModel(BaseModel):
    """Strict immutable configuration shared by structural evaluation records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


_PATH_KIND_NAMES = {
    StructuralNodeKind.PART: "part",
    StructuralNodeKind.CHAPTER: "chapter",
    StructuralNodeKind.SECTION: "section",
    StructuralNodeKind.SUBSECTION: "subsection",
    StructuralNodeKind.ARTICLE: "article",
    StructuralNodeKind.CLAUSE: "clause",
    StructuralNodeKind.POINT: "point",
    StructuralNodeKind.APPENDIX: "appendix",
    StructuralNodeKind.GENERIC_SECTION: "generic",
}
_REFERENCE_ALLOWED_PARENTS: dict[StructuralNodeKind, frozenset[StructuralNodeKind | None]] = {
    StructuralNodeKind.PART: frozenset({None, StructuralNodeKind.APPENDIX}),
    StructuralNodeKind.CHAPTER: frozenset(
        {None, StructuralNodeKind.PART, StructuralNodeKind.APPENDIX}
    ),
    StructuralNodeKind.SECTION: frozenset(
        {None, StructuralNodeKind.CHAPTER, StructuralNodeKind.PART, StructuralNodeKind.APPENDIX}
    ),
    StructuralNodeKind.SUBSECTION: frozenset(
        {
            None,
            StructuralNodeKind.SECTION,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.PART,
            StructuralNodeKind.APPENDIX,
        }
    ),
    StructuralNodeKind.ARTICLE: frozenset(
        {
            None,
            StructuralNodeKind.PART,
            StructuralNodeKind.CHAPTER,
            StructuralNodeKind.SECTION,
            StructuralNodeKind.SUBSECTION,
            StructuralNodeKind.APPENDIX,
        }
    ),
    StructuralNodeKind.CLAUSE: frozenset({StructuralNodeKind.ARTICLE}),
    StructuralNodeKind.POINT: frozenset({StructuralNodeKind.CLAUSE}),
    StructuralNodeKind.APPENDIX: frozenset({None}),
    StructuralNodeKind.GENERIC_SECTION: frozenset(
        {
            None,
            StructuralNodeKind.APPENDIX,
            StructuralNodeKind.ARTICLE,
            StructuralNodeKind.CLAUSE,
            StructuralNodeKind.GENERIC_SECTION,
        }
    ),
}


class StructuralReferenceNode(StructuralEvaluationModel):
    """One PDF-based structural marker, independent from parser block identity."""

    page_index: NonNegativeInt
    pdf_page_number_1_based: int = Field(ge=1)
    kind: StructuralNodeKind
    ordinal_raw: str | None = None
    ordinal_key: str | None = None
    marker_text: str = Field(min_length=1)
    title: str | None = None
    canonical_path: str = Field(min_length=1)
    parent_canonical_path: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=1)
    visual_evidence_note: str = Field(min_length=1)
    scored: bool = True

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return StructuralNodeKind(value)
            except ValueError:
                return value
        return value

    @model_validator(mode="after")
    def validate_reference_node(self) -> Self:
        if self.pdf_page_number_1_based != self.page_index + 1:
            raise ValueError("pdf_page_number_1_based must equal page_index + 1")
        if (self.ordinal_raw is None) != (self.ordinal_key is None):
            raise ValueError("ordinal_raw and ordinal_key must both be present or both be null")
        if "~" in self.canonical_path:
            raise ValueError("reference canonical paths cannot contain occurrence suffixes")
        segment = self.canonical_path.rsplit("/", maxsplit=1)[-1]
        expected_name = _PATH_KIND_NAMES[self.kind]
        match = re.fullmatch(rf"{expected_name}:(?P<value>[^/~]+)", segment)
        if match is None:
            raise ValueError("canonical path segment does not match reference kind")
        if self.ordinal_key is not None and match.group("value") != self.ordinal_key:
            raise ValueError("reference ordinal_key disagrees with canonical path")
        expected_path = (
            segment
            if self.parent_canonical_path == "document"
            else f"{self.parent_canonical_path}/{segment}"
        )
        if self.canonical_path != expected_path:
            raise ValueError("canonical_path must exactly extend parent_canonical_path")
        return self


class StructuralReferencePage(StructuralEvaluationModel):
    """One visually audited PDF page and its ordered reference markers."""

    page_index: NonNegativeInt
    pdf_page_number_1_based: int = Field(ge=1)
    selection_rationale: str = Field(min_length=1)
    nodes: tuple[StructuralReferenceNode, ...] = Field(default_factory=tuple)

    @field_validator("nodes", mode="before")
    @classmethod
    def coerce_nodes(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_page(self) -> Self:
        if self.pdf_page_number_1_based != self.page_index + 1:
            raise ValueError("pdf_page_number_1_based must equal page_index + 1")
        if any(node.scored and node.page_index != self.page_index for node in self.nodes):
            raise ValueError("scored reference node is assigned to the wrong audited page")
        if any(not node.scored and node.page_index > self.page_index for node in self.nodes):
            raise ValueError("context ancestor must begin on or before the audited page")
        return self


class StructuralReferenceAnnotation(StructuralEvaluationModel):
    """Versioned AI visual structural reference annotation for one PDF."""

    annotation_schema_version: Literal[2]
    annotation_version: Literal["v2"]
    annotator: str = Field(min_length=1)
    annotator_type: Literal["ai_visual_audit"]
    annotation_method: Literal["visual_pdf_structural_reaudit"]
    created_at: AwareDatetime
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_physical_ir_exposure: Literal[True]
    prior_structural_extractor_exposure: Literal[True]
    physical_ir_used_as_reference: Literal[False]
    structural_extractor_output_used_as_reference: Literal[False]
    selection_rationale: str = Field(min_length=1)
    audited_pages: tuple[StructuralReferencePage, ...]

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value

    @field_validator("audited_pages", mode="before")
    @classmethod
    def coerce_pages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_annotation(self) -> Self:
        pages = [page.page_index for page in self.audited_pages]
        if pages != sorted(set(pages)):
            raise ValueError("audited page indices must be unique and sorted")
        definitions: dict[str, tuple[StructuralNodeKind, str | None, str | None, str]] = {}
        scored_paths: set[str] = set()
        for page in self.audited_pages:
            for node in page.nodes:
                definition = (
                    node.kind,
                    node.ordinal_raw,
                    node.ordinal_key,
                    node.parent_canonical_path,
                )
                prior = definitions.setdefault(node.canonical_path, definition)
                if prior != definition:
                    raise ValueError("contradictory repeated reference context definition")
                if node.scored:
                    if node.canonical_path in scored_paths:
                        raise ValueError("duplicate scored canonical path")
                    scored_paths.add(node.canonical_path)
        for path, (kind, _, _, parent_path) in definitions.items():
            if parent_path == "document":
                parent_kind = None
            else:
                parent_definition = definitions.get(parent_path)
                if parent_definition is None:
                    raise ValueError(f"phantom parent_canonical_path: {parent_path}")
                parent_kind = parent_definition[0]
            if parent_kind not in _REFERENCE_ALLOWED_PARENTS[kind]:
                parent_name = "document" if parent_kind is None else parent_kind.value
                raise ValueError(f"invalid reference hierarchy: {kind.value} under {parent_name}")
            visited = {path}
            cursor = parent_path
            while cursor != "document":
                if cursor in visited:
                    raise ValueError("cycle in reference hierarchy")
                visited.add(cursor)
                parent = definitions.get(cursor)
                if parent is None:
                    break
                cursor = parent[3]
        return self


def load_structural_annotation(path: Path) -> StructuralReferenceAnnotation:
    """Load one strict structural reference annotation."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"structural annotation must be a JSON object: {path}")
    return StructuralReferenceAnnotation.model_validate(dict(value))


def load_structural_annotations(root: Path) -> dict[str, StructuralReferenceAnnotation]:
    """Load and identity-index every v2 structural reference annotation."""
    directory = root / "data/structural_annotations"
    paths = [
        path
        for path in sorted(directory.glob("*.v2.json"))
        if path.name != "reference_structural_audit.v2.json"
    ]
    annotations = [load_structural_annotation(path) for path in paths]
    result = {annotation.document_id: annotation for annotation in annotations}
    if len(result) != len(annotations):
        raise ValueError("duplicate document_id across structural annotations")
    return result


def _node_page(node: StructuralNode) -> int | None:
    for anchor in node.heading_anchors:
        if anchor.role.value == "marker":
            return int(anchor.page_index)
    return None


def _metric(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else 0.0
        if precision is not None and recall is not None
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _prediction_parent_path(
    node: StructuralNode, predicted_by_id: Mapping[str, StructuralNode]
) -> str:
    parent = predicted_by_id[node.parent_id or ""]
    return parent.canonical_path


def _hierarchy_aware_matches(
    references: list[StructuralReferenceNode],
    predictions: list[StructuralNode],
    predicted_by_id: Mapping[str, StructuralNode],
) -> tuple[
    list[tuple[StructuralReferenceNode, StructuralNode]],
    list[StructuralReferenceNode],
    list[StructuralNode],
    list[dict[str, Any]],
]:
    """Match exact page/kind/ordinal groups; duplicate groups require exact parent context."""
    reference_groups: dict[tuple[int, str, str | None], list[StructuralReferenceNode]] = (
        defaultdict(list)
    )
    prediction_groups: dict[tuple[int, str, str | None], list[StructuralNode]] = defaultdict(list)
    for reference in references:
        reference_groups[
            (reference.page_index, reference.kind.value, reference.ordinal_key)
        ].append(reference)
    for prediction in predictions:
        page = _node_page(prediction)
        if page is None:
            continue
        prediction_groups[(page, prediction.kind.value, prediction.ordinal_key)].append(prediction)

    matches: list[tuple[StructuralReferenceNode, StructuralNode]] = []
    unmatched_references: list[StructuralReferenceNode] = []
    unmatched_predictions: list[StructuralNode] = []
    ambiguity: list[dict[str, Any]] = []
    group_keys = reference_groups.keys() | prediction_groups.keys()
    for key in sorted(group_keys, key=lambda item: (item[0], item[1], item[2] or "")):
        refs = reference_groups.get(key, [])
        preds = prediction_groups.get(key, [])
        if len(refs) == 1 and len(preds) == 1:
            matches.append((refs[0], preds[0]))
            continue
        remaining_refs = list(refs)
        remaining_preds = list(preds)
        progress = True
        while progress:
            progress = False
            candidates: dict[int, list[int]] = {
                ref_index: [
                    pred_index
                    for pred_index, prediction in enumerate(remaining_preds)
                    if _prediction_parent_path(prediction, predicted_by_id)
                    == reference.parent_canonical_path
                ]
                for ref_index, reference in enumerate(remaining_refs)
            }
            inverse: Counter[int] = Counter(
                pred_index for values in candidates.values() for pred_index in values
            )
            unique_pairs = [
                (ref_index, values[0])
                for ref_index, values in candidates.items()
                if len(values) == 1 and inverse[values[0]] == 1
            ]
            if unique_pairs:
                matched_reference_ids = {id(remaining_refs[item[0]]) for item in unique_pairs}
                matched_prediction_ids = {id(remaining_preds[item[1]]) for item in unique_pairs}
                matches.extend(
                    (remaining_refs[ref_index], remaining_preds[pred_index])
                    for ref_index, pred_index in unique_pairs
                )
                remaining_refs = [
                    reference
                    for reference in remaining_refs
                    if id(reference) not in matched_reference_ids
                ]
                remaining_preds = [
                    prediction
                    for prediction in remaining_preds
                    if id(prediction) not in matched_prediction_ids
                ]
                progress = True
        if remaining_refs or remaining_preds:
            unmatched_references.extend(remaining_refs)
            unmatched_predictions.extend(remaining_preds)
            if refs and preds and (len(refs) > 1 or len(preds) > 1):
                ambiguity.append(
                    {
                        "page_index": key[0],
                        "kind": key[1],
                        "ordinal_key": key[2],
                        "reference_candidates": len(remaining_refs),
                        "prediction_candidates": len(remaining_preds),
                        "reason": "ambiguous_duplicate_group_unmatched",
                    }
                )
    return matches, unmatched_references, unmatched_predictions, ambiguity


def _normalized_title(value: str) -> str:
    return " ".join(value.casefold().split())


def evaluate_structural_document(
    document: StructuralDocument,
    annotation: StructuralReferenceAnnotation,
    *,
    parser: str | None = None,
    physical: PhysicalDocumentV1 | None = None,
) -> dict[str, Any]:
    """Evaluate exact marker identity and full parent edges without occurrence shifting."""
    if (document.document_id, document.version_id, document.source_artifact_sha256) != (
        annotation.document_id,
        annotation.version_id,
        annotation.source_sha256,
    ):
        raise ValueError("Structural IR identity does not match reference annotation")
    audited_pages = {page.page_index for page in annotation.audited_pages}
    all_references = [node for page in annotation.audited_pages for node in page.nodes]
    references = [node for node in all_references if node.scored]
    context_paths = {node.canonical_path for node in all_references if not node.scored}
    scored_paths = {node.canonical_path for node in references}
    context_only_paths = context_paths - scored_paths
    raw_predictions = [
        node
        for node in document.nodes[1:]
        if _node_page(node) is not None and _node_page(node) in audited_pages
    ]
    predictions = [
        node for node in raw_predictions if node.canonical_path not in context_only_paths
    ]
    predicted_by_id = {node.id: node for node in document.nodes}
    matches, unmatched_references, unmatched_predictions, matching_ambiguity = (
        _hierarchy_aware_matches(references, predictions, predicted_by_id)
    )

    per_kind: dict[str, dict[str, int | float | None]] = {}
    for kind in StructuralNodeKind:
        if kind == StructuralNodeKind.DOCUMENT:
            continue
        tp = sum(reference.kind == kind for reference, _ in matches)
        fp = sum(node.kind == kind for node in unmatched_predictions)
        fn = sum(node.kind == kind for node in unmatched_references)
        per_kind[kind.value] = _metric(
            tp,
            fp,
            fn,
        )

    node_metric = _metric(len(matches), len(unmatched_predictions), len(unmatched_references))
    wrong_parent = 0
    correct_edges = 0
    path_exact = 0
    title_total = 0
    title_exact = 0
    for reference, prediction in matches:
        if prediction.canonical_path == reference.canonical_path:
            path_exact += 1
        if reference.title is not None:
            title_total += 1
            if prediction.title is not None and _normalized_title(
                prediction.title
            ) == _normalized_title(reference.title):
                title_exact += 1
        if _prediction_parent_path(prediction, predicted_by_id) == reference.parent_canonical_path:
            correct_edges += 1
        else:
            wrong_parent += 1
    edge_metric = _metric(
        correct_edges,
        len(unmatched_predictions) + wrong_parent,
        len(unmatched_references) + wrong_parent,
    )

    block_kinds = (
        {block.id: block.kind.value for page in physical.pages for block in page.blocks}
        if physical is not None
        else {}
    )
    false_positives = []
    for prediction in unmatched_predictions:
        anchor = prediction.heading_anchors[0]
        excerpt = " ".join(
            part for part in (prediction.marker_text, prediction.title) if part is not None
        )
        false_positives.append(
            {
                "document_id": document.document_id,
                "parser": parser,
                "page_index": anchor.page_index,
                "pdf_page_number_1_based": anchor.page_index + 1,
                "kind": prediction.kind.value,
                "ordinal_raw": prediction.ordinal_raw,
                "ordinal_key": prediction.ordinal_key,
                "canonical_path": prediction.canonical_path,
                "source_excerpt": excerpt[:240],
                "physical_block_kind": block_kinds.get(anchor.block_id),
                "reason": "unmatched_exact_structural_identity",
            }
        )
    false_negatives = [
        {
            "document_id": document.document_id,
            "parser": parser,
            "page_index": reference.page_index,
            "pdf_page_number_1_based": reference.pdf_page_number_1_based,
            "kind": reference.kind.value,
            "ordinal_raw": reference.ordinal_raw,
            "ordinal_key": reference.ordinal_key,
            "canonical_path": reference.canonical_path,
            "reference_excerpt": reference.source_excerpt,
            "reason": "unresolved",
        }
        for reference in unmatched_references
    ]
    return {
        "document_id": document.document_id,
        "parser": parser,
        "audited_pages": sorted(audited_pages),
        "reference_scored_nodes": len(references),
        "reference_context_nodes": sum(not node.scored for node in all_references),
        "predictions_in_scope": len(predictions),
        "node_metrics": node_metric,
        "node_metrics_by_kind": per_kind,
        "parent_edge_metrics": edge_metric,
        "canonical_path_exact": {
            "matched_nodes": len(matches),
            "exact": path_exact,
            "rate": path_exact / len(matches) if matches else None,
        },
        "title_normalized_exact": {
            "eligible": title_total,
            "exact": title_exact,
            "rate": title_exact / title_total if title_total else None,
        },
        "matching_diagnostics": matching_ambiguity,
        "false_positive_trace": false_positives,
        "false_negative_trace": false_negatives,
    }


def _sum_metrics(results: Iterable[dict[str, Any]], field: str) -> dict[str, int | float | None]:
    values = [result[field] for result in results]
    return _metric(
        sum(int(value["tp"]) for value in values),
        sum(int(value["fp"]) for value in values),
        sum(int(value["fn"]) for value in values),
    )


def _aggregate_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [kind for kind in StructuralNodeKind if kind != StructuralNodeKind.DOCUMENT]
    by_kind: dict[str, dict[str, int | float | None]] = {}
    for kind in kinds:
        metrics = [result["node_metrics_by_kind"][kind.value] for result in results]
        by_kind[kind.value] = _metric(
            sum(int(value["tp"]) for value in metrics),
            sum(int(value["fp"]) for value in metrics),
            sum(int(value["fn"]) for value in metrics),
        )
    matched = sum(int(result["canonical_path_exact"]["matched_nodes"]) for result in results)
    path_exact = sum(int(result["canonical_path_exact"]["exact"]) for result in results)
    title_eligible = sum(int(result["title_normalized_exact"]["eligible"]) for result in results)
    title_exact = sum(int(result["title_normalized_exact"]["exact"]) for result in results)
    return {
        "node_metrics": _sum_metrics(results, "node_metrics"),
        "node_metrics_by_kind": by_kind,
        "parent_edge_metrics": _sum_metrics(results, "parent_edge_metrics"),
        "canonical_path_exact": {
            "matched_nodes": matched,
            "exact": path_exact,
            "rate": path_exact / matched if matched else None,
        },
        "title_normalized_exact": {
            "eligible": title_eligible,
            "exact": title_exact,
            "rate": title_exact / title_eligible if title_eligible else None,
        },
    }


def structural_set_jaccard(left: set[tuple[str, str]], right: set[tuple[str, str]]) -> float | None:
    """Return set Jaccard, or null when neither representation has an item."""
    union = left | right
    return len(left & right) / len(union) if union else None


def _cross_parser_agreement(
    marker: StructuralDocument, mineru: StructuralDocument
) -> dict[str, Any]:
    def keys(document: StructuralDocument, kinds: set[StructuralNodeKind]) -> set[tuple[str, str]]:
        return {
            (node.kind.value, node.canonical_path)
            for node in document.nodes[1:]
            if node.kind in kinds
        }

    legal_kinds = {
        StructuralNodeKind.PART,
        StructuralNodeKind.CHAPTER,
        StructuralNodeKind.SECTION,
        StructuralNodeKind.SUBSECTION,
        StructuralNodeKind.ARTICLE,
        StructuralNodeKind.CLAUSE,
        StructuralNodeKind.POINT,
        StructuralNodeKind.APPENDIX,
    }
    result: dict[str, Any] = {"document_id": marker.document_id}
    for label, kinds in (
        ("legal", legal_kinds),
        ("generic", {StructuralNodeKind.GENERIC_SECTION}),
        ("all", legal_kinds | {StructuralNodeKind.GENERIC_SECTION}),
    ):
        left = keys(marker, kinds)
        right = keys(mineru, kinds)
        union = left | right
        result[label] = {
            "marker": len(left),
            "mineru": len(right),
            "intersection": len(left & right),
            "union": len(union),
            "jaccard": structural_set_jaccard(left, right),
        }
    return result


_AMBIGUOUS_SIMPLE = re.compile(r"^\s*\d+\.\s+")
_AMBIGUOUS_LETTER = re.compile(r"^\s*[a-zđ]\)\s+", re.IGNORECASE)
_OCR_MARKER = re.compile(r"^\s*(?:chu[oơ]ng|[dđ]i[eề]u|ph[uụ]\s+l[uụ]c)\b", re.IGNORECASE)


def _ambiguous_candidates(
    physical: PhysicalDocumentV1, structural: StructuralDocument, parser: str
) -> dict[str, Any]:
    marker_spans = {
        (anchor.block_id, anchor.char_start, anchor.char_end)
        for node in structural.nodes
        for anchor in node.heading_anchors
        if anchor.role.value == "marker"
    }
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in physical_text_events(physical):
        span = (event.block_id, event.char_start, event.char_end)
        if any(
            block_id == span[0] and start is not None and span[1] <= start < span[2]
            for block_id, start, _ in marker_spans
        ):
            continue
        category: str | None = None
        if _AMBIGUOUS_LETTER.match(event.text):
            category = "letter_item_outside_legal_clause"
        elif _AMBIGUOUS_SIMPLE.match(event.text):
            category = "simple_decimal_without_heading_evidence"
        elif _OCR_MARKER.match(event.text):
            category = "marker_like_text_rejected"
        if category is not None:
            counts[category] += 1
            if len(examples[category]) < 3:
                examples[category].append(
                    {
                        "document_id": physical.document_id,
                        "parser": parser,
                        "page_index": event.page_index,
                        "excerpt": " ".join(event.text.split())[:160],
                    }
                )
    return {"counts": dict(sorted(counts.items())), "examples": dict(sorted(examples.items()))}


def _normalize_input(parser: str, raw_directory: Path) -> PhysicalDocumentV1:
    if parser == "marker":
        return MarkerPhysicalNormalizerV1().normalize(raw_directory)
    if parser == "mineru":
        return MinerUPhysicalNormalizerV1().normalize(raw_directory)
    raise ValueError(f"unsupported retained parser {parser!r}")


def _read_json_object(path: Path) -> dict[str, Any]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _output_relative(document_id: str, parser: str) -> Path:
    return Path("data/structural_ir") / document_id.replace("-", "_") / f"{parser}.v1.json"


def collect_structural_ir_v1_validation(root: Path) -> dict[str, Any]:
    """Extract all retained pairs twice, persist outputs, and collect benchmark evidence."""
    source = _read_json_object(root / "data/benchmarks/physical_ir_v1_validation.v1.json")
    entries = source.get("entries")
    if not isinstance(entries, list):
        raise ValueError("physical validation evidence has no entries")
    annotations = load_structural_annotations(root)
    reference_audit_relative = Path(
        "data/structural_annotations/reference_structural_audit.v2.json"
    )
    reference_audit_bytes = (root / reference_audit_relative).read_bytes()
    reference_audit = _read_json_object(root / reference_audit_relative)
    if reference_audit.get("page_count") != sum(
        len(annotation.audited_pages) for annotation in annotations.values()
    ):
        raise ValueError("reference audit page count does not match v2 annotations")
    output_entries: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    documents: dict[tuple[str, str], StructuralDocument] = {}
    aggregate_kinds: Counter[str] = Counter()
    aggregate_depth: Counter[int] = Counter()
    ambiguous_counts: Counter[str] = Counter()
    ambiguous_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    rejection_records: list[dict[str, Any]] = []
    total_text_blocks = 0
    total_objects = 0
    total_pages = 0
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("invalid physical validation entry")
        parser = str(raw_entry["parser"])
        raw_directory = root / str(raw_entry["raw_directory"])
        physical = _normalize_input(parser, raw_directory)
        physical_bytes = physical_document_v1_to_json(physical).encode("utf-8")
        expected_physical_sha = str(raw_entry["sha256_a"])
        if _sha(physical_bytes) != expected_physical_sha:
            raise ValueError(f"Physical IR hash changed for {physical.document_id}/{parser}")
        diagnostics_a: list[StructuralDiagnostic] = []
        diagnostics_b: list[StructuralDiagnostic] = []
        result_a = VietnameseStructuralExtractor().extract(physical, diagnostics=diagnostics_a)
        result_b = VietnameseStructuralExtractor().extract(physical, diagnostics=diagnostics_b)
        validate_against_physical(result_a, physical)
        payload_a = structural_document_to_json(result_a).encode("utf-8")
        payload_b = structural_document_to_json(result_b).encode("utf-8")
        if payload_a != payload_b or diagnostics_a != diagnostics_b:
            raise ValueError(
                f"Structural IR is non-deterministic for {physical.document_id}/{parser}"
            )
        output_relative = _output_relative(physical.document_id, parser)
        output_path = root / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(payload_a)
        kinds = Counter(node.kind.value for node in result_a.nodes)
        depth = Counter(int(node.depth) for node in result_a.nodes)
        text_blocks = sum(
            block.disposition == BlockDisposition.CONTENT and bool(block.text)
            for page in physical.pages
            for block in page.blocks
        )
        objects = sum(
            block.disposition == BlockDisposition.CONTENT and not block.text
            for page in physical.pages
            for block in page.blocks
        )
        ambiguity = _ambiguous_candidates(physical, result_a, parser)
        ambiguous_counts.update(ambiguity["counts"])
        for category, examples in ambiguity["examples"].items():
            ambiguous_examples[category].extend(examples)
            ambiguous_examples[category] = ambiguous_examples[category][:5]
        annotation = annotations.get(physical.document_id)
        if annotation is None:
            raise ValueError(f"no structural reference annotation for {physical.document_id}")
        pair_rejections = [
            {"document_id": physical.document_id, "parser": parser, **asdict(item)}
            for item in diagnostics_a
        ]
        rejection_records.extend(pair_rejections)
        rejection_counts.update(item.category for item in diagnostics_a)
        evaluation = evaluate_structural_document(
            result_a, annotation, parser=parser, physical=physical
        )
        evaluations.append(evaluation)
        output_entries.append(
            {
                "document_id": physical.document_id,
                "version_id": physical.version_id,
                "parser": parser,
                "source_artifact_sha256": physical.source_artifact_sha256,
                "input_physical_ir_sha256": expected_physical_sha,
                "output_path": output_relative.as_posix(),
                "pages": physical.page_count,
                "node_count": len(result_a.nodes),
                "node_kind_histogram": dict(sorted(kinds.items())),
                "legal_node_count": sum(
                    count
                    for kind, count in kinds.items()
                    if kind not in {"document", "generic_section"}
                ),
                "generic_node_count": kinds["generic_section"],
                "depth_histogram": {str(key): value for key, value in sorted(depth.items())},
                "content_text_blocks": text_blocks,
                "content_object_blocks": objects,
                "exact_text_partition": True,
                "single_object_ownership": True,
                "byte_size_a": len(payload_a),
                "sha256_a": _sha(payload_a),
                "byte_size_b": len(payload_b),
                "sha256_b": _sha(payload_b),
                "equal": payload_a == payload_b,
                "ambiguous_candidates": ambiguity,
                "rejection_diagnostics": pair_rejections,
            }
        )
        documents[(physical.document_id, parser)] = result_a
        aggregate_kinds.update(kinds)
        aggregate_depth.update(depth)
        total_text_blocks += text_blocks
        total_objects += objects
        total_pages += physical.page_count

    agreements = []
    for document_id in sorted({key[0] for key in documents}):
        marker = documents.get((document_id, "marker"))
        mineru = documents.get((document_id, "mineru"))
        if marker is not None and mineru is not None:
            agreements.append(_cross_parser_agreement(marker, mineru))
    reference_kind_counts = Counter(
        node.kind.value
        for annotation in annotations.values()
        for page in annotation.audited_pages
        for node in page.nodes
        if node.scored
    )
    marker_evaluations = [item for item in evaluations if item["parser"] == "marker"]
    mineru_evaluations = [item for item in evaluations if item["parser"] == "mineru"]
    false_positives = [trace for result in evaluations for trace in result["false_positive_trace"]]
    false_negatives = [trace for result in evaluations for trace in result["false_negative_trace"]]
    matching_ambiguities = [
        {"document_id": result["document_id"], "parser": result["parser"], **item}
        for result in evaluations
        for item in result["matching_diagnostics"]
    ]
    unique_scored_nodes = sum(reference_kind_counts.values())
    parser_evaluation_instances = sum(
        int(result["reference_scored_nodes"]) for result in evaluations
    )
    toc_records = [
        item for item in rejection_records if item["category"] == "table_of_contents_entry_rejected"
    ]
    prose_records = [
        item
        for item in rejection_records
        if item["category"] == "line_start_prose_reference_rejected"
    ]
    return {
        "validation_schema_version": 2,
        "validation_protocol": "structural_ir_v1_retained_corpus_reference_v2",
        "structural_ir_version": 1,
        "profile": "vi_legal_planning_v1",
        "reference_annotation_schema_version": 2,
        "reference_annotation_version": "v2",
        "extractor_inputs": "PhysicalDocumentV1 only; no raw parser fields or reference labels",
        "matching_algorithm": (
            "Candidates require exact audited page, kind, and ordinal_key. Singleton groups match "
            "directly. Multiplicity groups match only uniquely resolvable exact parent canonical paths; "
            "remaining ambiguity is left unmatched. Matching is one-to-one and uses no occurrence shift, "
            "fuzzy text, geometry, parser IDs, or hidden reference labels. Predictions outside audited "
            "pages and context-only ancestor paths are excluded."
        ),
        "metric_formulas": {
            "node_precision": "TP / (TP + FP)",
            "node_recall": "TP / (TP + FN)",
            "node_f1": "2 * precision * recall / (precision + recall)",
            "undefined_metric": (
                "null when its denominator is zero; F1 is null when precision or recall is undefined"
            ),
            "parent_edge": (
                "full in-scope edge detection including document-root edges: matched child with correct "
                "parent is TP; unmatched prediction is FP; unmatched reference is FN; matched child with "
                "wrong parent contributes one FP and one FN"
            ),
            "canonical_path_exact_rate": "exact full paths / matched scored nodes",
            "title_normalized_exact_rate": (
                "casefolded whitespace-collapsed exact titles / matched nodes with reference title"
            ),
            "cross_parser_jaccard": (
                "intersection / union of corrected canonical path identities; null for an empty union"
            ),
        },
        "reference_scope": {
            "documents": len(annotations),
            "pages": sum(len(value.audited_pages) for value in annotations.values()),
            "pages_by_document": {
                key: len(value.audited_pages) for key, value in sorted(annotations.items())
            },
            "scored_nodes": sum(reference_kind_counts.values()),
            "context_nodes": sum(
                not node.scored
                for annotation in annotations.values()
                for page in annotation.audited_pages
                for node in page.nodes
            ),
            "unique_scored_nodes": unique_scored_nodes,
            "parser_evaluation_instances": parser_evaluation_instances,
            "scored_nodes_by_kind": dict(sorted(reference_kind_counts.items())),
        },
        "reference_reaudit": {
            "artifact_path": reference_audit_relative.as_posix(),
            "sha256": _sha(reference_audit_bytes),
            "page_records": reference_audit["page_count"],
            **dict(reference_audit["summary"]),
        },
        "available_pairs": len(output_entries),
        "all_deterministic": all(bool(entry["equal"]) for entry in output_entries),
        "entries": output_entries,
        "aggregate": {
            "pages": total_pages,
            "nodes": sum(int(entry["node_count"]) for entry in output_entries),
            "node_kind_histogram": dict(sorted(aggregate_kinds.items())),
            "legal_node_count": sum(
                count
                for kind, count in aggregate_kinds.items()
                if kind not in {"document", "generic_section"}
            ),
            "generic_node_count": aggregate_kinds["generic_section"],
            "depth_histogram": {str(key): value for key, value in sorted(aggregate_depth.items())},
            "content_text_blocks": total_text_blocks,
            "content_object_blocks": total_objects,
            "exact_text_partition_pairs": len(output_entries),
            "single_object_ownership_pairs": len(output_entries),
            "output_byte_size": sum(int(entry["byte_size_a"]) for entry in output_entries),
        },
        "benchmark": {
            "pair_results": evaluations,
            "by_parser": {
                "marker": {
                    "documents": len(marker_evaluations),
                    **_aggregate_evaluation(marker_evaluations),
                },
                "mineru": {
                    "documents": len(mineru_evaluations),
                    **_aggregate_evaluation(mineru_evaluations),
                },
            },
            "parser_representation_weighted_aggregate": _aggregate_evaluation(evaluations),
        },
        "cross_parser_agreement": agreements,
        "unresolved_candidate_diagnostics": {
            "counts": dict(sorted(ambiguous_counts.items())),
            "examples": dict(sorted(ambiguous_examples.items())),
        },
        "rejection_diagnostics": {
            "counts": dict(sorted(rejection_counts.items())),
            "records": rejection_records,
        },
        "toc_corpus_audit": {
            "rejected_entries": len(toc_records),
            "parser_page_instances": [
                {
                    "document_id": document_id,
                    "parser": parser,
                    "page_index": page_index,
                }
                for document_id, parser, page_index in sorted(
                    {
                        (item["document_id"], item["parser"], item["page_index"])
                        for item in toc_records
                    }
                )
            ],
        },
        "prose_false_positive_audit": {
            "rejected_candidates": len(prose_records),
            "records": prose_records,
        },
        "stale_context_audit": {
            "duplicate_structural_keys_rejected_to_body": rejection_counts[
                "duplicate_structural_key_rejected"
            ],
            "letter_items_outside_legal_clause_left_as_body": ambiguous_counts[
                "letter_item_outside_legal_clause"
            ],
        },
        "matching_ambiguities": matching_ambiguities,
        "false_positive_trace": false_positives,
        "false_negative_trace": false_negatives,
        "canonical_path_correction": {
            "reviewed_head_suffix_paths_found": 67,
            "reviewer_reported_suffix_paths": 86,
            "reviewed_head_classification": {
                "toc_or_repeated_outline": 21,
                "line_start_prose_reference": 1,
                "stale_or_repeated_legal_list_context": 45,
            },
            "regenerated_suffix_paths": sum(
                "~" in node.canonical_path
                for document in documents.values()
                for node in document.nodes
            ),
        },
        "scope_boundary": (
            "Deterministic structural marker detection and physical anchoring only; no semantic legal "
            "edges, VLM/LLM, retrieval, RAG, or knowledge graph functionality."
        ),
    }


def verify_committed_structural_evidence(root: Path, evidence: dict[str, Any]) -> None:
    """Re-evaluate committed outputs and annotations without raw parser evidence."""
    annotations = load_structural_annotations(root)
    audit = evidence["reference_reaudit"]
    audit_payload = (root / audit["artifact_path"]).read_bytes()
    if _sha(audit_payload) != audit["sha256"]:
        raise ValueError("reference structural audit artifact changed")
    evaluations: list[dict[str, Any]] = []
    for entry in evidence["entries"]:
        path = root / entry["output_path"]
        payload = path.read_bytes()
        if len(payload) != entry["byte_size_a"] or _sha(payload) != entry["sha256_a"]:
            raise ValueError(f"committed Structural IR artifact changed: {path}")
        document = structural_document_from_json(payload.decode("utf-8"))
        result = evaluate_structural_document(
            document, annotations[document.document_id], parser=entry["parser"]
        )
        evaluations.append(result)
    if (
        _aggregate_evaluation(evaluations)
        != evidence["benchmark"]["parser_representation_weighted_aggregate"]
    ):
        raise ValueError("offline structural benchmark metrics differ from committed evidence")
    for parser in ("marker", "mineru"):
        expected = dict(evidence["benchmark"]["by_parser"][parser])
        expected.pop("documents")
        actual = _aggregate_evaluation(
            [result for result in evaluations if result["parser"] == parser]
        )
        if actual != expected:
            raise ValueError(f"offline {parser} structural metrics differ from committed evidence")


def _format_optional_metric(value: object) -> str:
    if value is None:
        return "N/A"
    if not isinstance(value, (int, float)):
        raise TypeError("metric value must be numeric or null")
    return f"{value:.3f}"


def render_structural_ir_v1_validation(evidence: dict[str, Any]) -> str:
    """Render the structural validation research report from machine evidence only."""
    aggregate = evidence["aggregate"]
    benchmark = evidence["benchmark"]["parser_representation_weighted_aggregate"]
    by_parser = evidence["benchmark"]["by_parser"]
    lines = [
        "# Structural IR v1 retained-corpus validation",
        "",
        "> Generated from `data/benchmarks/structural_ir_v1_validation.v1.json`. Do not edit measured values by hand.",
        "",
        "## Scope",
        "",
        "This report evaluates deterministic Vietnamese legal/planning structural marker detection, hierarchy reconstruction, and exact physical anchoring. It does not evaluate legal meaning, entities, relations, retrieval, or VLM/LLM behavior.",
        "",
        "## Structural schema",
        "",
        "Structural IR wire version 1 binds to the exact deterministic Physical IR v1 (wire version 2) bytes. Nodes are strict, immutable, parser-independent records with deterministic IDs, canonical paths, recognition evidence, and exact physical anchors.",
        "",
        "## Reference annotation policy",
        "",
        f"Reference v2 is an AI visual PDF structural re-audit of {evidence['reference_scope']['pages']} pages across {evidence['reference_scope']['documents']} PDFs and {evidence['reference_scope']['unique_scored_nodes']} unique scored nodes. Prior Physical IR and Structural extractor exposure is disclosed; neither Physical IR nor extractor output was used as reference truth.",
        "",
        f"The machine-readable re-audit log is `{evidence['reference_reaudit']['artifact_path']}` (SHA-256 `{evidence['reference_reaudit']['sha256']}`), with {evidence['reference_reaudit']['corrected_point_ordinal_records']} corrected point-ordinal records and {evidence['reference_reaudit']['changed_or_removed_v1_node_records']} changed or removed v1 node records.",
        "",
        "## Evaluation methodology",
        "",
        f"The combined result is a parser-representation-weighted aggregate over {evidence['reference_scope']['parser_evaluation_instances']} scored parser/reference instances. Documents represented by both Marker and MinerU contribute twice; this is not unique-corpus accuracy. Duplicate page/kind/ordinal groups match only through uniquely resolvable exact parent paths, and unresolved groups remain unmatched.",
        "",
        "## Legal marker rules",
        "",
        "Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Roman conversion is strict and kind-aware; Vietnamese POINT letters remain letters. Arabic clauses require an active ARTICLE and letter points require an active CLAUSE. TOC entries and line-start prose citations are suppressed before canonical-key reservation while their text remains BODY.",
        "",
        "## Planning generic rules",
        "",
        "Heading-like Roman and decimal outlines become GENERIC_SECTION nodes conservatively. A simple decimal outside legal context requires TITLE or strong heading evidence; outside-clause letter lists remain body text.",
        "",
        "## Content anchoring coverage",
        "",
        f"All {aggregate['content_text_blocks']} non-empty CONTENT blocks were exactly partitioned for all {aggregate['exact_text_partition_pairs']} parser/document pairs. All {aggregate['content_object_blocks']} empty CONTENT objects had one structural owner.",
        "",
        "## Per-document/per-parser results",
        "",
        "| Document | Parser | Pages | Nodes | Bytes | SHA-256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for entry in evidence["entries"]:
        lines.append(
            f"| {entry['document_id']} | {entry['parser']} | {entry['pages']} | {entry['node_count']} | {entry['byte_size_a']} | `{entry['sha256_a']}` |"
        )
    lines.extend(
        [
            "",
            "## Per-parser aggregates",
            "",
            "| Parser | Documents | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | Path exact | Title exact |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for parser in ("marker", "mineru"):
        result = by_parser[parser]
        node_metric = result["node_metrics"]
        edge_metric = result["parent_edge_metrics"]
        lines.append(
            f"| {parser} | {result['documents']} | {_format_optional_metric(node_metric['precision'])} | {_format_optional_metric(node_metric['recall'])} | {_format_optional_metric(node_metric['f1'])} | {_format_optional_metric(edge_metric['precision'])} | {_format_optional_metric(edge_metric['recall'])} | {_format_optional_metric(edge_metric['f1'])} | {_format_optional_metric(result['canonical_path_exact']['rate'])} | {_format_optional_metric(result['title_normalized_exact']['rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Combined parser-representation-weighted per-kind metrics",
            "",
            "| Kind | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kind, metric in benchmark["node_metrics_by_kind"].items():
        lines.append(
            f"| {kind} | {metric['tp']} | {metric['fp']} | {metric['fn']} | {_format_optional_metric(metric['precision'])} | {_format_optional_metric(metric['recall'])} | {_format_optional_metric(metric['f1'])} |"
        )
    node = benchmark["node_metrics"]
    edge = benchmark["parent_edge_metrics"]
    path = benchmark["canonical_path_exact"]
    title = benchmark["title_normalized_exact"]
    lines.extend(
        [
            "",
            f"Combined node detection: precision {_format_optional_metric(node['precision'])}, recall {_format_optional_metric(node['recall'])}, F1 {_format_optional_metric(node['f1'])} ({node['tp']} TP / {node['fp']} FP / {node['fn']} FN).",
            "",
            "## Hierarchy metrics",
            "",
            f"Full in-scope parent edges, including document-root edges: precision {_format_optional_metric(edge['precision'])}, recall {_format_optional_metric(edge['recall'])}, F1 {_format_optional_metric(edge['f1'])} ({edge['tp']} TP / {edge['fp']} FP / {edge['fn']} FN). Unmatched predictions are FP edges, unmatched references are FN edges, and a wrong parent contributes one FP plus one FN.",
            "",
            f"Canonical-path exact rate: {_format_optional_metric(path['rate'])} ({path['exact']}/{path['matched_nodes']} matched nodes).",
            "",
            f"Whitespace/case-normalized exact title rate: {_format_optional_metric(title['rate'])} ({title['exact']}/{title['eligible']}). No semantic title similarity is used.",
            "",
            "## Cross-parser agreement",
            "",
            "Agreement is consistency, not reference accuracy.",
            "",
            "| Document | Legal Jaccard | Generic Jaccard | All Jaccard |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in evidence["cross_parser_agreement"]:
        lines.append(
            f"| {item['document_id']} | {_format_optional_metric(item['legal']['jaccard'])} | {_format_optional_metric(item['generic']['jaccard'])} | {_format_optional_metric(item['all']['jaccard'])} |"
        )
    lines.extend(
        [
            "",
            "## False-positive analysis",
            "",
            f"The machine artifact contains all {len(evidence['false_positive_trace'])} unmatched predicted identities with page, parser, path, excerpt, physical kind, and deterministic category. TOC and line-start prose suppression occur before path reservation.",
            "",
            "## False-negative analysis",
            "",
            f"The machine artifact contains all {len(evidence['false_negative_trace'])} unmatched reference identities. Causes remain `unresolved` unless directly evidenced; the report does not infer OCR or scan causation per record.",
            "",
            "## Ambiguous cases",
            "",
            f"Rejected candidate counts: `{json.dumps(evidence['unresolved_candidate_diagnostics']['counts'], ensure_ascii=False, sort_keys=True)}`.",
            "",
            f"Corpus TOC audit rejected {evidence['toc_corpus_audit']['rejected_entries']} marker-shaped entries across {len(evidence['toc_corpus_audit']['parser_page_instances'])} parser/page instances before key reservation. Line-start prose audit rejected {evidence['prose_false_positive_audit']['rejected_candidates']} candidates. Stale/repeated context handling left {evidence['stale_context_audit']['duplicate_structural_keys_rejected_to_body']} duplicate keys and {evidence['stale_context_audit']['letter_items_outside_legal_clause_left_as_body']} outside-clause letter items as BODY.",
            "",
            f"Canonical occurrence suffixes after regeneration: {evidence['canonical_path_correction']['regenerated_suffix_paths']}. Duplicate structural keys are rejected to BODY with a machine diagnostic.",
            "",
            "## Limitations",
            "",
            "The reference is a partial-page AI visual re-audit, not human ground truth, and prior system exposure is disclosed. Combined metrics are representation-weighted. Matching has no fuzzy recovery. APPENDIX is terminal in profile v1, and title continuation remains conservative.",
            "",
            "## Evidence for #009 Selective VLM",
            "",
            "Observed unresolved categories include scanned pages without usable Physical IR text, visually clear structure hidden in tables, OCR-corrupted/split markers, and ambiguous generic numbering. These are candidates for later selective escalation; no VLM behavior is implemented here.",
            "",
        ]
    )
    return "\n".join(lines)


def write_structural_ir_v1_validation(root: Path, *, collect: bool) -> None:
    """Collect or offline-verify evidence, then deterministically render its report."""
    evidence_path = root / "data/benchmarks/structural_ir_v1_validation.v1.json"
    if collect:
        evidence = collect_structural_ir_v1_validation(root)
        evidence_path.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        evidence = _read_json_object(evidence_path)
        verify_committed_structural_evidence(root, evidence)
    report = root / "docs/research/structural-ir-v1-validation.md"
    report.write_text(render_structural_ir_v1_validation(evidence), encoding="utf-8", newline="\n")


__all__ = [
    "StructuralReferenceAnnotation",
    "StructuralReferenceNode",
    "StructuralReferencePage",
    "collect_structural_ir_v1_validation",
    "evaluate_structural_document",
    "load_structural_annotation",
    "load_structural_annotations",
    "render_structural_ir_v1_validation",
    "structural_set_jaccard",
    "verify_committed_structural_evidence",
    "write_structural_ir_v1_validation",
]
