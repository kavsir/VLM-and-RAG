# ruff: noqa: E501
"""Parser-neutral reference models and evaluation for Structural IR v1."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
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
from vlm_rag.structural_ir.vietnamese import VietnameseStructuralExtractor

if TYPE_CHECKING:
    from vlm_rag.physical_ir.v1 import PhysicalDocumentV1


class StructuralEvaluationModel(BaseModel):
    """Strict immutable configuration shared by structural evaluation records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=False)


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
    parent_canonical_path: str | None = None
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
        if self.parent_canonical_path not in {
            None,
            "document",
        } and not self.canonical_path.startswith(f"{self.parent_canonical_path}/"):
            raise ValueError("canonical_path must extend parent_canonical_path")
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

    annotation_schema_version: Literal[1]
    annotation_version: Literal["v1"]
    annotator: str = Field(min_length=1)
    annotator_type: Literal["ai_visual_audit"]
    annotation_method: Literal["visual_pdf_structural_audit"]
    created_at: AwareDatetime
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_physical_ir_exposure: Literal[True]
    physical_ir_used_as_reference: Literal[False]
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
        return self


def load_structural_annotation(path: Path) -> StructuralReferenceAnnotation:
    """Load one strict structural reference annotation."""
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"structural annotation must be a JSON object: {path}")
    return StructuralReferenceAnnotation.model_validate(dict(value))


def load_structural_annotations(root: Path) -> dict[str, StructuralReferenceAnnotation]:
    """Load and identity-index every v1 structural reference annotation."""
    directory = root / "data/structural_annotations"
    annotations = [load_structural_annotation(path) for path in sorted(directory.glob("*.v1.json"))]
    result = {annotation.document_id: annotation for annotation in annotations}
    if len(result) != len(annotations):
        raise ValueError("duplicate document_id across structural annotations")
    return result


def _node_page(node: StructuralNode) -> int | None:
    for anchor in node.heading_anchors:
        if anchor.role.value == "marker":
            return int(anchor.page_index)
    return None


def _occurrence_keys(
    values: Iterable[tuple[int, StructuralNodeKind, str | None, Any]],
) -> list[tuple[tuple[int, str, str | None, int], Any]]:
    counts: Counter[tuple[int, str, str | None]] = Counter()
    result: list[tuple[tuple[int, str, str | None, int], Any]] = []
    for page, kind, ordinal, value in values:
        base = (page, kind.value, ordinal)
        counts[base] += 1
        result.append(((*base, counts[base]), value))
    return result


def _metric(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _normalized_title(value: str) -> str:
    return " ".join(value.casefold().split())


def evaluate_structural_document(
    document: StructuralDocument, annotation: StructuralReferenceAnnotation
) -> dict[str, Any]:
    """Evaluate parser-neutral marker occurrence, hierarchy, path, and exact normalized title."""
    if (document.document_id, document.version_id, document.source_artifact_sha256) != (
        annotation.document_id,
        annotation.version_id,
        annotation.source_sha256,
    ):
        raise ValueError("Structural IR identity does not match reference annotation")
    audited_pages = {page.page_index for page in annotation.audited_pages}
    references = [node for page in annotation.audited_pages for node in page.nodes]
    predictions = [
        node
        for node in document.nodes[1:]
        if _node_page(node) is not None and _node_page(node) in audited_pages
    ]
    reference_keys = _occurrence_keys(
        (node.page_index, node.kind, node.ordinal_key, node) for node in references
    )
    prediction_keys = _occurrence_keys(
        (_node_page(node) or 0, node.kind, node.ordinal_key, node) for node in predictions
    )
    reference_by_key = dict(reference_keys)
    prediction_by_key = dict(prediction_keys)
    matched_keys = reference_by_key.keys() & prediction_by_key.keys()
    scored_keys = {key for key, node in reference_keys if node.scored}
    context_keys = {key for key, node in reference_keys if not node.scored}

    per_kind: dict[str, dict[str, int | float]] = {}
    for kind in StructuralNodeKind:
        if kind == StructuralNodeKind.DOCUMENT:
            continue
        kind_scored = {key for key in scored_keys if key[1] == kind.value}
        kind_predictions = {
            key for key in prediction_by_key if key[1] == kind.value and key not in context_keys
        }
        tp = len(kind_scored & kind_predictions)
        per_kind[kind.value] = _metric(
            tp, len(kind_predictions - kind_scored), len(kind_scored - kind_predictions)
        )

    scored_matches = matched_keys & scored_keys
    aggregate_tp = len(scored_matches)
    aggregate_predictions = set(prediction_by_key) - context_keys
    node_metric = _metric(
        aggregate_tp,
        len(aggregate_predictions - scored_keys),
        len(scored_keys - set(prediction_by_key)),
    )
    predicted_by_id = {node.id: node for node in document.nodes}
    reference_edges = {
        key
        for key in scored_keys
        if reference_by_key[key].parent_canonical_path not in {None, "document"}
    }
    predicted_edge_count = 0
    correct_edges = 0
    path_exact = 0
    title_total = 0
    title_exact = 0
    for key in scored_matches:
        reference = reference_by_key[key]
        prediction = prediction_by_key[key]
        if prediction.canonical_path == reference.canonical_path:
            path_exact += 1
        if reference.title is not None:
            title_total += 1
            if prediction.title is not None and _normalized_title(
                prediction.title
            ) == _normalized_title(reference.title):
                title_exact += 1
        if reference.parent_canonical_path not in {None, "document"}:
            predicted_edge_count += 1
            parent = predicted_by_id.get(prediction.parent_id or "")
            if parent is not None and parent.canonical_path == reference.parent_canonical_path:
                correct_edges += 1
    edge_metric = _metric(
        correct_edges,
        predicted_edge_count - correct_edges,
        len(reference_edges) - correct_edges,
    )
    return {
        "document_id": document.document_id,
        "audited_pages": sorted(audited_pages),
        "reference_scored_nodes": len(scored_keys),
        "reference_context_nodes": len(context_keys),
        "predictions_in_scope": len(predictions),
        "node_metrics": node_metric,
        "node_metrics_by_kind": per_kind,
        "parent_edge_metrics": edge_metric,
        "canonical_path_exact": {
            "matched_nodes": len(scored_matches),
            "exact": path_exact,
            "rate": path_exact / len(scored_matches) if scored_matches else 0.0,
        },
        "title_normalized_exact": {
            "eligible": title_total,
            "exact": title_exact,
            "rate": title_exact / title_total if title_total else 0.0,
        },
    }


def _sum_metrics(results: Iterable[dict[str, Any]], field: str) -> dict[str, int | float]:
    values = [result[field] for result in results]
    return _metric(
        sum(int(value["tp"]) for value in values),
        sum(int(value["fp"]) for value in values),
        sum(int(value["fn"]) for value in values),
    )


def _aggregate_evaluation(results: list[dict[str, Any]]) -> dict[str, Any]:
    kinds = [kind for kind in StructuralNodeKind if kind != StructuralNodeKind.DOCUMENT]
    by_kind: dict[str, dict[str, int | float]] = {}
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
            "rate": path_exact / matched if matched else 0.0,
        },
        "title_normalized_exact": {
            "eligible": title_eligible,
            "exact": title_exact,
            "rate": title_exact / title_eligible if title_eligible else 0.0,
        },
    }


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
            "jaccard": len(left & right) / len(union) if union else 1.0,
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
    extractor = VietnameseStructuralExtractor()
    output_entries: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    documents: dict[tuple[str, str], StructuralDocument] = {}
    aggregate_kinds: Counter[str] = Counter()
    aggregate_depth: Counter[int] = Counter()
    ambiguous_counts: Counter[str] = Counter()
    ambiguous_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
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
        result_a = extractor.extract(physical)
        result_b = extractor.extract(physical)
        validate_against_physical(result_a, physical)
        payload_a = structural_document_to_json(result_a).encode("utf-8")
        payload_b = structural_document_to_json(result_b).encode("utf-8")
        if payload_a != payload_b:
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
        evaluation = evaluate_structural_document(result_a, annotation)
        evaluation["parser"] = parser
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
    return {
        "validation_schema_version": 1,
        "validation_protocol": "structural_ir_v1_retained_corpus",
        "structural_ir_version": 1,
        "profile": "vi_legal_planning_v1",
        "reference_annotation_version": "v1",
        "extractor_inputs": "PhysicalDocumentV1 only; no raw parser fields or reference labels",
        "matching_algorithm": (
            "Within each audited page, nodes are paired one-to-one by exact kind, exact ordinal_key, "
            "and one-based reading occurrence for that page/kind/key. No fuzzy text or geometry match. "
            "Unscored context matches are excluded and predictions outside audited pages are ignored."
        ),
        "metric_formulas": {
            "node_precision": "TP / (TP + FP)",
            "node_recall": "TP / (TP + FN)",
            "node_f1": "2 * precision * recall / (precision + recall)",
            "parent_edge": (
                "correct matched scored child-to-parent canonical path; incorrect predicted edge is FP "
                "and incorrect or absent reference edge is FN"
            ),
            "canonical_path_exact_rate": "exact full paths / matched scored nodes",
            "title_normalized_exact_rate": (
                "casefolded whitespace-collapsed exact titles / matched nodes with reference title"
            ),
        },
        "reference_scope": {
            "documents": len(annotations),
            "pages": sum(len(value.audited_pages) for value in annotations.values()),
            "pages_by_document": {
                key: len(value.audited_pages) for key, value in sorted(annotations.items())
            },
            "scored_nodes": sum(reference_kind_counts.values()),
            "scored_nodes_by_kind": dict(sorted(reference_kind_counts.items())),
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
            "aggregate": _aggregate_evaluation(evaluations),
        },
        "cross_parser_agreement": agreements,
        "unresolved_candidate_diagnostics": {
            "counts": dict(sorted(ambiguous_counts.items())),
            "examples": dict(sorted(ambiguous_examples.items())),
        },
        "scope_boundary": (
            "Deterministic structural marker detection and physical anchoring only; no semantic legal "
            "edges, VLM/LLM, retrieval, RAG, or knowledge graph functionality."
        ),
    }


def verify_committed_structural_evidence(root: Path, evidence: dict[str, Any]) -> None:
    """Re-evaluate committed outputs and annotations without raw parser evidence."""
    annotations = load_structural_annotations(root)
    evaluations: list[dict[str, Any]] = []
    for entry in evidence["entries"]:
        path = root / entry["output_path"]
        payload = path.read_bytes()
        if len(payload) != entry["byte_size_a"] or _sha(payload) != entry["sha256_a"]:
            raise ValueError(f"committed Structural IR artifact changed: {path}")
        document = structural_document_from_json(payload.decode("utf-8"))
        result = evaluate_structural_document(document, annotations[document.document_id])
        result["parser"] = entry["parser"]
        evaluations.append(result)
    if _aggregate_evaluation(evaluations) != evidence["benchmark"]["aggregate"]:
        raise ValueError("offline structural benchmark metrics differ from committed evidence")


def render_structural_ir_v1_validation(evidence: dict[str, Any]) -> str:
    """Render the structural validation research report from machine evidence only."""
    aggregate = evidence["aggregate"]
    benchmark = evidence["benchmark"]["aggregate"]
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
        f"The v1 AI visual structural reference contains {evidence['reference_scope']['pages']} pages across {evidence['reference_scope']['documents']} PDFs and {evidence['reference_scope']['scored_nodes']} scored nodes. The annotator had prior Physical IR exposure, but reference structure was established from rendered PDFs and contains no parser block identity.",
        "",
        "## Legal marker rules",
        "",
        "Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Arabic clauses require an active ARTICLE; letter points require an active CLAUSE. Embedded cross-references are not headings.",
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
            "## Per-kind metrics",
            "",
            "| Kind | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for kind, metric in benchmark["node_metrics_by_kind"].items():
        lines.append(
            f"| {kind} | {metric['tp']} | {metric['fp']} | {metric['fn']} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['f1']:.3f} |"
        )
    edge = benchmark["parent_edge_metrics"]
    path = benchmark["canonical_path_exact"]
    title = benchmark["title_normalized_exact"]
    lines.extend(
        [
            "",
            "## Hierarchy metrics",
            "",
            f"Parent edges: precision {edge['precision']:.3f}, recall {edge['recall']:.3f}, F1 {edge['f1']:.3f} ({edge['tp']} TP / {edge['fp']} FP / {edge['fn']} FN).",
            "",
            f"Canonical-path exact rate: {path['rate']:.3f} ({path['exact']}/{path['matched_nodes']} matched nodes).",
            "",
            f"Whitespace/case-normalized exact title rate: {title['rate']:.3f} ({title['exact']}/{title['eligible']}). No semantic title similarity is used.",
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
            f"| {item['document_id']} | {item['legal']['jaccard']:.3f} | {item['generic']['jaccard']:.3f} | {item['all']['jaccard']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## False-positive analysis",
            "",
            "False positives are audited-page predictions that fail exact parser-neutral marker occurrence matching. The main risks are table-of-contents entries and short numbered body lines represented as headings by a parser.",
            "",
            "## False-negative analysis",
            "",
            "False negatives arise when physical text is absent (especially scans), marker text is fragmented/corrupted, or conservative heading evidence intentionally rejects ambiguous numbering.",
            "",
            "## Ambiguous cases",
            "",
            f"Rejected candidate counts: `{json.dumps(evidence['unresolved_candidate_diagnostics']['counts'], ensure_ascii=False, sort_keys=True)}`.",
            "",
            "## Limitations",
            "",
            "The reference is partial-page AI visual audit, not human ground truth. Metrics are repeated per available parser representation. Exact ordinal occurrence matching intentionally has no fuzzy recovery. Title continuation remains conservative.",
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
    "verify_committed_structural_evidence",
    "write_structural_ir_v1_validation",
]
