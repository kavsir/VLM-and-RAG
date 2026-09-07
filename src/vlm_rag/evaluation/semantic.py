# ruff: noqa: E501
"""Offline reference, evaluation, and retained-corpus evidence for Semantic IR v1."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, field_validator

from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1
from vlm_rag.semantic_ir import (
    SemanticDocument,
    SemanticMentionKind,
    build_semantic_document,
    extract_mention_candidates,
    semantic_document_from_json,
    semantic_document_to_json,
)
from vlm_rag.structural_ir import structural_document_from_json
from vlm_rag.vlm import SelectionBudget, VLMTaskType, select_visual_evidence


class SemanticEvaluationModel(BaseModel):
    """Strict immutable reference model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SemanticReferenceMention(SemanticEvaluationModel):
    """PDF-excerpt-based mention identity independent of runtime object IDs."""

    reference_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    kind: SemanticMentionKind
    raw_text: str = Field(min_length=1)
    normalized_value: str | None = None
    evidence_excerpt: str = Field(min_length=1)
    char_start: NonNegativeInt
    char_end: int = Field(gt=0)
    legal_reference: dict[str, str | None] | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, value: object) -> object:
        return SemanticMentionKind(value) if isinstance(value, str) else value


class SemanticReferenceStatement(SemanticEvaluationModel):
    reference_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    exact_evidence_excerpt: str = Field(min_length=1)


class VisualAssistanceReference(SemanticEvaluationModel):
    reference_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    bbox_normalized_1000: tuple[int, int, int, int]
    task_type: VLMTaskType
    evidence_note: str = Field(min_length=1)

    @field_validator("bbox_normalized_1000", mode="before")
    @classmethod
    def coerce_bbox(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("task_type", mode="before")
    @classmethod
    def coerce_task(cls, value: object) -> object:
        return VLMTaskType(value) if isinstance(value, str) else value


class SemanticReferencePage(SemanticEvaluationModel):
    page_index: NonNegativeInt
    pdf_page_number_1_based: int = Field(gt=0)
    statements: tuple[SemanticReferenceStatement, ...] = ()
    mentions: tuple[SemanticReferenceMention, ...] = ()
    visual_assistance: tuple[VisualAssistanceReference, ...] = ()

    @field_validator("statements", "mentions", "visual_assistance", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class SemanticReferenceAnnotation(SemanticEvaluationModel):
    annotation_schema_version: Literal[1] = 1
    annotation_version: Literal["v1"] = "v1"
    annotator_type: Literal["ai_assisted_evidence_audit"]
    annotation_method: Literal[
        "fixed_visual_page_audit_plus_retained_representation_candidate_audit"
    ]
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_physical_ir_exposure: Literal[True]
    prior_structural_ir_exposure: Literal[True]
    prior_semantic_extractor_exposure: Literal[True]
    independent_or_blind_ground_truth: Literal[False]
    candidate_representations: tuple[str, ...]
    pages: tuple[SemanticReferencePage, ...]

    @field_validator("candidate_representations", "pages", mode="before")
    @classmethod
    def coerce_pages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def create_semantic_reference_annotations(root: Path) -> tuple[SemanticReferenceAnnotation, ...]:
    """Create disclosed candidates on the frozen visually audited page set."""
    physical_evidence = json.loads(
        (root / "data/benchmarks/physical_ir_v1_validation.v1.json").read_text(encoding="utf-8")
    )
    candidates_by_document: dict[str, list[tuple[str, SemanticDocument]]] = {}
    for raw_entry in physical_evidence["entries"]:
        document_id = str(raw_entry["document_id"])
        parser = str(raw_entry["parser"])
        physical = _normalize(parser, root / str(raw_entry["raw_directory"]))
        structural_path = (
            root / "data/structural_ir" / document_id.replace("-", "_") / f"{parser}.v1.json"
        )
        structural = structural_document_from_json(structural_path.read_text(encoding="utf-8"))
        candidates_by_document.setdefault(document_id, []).append(
            (parser, build_semantic_document(physical, structural))
        )
    source_paths = sorted((root / "data/structural_annotations").glob("*.v4.json"))
    annotations: list[SemanticReferenceAnnotation] = []
    for source_path in source_paths:
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if "audited_pages" not in source:
            continue
        candidate_documents = candidates_by_document[str(source["document_id"])]
        candidate_representations = tuple(sorted(parser for parser, _ in candidate_documents))
        pages: list[SemanticReferencePage] = []
        for source_page in source["audited_pages"]:
            page_index = int(source_page["page_index"])
            statements: list[SemanticReferenceStatement] = []
            mentions: list[SemanticReferenceMention] = []
            seen_excerpts: set[str] = set()
            seen_mentions: set[tuple[str, str, str | None]] = set()
            page_statements = [
                statement
                for _, semantic in candidate_documents
                for statement in semantic.statements
                if statement.evidence_anchors[0].page_index == page_index
            ]
            for statement in page_statements:
                excerpt = statement.text
                if excerpt in seen_excerpts:
                    continue
                seen_excerpts.add(excerpt)
                statement_id = _stable_id(
                    "semantic-reference-statement", source["document_id"], page_index, excerpt
                )
                statements.append(
                    SemanticReferenceStatement(
                        reference_id=statement_id,
                        page_index=page_index,
                        exact_evidence_excerpt=excerpt,
                    )
                )
                for candidate in extract_mention_candidates(excerpt):
                    mention_key = (
                        candidate.kind.value,
                        candidate.raw_text,
                        candidate.normalized_value,
                    )
                    if mention_key in seen_mentions:
                        continue
                    seen_mentions.add(mention_key)
                    legal = (
                        candidate.legal_reference.model_dump(mode="json")
                        if candidate.legal_reference
                        else None
                    )
                    mentions.append(
                        SemanticReferenceMention(
                            reference_id=_stable_id(
                                "semantic-reference-mention",
                                source["document_id"],
                                page_index,
                                excerpt,
                                candidate.start,
                                candidate.end,
                                candidate.kind.value,
                            ),
                            page_index=page_index,
                            kind=candidate.kind,
                            raw_text=candidate.raw_text,
                            normalized_value=candidate.normalized_value,
                            evidence_excerpt=excerpt,
                            char_start=candidate.start,
                            char_end=candidate.end,
                            legal_reference=legal,
                        )
                    )
            visual: list[VisualAssistanceReference] = []
            rationale = str(source_page["selection_rationale"])
            if "table/figure" in rationale.casefold():
                visual.append(
                    VisualAssistanceReference(
                        reference_id=_stable_id(
                            "visual-assistance", source["document_id"], page_index
                        ),
                        page_index=page_index,
                        bbox_normalized_1000=(0, 0, 1000, 1000),
                        task_type=VLMTaskType.FIGURE_OR_IMAGE_OBSERVATION,
                        evidence_note="The frozen #008 visual audit selected this page for a table/figure in active structural context; the full-page label is coarse and is not runtime input.",
                    )
                )
            pages.append(
                SemanticReferencePage(
                    page_index=page_index,
                    pdf_page_number_1_based=int(source_page["pdf_page_number_1_based"]),
                    statements=tuple(statements),
                    mentions=tuple(mentions),
                    visual_assistance=tuple(visual),
                )
            )
        annotations.append(
            SemanticReferenceAnnotation(
                annotator_type="ai_assisted_evidence_audit",
                annotation_method="fixed_visual_page_audit_plus_retained_representation_candidate_audit",
                document_id=source["document_id"],
                version_id=source["version_id"],
                source_artifact_sha256=source["source_sha256"],
                prior_physical_ir_exposure=True,
                prior_structural_ir_exposure=True,
                prior_semantic_extractor_exposure=True,
                independent_or_blind_ground_truth=False,
                candidate_representations=candidate_representations,
                pages=tuple(pages),
            )
        )
    return tuple(annotations)


def write_semantic_reference_annotations(root: Path) -> tuple[SemanticReferenceAnnotation, ...]:
    annotations = create_semantic_reference_annotations(root)
    output_root = root / "data/semantic_annotations"
    output_root.mkdir(parents=True, exist_ok=True)
    for annotation in annotations:
        name = annotation.document_id.replace("-", "_") + ".v1.json"
        (output_root / name).write_bytes(_canonical_bytes(annotation.model_dump(mode="json")))
    return annotations


def load_semantic_reference_annotations(root: Path) -> dict[str, SemanticReferenceAnnotation]:
    result: dict[str, SemanticReferenceAnnotation] = {}
    for path in sorted((root / "data/semantic_annotations").glob("*.v1.json")):
        annotation = SemanticReferenceAnnotation.model_validate_json(path.read_bytes())
        result[annotation.document_id] = annotation
    if len(result) != 6 or sum(len(value.pages) for value in result.values()) != 46:
        raise ValueError("Semantic reference v1 must contain the exact six-document, 46-page set")
    return result


def _normalize(parser: str, raw_directory: Path) -> Any:
    if parser == "marker":
        return MarkerPhysicalNormalizerV1().normalize(raw_directory)
    if parser == "mineru":
        return MinerUPhysicalNormalizerV1().normalize(raw_directory)
    raise ValueError(f"unsupported parser {parser!r}")


def _metric(tp: int, fp: int, fn: int) -> dict[str, int | float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _mention_key(
    page: int, kind: str, raw: str, normalized: str | None
) -> tuple[int, str, str, str | None]:
    return page, kind, raw, normalized


def _evaluate_document(
    document: SemanticDocument, annotation: SemanticReferenceAnnotation
) -> dict[str, Any]:
    audited_pages = {page.page_index for page in annotation.pages}
    reference_mentions = [mention for page in annotation.pages for mention in page.mentions]
    predictions = [
        mention
        for mention in document.mentions
        if mention.evidence_anchors[0].page_index in audited_pages
    ]
    reference_counter = Counter(
        _mention_key(item.page_index, item.kind.value, item.raw_text, item.normalized_value)
        for item in reference_mentions
    )
    prediction_counter = Counter(
        _mention_key(
            item.evidence_anchors[0].page_index,
            item.kind.value,
            item.raw_text,
            item.normalized_value,
        )
        for item in predictions
    )
    statements_by_id = {statement.id: statement for statement in document.statements}
    reference_span_counter = Counter(
        (
            item.page_index,
            item.kind.value,
            item.evidence_excerpt,
            item.char_start,
            item.char_end,
            item.raw_text,
        )
        for item in reference_mentions
    )
    prediction_span_counter: Counter[tuple[int, str, str, int, int, str]] = Counter()
    for item in predictions:
        anchor = item.evidence_anchors[0]
        statement = statements_by_id[item.statement_id]
        statement_anchor = statement.evidence_anchors[0]
        if anchor.anchor_type != "text" or statement_anchor.anchor_type != "text":
            continue
        prediction_span_counter[
            (
                anchor.page_index,
                item.kind.value,
                statement.text,
                anchor.char_start - statement_anchor.char_start,
                anchor.char_end - statement_anchor.char_start,
                item.raw_text,
            )
        ] += 1
    kinds: dict[str, dict[str, int | float | None]] = {}
    for kind in SemanticMentionKind:
        ref = Counter(
            {key: count for key, count in reference_counter.items() if key[1] == kind.value}
        )
        pred = Counter(
            {key: count for key, count in prediction_counter.items() if key[1] == kind.value}
        )
        tp = sum((ref & pred).values())
        kinds[kind.value] = _metric(tp, sum(pred.values()) - tp, sum(ref.values()) - tp)
    tp = sum((reference_counter & prediction_counter).values())
    overall = _metric(
        tp, sum(prediction_counter.values()) - tp, sum(reference_counter.values()) - tp
    )
    exact_span_tp = sum((reference_span_counter & prediction_span_counter).values())
    statement_refs = [statement for page in annotation.pages for statement in page.statements]
    statement_hits = 0
    for reference in statement_refs:
        if any(
            statement.evidence_anchors[0].page_index == reference.page_index
            and reference.exact_evidence_excerpt == statement.text
            for statement in document.statements
        ):
            statement_hits += 1
    legal_refs = [
        item for item in reference_mentions if item.kind == SemanticMentionKind.LEGAL_REFERENCE
    ]
    legal_exact = sum(
        1
        for reference in legal_refs
        if any(
            prediction.evidence_anchors[0].page_index == reference.page_index
            and prediction.raw_text == reference.raw_text
            and (
                prediction.legal_reference.model_dump(mode="json")
                if prediction.legal_reference
                else None
            )
            == reference.legal_reference
            for prediction in predictions
        )
    )
    return {
        "mention_metrics": overall,
        "mention_metrics_by_kind": kinds,
        "exact_evidence_span_match": {
            "matched": exact_span_tp,
            "reference": sum(reference_span_counter.values()),
            "rate": exact_span_tp / sum(reference_span_counter.values())
            if reference_span_counter
            else None,
        },
        "normalized_value_exact_match": {
            "matched": tp,
            "reference": sum(reference_counter.values()),
            "rate": tp / sum(reference_counter.values()) if reference_counter else None,
        },
        "legal_reference_component_exact_match": {
            "matched": legal_exact,
            "reference": len(legal_refs),
            "rate": legal_exact / len(legal_refs) if legal_refs else None,
        },
        "statement_evidence_coverage": {
            "covered": statement_hits,
            "reference": len(statement_refs),
            "rate": statement_hits / len(statement_refs) if statement_refs else None,
        },
        "false_positive_trace": [
            {
                "document_id": document.document_id,
                "page_index": key[0],
                "semantic_kind": key[1],
                "raw_evidence": key[2],
                "normalized_value": key[3],
                "reason": "unresolved",
            }
            for key, count in sorted((prediction_counter - reference_counter).items())
            for _ in range(count)
        ],
        "false_negative_trace": [
            {
                "document_id": document.document_id,
                "page_index": key[0],
                "semantic_kind": key[1],
                "raw_evidence": key[2],
                "normalized_value": key[3],
                "reason": "unresolved",
            }
            for key, count in sorted((reference_counter - prediction_counter).items())
            for _ in range(count)
        ],
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    overall = _metric(
        sum(int(x["mention_metrics"]["tp"]) for x in results),
        sum(int(x["mention_metrics"]["fp"]) for x in results),
        sum(int(x["mention_metrics"]["fn"]) for x in results),
    )
    by_kind: dict[str, Any] = {}
    for kind in SemanticMentionKind:
        values = [x["mention_metrics_by_kind"][kind.value] for x in results]
        by_kind[kind.value] = _metric(
            sum(int(x["tp"]) for x in values),
            sum(int(x["fp"]) for x in values),
            sum(int(x["fn"]) for x in values),
        )

    def rate(field: str, matched_key: str, reference_key: str) -> dict[str, int | float | None]:
        matched = sum(int(item[field][matched_key]) for item in results)
        reference = sum(int(item[field][reference_key]) for item in results)
        return {
            matched_key: matched,
            reference_key: reference,
            "rate": matched / reference if reference else None,
        }

    return {
        "mention_metrics": overall,
        "mention_metrics_by_kind": by_kind,
        "exact_evidence_span_match": rate("exact_evidence_span_match", "matched", "reference"),
        "normalized_value_exact_match": rate(
            "normalized_value_exact_match", "matched", "reference"
        ),
        "legal_reference_component_exact_match": rate(
            "legal_reference_component_exact_match", "matched", "reference"
        ),
        "statement_evidence_coverage": rate("statement_evidence_coverage", "covered", "reference"),
    }


def normalized_edit_distance(reference: str, prediction: str) -> float:
    """Return Unicode-codepoint edit distance divided by the longer input length."""
    if reference == prediction:
        return 0.0
    if not reference or not prediction:
        return 1.0
    previous = list(range(len(prediction) + 1))
    for row, left in enumerate(reference, start=1):
        current = [row]
        for column, right in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1] / max(len(reference), len(prediction))


def _semantic_output(document_id: str, parser: str) -> Path:
    return Path("data/semantic_ir") / document_id.replace("-", "_") / f"{parser}.v1.json"


def _cross_parser(documents: dict[tuple[str, str], SemanticDocument]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for document_id in sorted({key[0] for key in documents}):
        marker = documents.get((document_id, "marker"))
        mineru = documents.get((document_id, "mineru"))
        if marker is None or mineru is None:
            continue
        marker_statements = {item.id: item for item in marker.statements}
        mineru_statements = {item.id: item for item in mineru.statements}
        left = {
            (
                m.kind.value,
                m.raw_text,
                m.normalized_value,
                marker_statements[m.statement_id].structural_canonical_path,
            )
            for m in marker.mentions
        }
        right = {
            (
                m.kind.value,
                m.raw_text,
                m.normalized_value,
                mineru_statements[m.statement_id].structural_canonical_path,
            )
            for m in mineru.mentions
        }
        union = left | right
        output.append(
            {
                "document_id": document_id,
                "marker_mentions": len(left),
                "mineru_mentions": len(right),
                "intersection": len(left & right),
                "union": len(union),
                "jaccard_consistency": len(left & right) / len(union) if union else None,
            }
        )
    return output


def collect_semantic_ir_v1_validation(root: Path) -> dict[str, Any]:
    """Build all ten outputs twice, without VLM/network, and collect honest evidence."""
    annotations = load_semantic_reference_annotations(root)
    physical_evidence_path = root / "data/benchmarks/physical_ir_v1_validation.v1.json"
    structural_evidence_path = root / "data/benchmarks/structural_ir_v1_validation.v1.json"
    physical_evidence = json.loads(physical_evidence_path.read_text(encoding="utf-8"))
    physical_freeze_before = _sha(physical_evidence_path.read_bytes())
    structural_paths = sorted((root / "data/structural_ir").glob("**/*.v1.json"))
    structural_freeze_before = {
        str(path.relative_to(root)).replace("\\", "/"): _sha(path.read_bytes())
        for path in structural_paths
    }
    entries: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    selector_pairs: Counter[tuple[str, int]] = Counter()
    reference_pairs: Counter[tuple[str, int]] = Counter()
    semantic_documents: dict[tuple[str, str], SemanticDocument] = {}
    for raw_entry in physical_evidence["entries"]:
        document_id = str(raw_entry["document_id"])
        parser = str(raw_entry["parser"])
        physical = _normalize(parser, root / str(raw_entry["raw_directory"]))
        structural_relative = (
            Path("data/structural_ir") / document_id.replace("-", "_") / f"{parser}.v1.json"
        )
        structural = structural_document_from_json(
            (root / structural_relative).read_text(encoding="utf-8")
        )
        first = build_semantic_document(physical, structural)
        second = build_semantic_document(physical, structural)
        first_bytes = semantic_document_to_json(first).encode("utf-8")
        second_bytes = semantic_document_to_json(second).encode("utf-8")
        if first_bytes != second_bytes:
            raise ValueError(f"non-deterministic Semantic IR for {document_id}/{parser}")
        output_relative = _semantic_output(document_id, parser)
        output_path = root / output_relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(first_bytes)
        semantic_documents[(document_id, parser)] = first
        evaluation = _evaluate_document(first, annotations[document_id])
        evaluation.update({"document_id": document_id, "parser": parser})
        evaluations.append(evaluation)
        selection = select_visual_evidence(physical, budget=SelectionBudget())
        for request in selection.requests:
            selector_pairs[(document_id, request.page_index)] += 1
        for page in annotations[document_id].pages:
            if page.visual_assistance:
                reference_pairs[(document_id, page.page_index)] += 1
        entries.append(
            {
                "document_id": document_id,
                "parser": parser,
                "output_path": str(output_relative).replace("\\", "/"),
                "sha256_a": _sha(first_bytes),
                "sha256_b": _sha(second_bytes),
                "byte_identical": first_bytes == second_bytes,
                "statement_count": len(first.statements),
                "mention_count": len(first.mentions),
                "mention_kind_histogram": dict(
                    sorted(Counter(item.kind.value for item in first.mentions).items())
                ),
                "eligible_vlm_requests": len(selection.requests) + len(selection.non_selections),
                "selected_vlm_requests": len(selection.requests),
                "non_selected_vlm_requests": len(selection.non_selections),
            }
        )
    if structural_freeze_before != {
        str(path.relative_to(root)).replace("\\", "/"): _sha(path.read_bytes())
        for path in structural_paths
    }:
        raise ValueError("Structural IR freeze violation")
    if physical_freeze_before != _sha(physical_evidence_path.read_bytes()):
        raise ValueError("Physical IR evidence freeze violation")
    selector_page_keys = set(selector_pairs)
    reference_page_keys = set(reference_pairs)
    selector_tp = len(selector_page_keys & reference_page_keys)
    selector_metric = _metric(
        selector_tp,
        len(selector_page_keys - reference_page_keys),
        len(reference_page_keys - selector_page_keys),
    )
    aggregate = _aggregate(evaluations)
    by_parser = {
        parser: _aggregate([x for x in evaluations if x["parser"] == parser])
        for parser in ("marker", "mineru")
    }
    evidence: dict[str, Any] = {
        "validation_schema_version": 1,
        "validation_protocol": "semantic_ir_v1_fixed_46_page_reference_v1",
        "semantic_ir_version": 1,
        "semantic_profile": "vi_legal_planning_semantic_v1",
        "runtime_inputs": [
            "PhysicalDocumentV1",
            "StructuralDocument v1",
            "validated VLMObservation records",
        ],
        "reference": {
            "version": "v1",
            "schema": 1,
            "documents": 6,
            "pages": sum(len(x.pages) for x in annotations.values()),
            "provenance": "AI-assisted evidence audit with disclosed prior Physical, Structural, and Semantic extractor exposure; not human, blind, or independent ground truth.",
            "mention_counts_by_kind": dict(
                sorted(
                    Counter(
                        m.kind.value
                        for a in annotations.values()
                        for p in a.pages
                        for m in p.mentions
                    ).items()
                )
            ),
        },
        "representations": 10,
        "entries": entries,
        "text_only": aggregate,
        "selective_vlm": {
            **aggregate,
            "quality_experiment_completed": False,
            "executed_requests": 0,
            "semantic_gain": {
                "additional_correct_mentions": 0,
                "new_false_mentions": 0,
                "mention_f1_net_change": 0.0,
            },
            "warning": "No real model execution or corpus replay output exists; equality with TEXT_ONLY is an engineering baseline, not evidence of VLM quality.",
        },
        "all_eligible_vlm": None,
        "selector": {
            **selector_metric,
            "selected_requests": sum(item["selected_vlm_requests"] for item in entries),
            "eligible_requests": sum(item["eligible_vlm_requests"] for item in entries),
            "requests_per_parser_representation": sum(
                item["selected_vlm_requests"] for item in entries
            )
            / 10,
            "evaluation_unit": "document/page; coarse audited visual label",
        },
        "marker_metrics": by_parser["marker"],
        "mineru_metrics": by_parser["mineru"],
        "combined_representation_weighted": aggregate,
        "cross_parser_consistency": _cross_parser(semantic_documents),
        "false_positive_trace": [
            item | {"parser": result["parser"]}
            for result in evaluations
            for item in result["false_positive_trace"]
        ],
        "false_negative_trace": [
            item | {"parser": result["parser"]}
            for result in evaluations
            for item in result["false_negative_trace"]
        ],
        "vlm_failure_trace": {
            "selector_miss": selector_metric["fn"],
            "render_failure": 0,
            "hash_mismatch": 0,
            "vlm_request_failure": 0,
            "malformed_output": 0,
            "normalization_failure": 0,
            "unsupported_observation": 0,
        },
        "vlm_transcription_quality": None,
        "real_vlm_experiment": {
            "completed": False,
            "model_id": None,
            "endpoint_type": None,
            "prompt_version": "semantic-evidence-v1",
            "temperature": None,
            "request_count": 0,
            "latency_ms": None,
            "statement": "REAL VLM QUALITY EXPERIMENT NOT COMPLETED",
        },
        "semantic_determinism": {
            "byte_identical": sum(bool(item["byte_identical"]) for item in entries),
            "required": 10,
            "all_passed": all(bool(item["byte_identical"]) for item in entries),
        },
        "structural_hash_freeze": {
            "unchanged": True,
            "files": len(structural_freeze_before),
            "sha256_by_path": structural_freeze_before,
        },
        "physical_hash_freeze": {
            "unchanged": True,
            "evidence_path": "data/benchmarks/physical_ir_v1_validation.v1.json",
            "evidence_sha256": physical_freeze_before,
            "all_v0_hashes_preserved": physical_evidence["all_v0_hashes_preserved"],
        },
        "source_structural_validation_evidence_sha256": _sha(structural_evidence_path.read_bytes()),
    }
    return evidence


def render_semantic_report(evidence: dict[str, Any]) -> str:
    text = evidence["text_only"]["mention_metrics"]
    selector = evidence["selector"]
    kinds = evidence["text_only"]["mention_metrics_by_kind"]
    kind_rows = "\n".join(
        f"| {kind} | {value['tp']} | {value['fp']} | {value['fn']} | {value['precision']!r} | {value['recall']!r} | {value['f1']!r} |"
        for kind, value in kinds.items()
    )
    marker = evidence["marker_metrics"]["mention_metrics"]
    mineru = evidence["mineru_metrics"]["mention_metrics"]
    cross_parser = "\n".join(
        f"- `{item['document_id']}`: {item['jaccard_consistency']!r} (intersection {item['intersection']}, union {item['union']})."
        for item in evidence["cross_parser_consistency"]
    )
    return f"""# Semantic IR v1 + Selective VLM validation

This evaluates ten parser representations over the exact fixed 46-page #008 audit set. The reference is AI-assisted and discloses prior Physical IR, Structural IR, and Semantic extractor exposure; it is not human, blind, independent ground truth.

## Reference provenance

Semantic reference v1/schema 1 covers six documents and 46 fixed pages. Identity uses document, PDF page, exact evidence excerpt, and character span—never parser block IDs or runtime Semantic IDs. Candidate coverage is the union of retained parser representations and is therefore exposure-biased; metrics are diagnostic, not an independent estimate of real-world accuracy. Counts by kind: `{json.dumps(evidence["reference"]["mention_counts_by_kind"], ensure_ascii=False, sort_keys=True)}`.

## Text-only baseline

- TEXT_ONLY mention P/R/F1: {text["precision"]!r} / {text["recall"]!r} / {text["f1"]!r}.
- Exact evidence-span rate: {evidence["text_only"]["exact_evidence_span_match"]["rate"]!r}.
- Normalized-value exact rate: {evidence["text_only"]["normalized_value_exact_match"]["rate"]!r}.
- Legal-reference component exact rate: {evidence["text_only"]["legal_reference_component_exact_match"]["rate"]!r}.
- Statement exact-evidence coverage: {evidence["text_only"]["statement_evidence_coverage"]["rate"]!r}.

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
{kind_rows}

## Selector and A/B ablation

- Selector coarse page-level P/R/F1: {selector["precision"]!r} / {selector["recall"]!r} / {selector["f1"]!r}.
- Selected requests: {selector["selected_requests"]}; eligible requests: {selector["eligible_requests"]}; requests/representation: {selector["requests_per_parser_representation"]!r}.
- TEXT_ONLY F1: {text["f1"]!r}.
- SELECTIVE_VLM F1: {evidence["selective_vlm"]["mention_metrics"]["f1"]!r}; executed requests: 0; F1 delta: 0.0.
- ALL_ELIGIBLE_VLM: N/A.
- Zero-request equality is an engineering baseline, not evidence that a model improves quality.

## Parser representations and consistency

- Marker P/R/F1: {marker["precision"]!r} / {marker["recall"]!r} / {marker["f1"]!r}.
- MinerU P/R/F1: {mineru["precision"]!r} / {mineru["recall"]!r} / {mineru["f1"]!r}.
- Combined results are parser-representation-weighted: six source documents produce ten representations.

Cross-parser mention/path/value Jaccard consistency:

{cross_parser}

## Failure and efficiency trace

- Semantic false positives: {len(evidence["false_positive_trace"])}; semantic false negatives: {len(evidence["false_negative_trace"])}. Machine records retain document, parser, page, kind, raw evidence, normalized value, and `unresolved` cause.
- VLM selector misses: {evidence["vlm_failure_trace"]["selector_miss"]}; other VLM failures: zero because no corpus requests were executed.
- Transcription exact match/edit distance, model ID, endpoint, temperature, image bytes/dimensions, latency, failed requests, and tokens: N/A.

## Reproducibility

- Semantic determinism: {evidence["semantic_determinism"]["byte_identical"]}/10 byte-identical.
- Structural and Physical freeze checks: passed.

## Real VLM experiment

**REAL VLM QUALITY EXPERIMENT NOT COMPLETED.** No model ID, endpoint, latency, or model-derived corpus outputs are claimed. Replay/contract tests prove offline engineering behavior only. A real model execution is required before claiming improvement.

## Limitations

Statements are exact direct-content spans, not paraphrases. Mentions use a controlled deterministic Vietnamese taxonomy. Matching uses page, kind, raw text, and normalized value; failures remain `unresolved` unless evidence establishes a cause. Parser aggregates are representation-weighted: six source documents produce ten parser representations. Selector labels are coarse page-level visual-audit labels, so selector metrics are diagnostic rather than region-level quality estimates. No RAG, KG, cross-document citation resolution, or entity resolution is implemented.
"""


def write_semantic_ir_v1_validation(root: Path, *, collect: bool) -> dict[str, Any]:
    evidence_path = root / "data/benchmarks/semantic_ir_v1_validation.v1.json"
    report_path = root / "docs/research/semantic-ir-selective-vlm-v1-validation.md"
    if collect:
        evidence = collect_semantic_ir_v1_validation(root)
        evidence_path.write_bytes(_canonical_bytes(evidence))
    else:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        for entry in evidence["entries"]:
            payload = (root / entry["output_path"]).read_bytes()
            semantic_document_from_json(payload.decode("utf-8"))
            if _sha(payload) != entry["sha256_a"]:
                raise ValueError(f"committed Semantic IR hash mismatch: {entry['output_path']}")
    report_path.write_text(render_semantic_report(evidence), encoding="utf-8", newline="\n")
    return evidence


__all__ = [
    "SemanticReferenceAnnotation",
    "collect_semantic_ir_v1_validation",
    "create_semantic_reference_annotations",
    "load_semantic_reference_annotations",
    "normalized_edit_distance",
    "render_semantic_report",
    "write_semantic_ir_v1_validation",
    "write_semantic_reference_annotations",
]
