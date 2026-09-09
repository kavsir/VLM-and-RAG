# ruff: noqa: E501
"""Offline reference, evaluation, and retained-corpus evidence for Semantic IR v1."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from vlm_rag.normalizers.marker_v1 import MarkerPhysicalNormalizerV1
from vlm_rag.normalizers.mineru_v1 import MinerUPhysicalNormalizerV1
from vlm_rag.semantic_ir import (
    LegalReferenceComponents,
    QuantityComponents,
    SemanticDocument,
    SemanticMention,
    SemanticMentionKind,
    build_semantic_document,
    semantic_document_from_json,
    semantic_document_to_json,
)
from vlm_rag.structural_ir import structural_document_from_json
from vlm_rag.vlm import (
    SelectionBudget,
    VLMSelectionResult,
    VLMTaskType,
    select_visual_evidence,
)

if TYPE_CHECKING:
    from vlm_rag.physical_ir.models import BoundingBox


class SemanticEvaluationModel(BaseModel):
    """Strict immutable reference model configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SemanticReferenceMention(SemanticEvaluationModel):
    """PDF-excerpt-based mention identity independent of runtime object IDs."""

    reference_mention_id: str = Field(min_length=1)
    reference_statement_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    kind: SemanticMentionKind
    raw_text: str = Field(min_length=1)
    normalized_value: str | None = None
    char_start: NonNegativeInt
    char_end: PositiveInt
    audit_note: str = Field(min_length=1)
    legal_reference: LegalReferenceComponents | None = None
    quantity: QuantityComponents | None = None

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, value: object) -> object:
        return SemanticMentionKind(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_kind_details(self) -> Self:
        if self.char_start >= self.char_end:
            raise ValueError("reference mention requires char_start < char_end")
        if (self.kind == SemanticMentionKind.LEGAL_REFERENCE) != (self.legal_reference is not None):
            raise ValueError("legal details are required exactly for legal references")
        if (self.kind == SemanticMentionKind.QUANTITY) != (self.quantity is not None):
            raise ValueError("quantity details are required exactly for quantities")
        if (
            self.kind
            in {
                SemanticMentionKind.DOCUMENT_IDENTIFIER,
                SemanticMentionKind.TEMPORAL_EXPRESSION,
            }
            and self.normalized_value is None
        ):
            raise ValueError("document identifiers and temporal expressions require normalization")
        if self.kind == SemanticMentionKind.TEMPORAL_EXPRESSION:
            assert self.normalized_value is not None
            try:
                date.fromisoformat(self.normalized_value)
            except ValueError as exc:
                raise ValueError("normalized temporal value must be an ISO calendar date") from exc
        if self.kind == SemanticMentionKind.DOCUMENT_IDENTIFIER and self.normalized_value:
            expected = unicodedata.normalize("NFKC", self.raw_text).upper().replace(" ", "")
            if self.normalized_value != expected:
                raise ValueError("normalized document identifier differs from audited raw text")
        if self.legal_reference is not None:
            legal = self.legal_reference
            if legal.article is None:
                raise ValueError("audited legal reference requires an article component")
            if (legal.instrument_number_raw is None) != (
                legal.instrument_number_normalized is None
            ):
                raise ValueError("legal instrument raw and normalized numbers must be paired")
            if legal.instrument_number_raw is not None:
                expected_number = (
                    unicodedata.normalize("NFKC", legal.instrument_number_raw)
                    .upper()
                    .replace(" ", "")
                )
                if (
                    legal.instrument_number_raw not in self.raw_text
                    or legal.instrument_number_normalized != expected_number
                ):
                    raise ValueError("legal instrument number differs from audited raw text")
        if self.quantity is not None:
            if (
                self.quantity.raw_value not in self.raw_text
                or self.quantity.raw_unit not in self.raw_text
            ):
                raise ValueError("quantity components differ from audited raw text")
            expected_quantity = (
                f"{self.quantity.normalized_numeric_value} {self.quantity.normalized_unit}"
                if self.quantity.normalized_numeric_value is not None
                and self.quantity.normalized_unit is not None
                else None
            )
            if self.normalized_value != expected_quantity:
                raise ValueError("normalized quantity differs from audited quantity components")
        return self


class SemanticReferenceStatement(SemanticEvaluationModel):
    reference_statement_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    exact_evidence_excerpt: str = Field(min_length=1)
    audit_note: str = Field(min_length=1)


class VisualAssistanceReference(SemanticEvaluationModel):
    reference_id: str = Field(min_length=1)
    page_index: NonNegativeInt
    bbox_normalized_1000: tuple[int, int, int, int]
    task_type: VLMTaskType
    need_reason: str = Field(min_length=1)
    evidence_note: str = Field(min_length=1)

    @field_validator("bbox_normalized_1000", mode="before")
    @classmethod
    def coerce_bbox(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("task_type", mode="before")
    @classmethod
    def coerce_task(cls, value: object) -> object:
        return VLMTaskType(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def validate_bbox(self) -> Self:
        x0, y0, x1, y1 = self.bbox_normalized_1000
        if not (0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000):
            raise ValueError("visual-assistance bbox must be ordered normalized_1000 coordinates")
        return self


class SemanticReferencePage(SemanticEvaluationModel):
    page_index: NonNegativeInt
    pdf_page_number_1_based: int = Field(gt=0)
    render_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    statements: tuple[SemanticReferenceStatement, ...] = ()
    mentions: tuple[SemanticReferenceMention, ...] = ()
    visual_assistance: tuple[VisualAssistanceReference, ...] = ()

    @field_validator("statements", "mentions", "visual_assistance", mode="before")
    @classmethod
    def coerce_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_occurrences(self) -> Self:
        if self.pdf_page_number_1_based != self.page_index + 1:
            raise ValueError("reference PDF page number must equal page_index + 1")
        statements = {item.reference_statement_id: item for item in self.statements}
        mention_ids = {item.reference_mention_id for item in self.mentions}
        visual_ids = {item.reference_id for item in self.visual_assistance}
        if len(statements) != len(self.statements):
            raise ValueError("duplicate reference statement ID")
        if len(mention_ids) != len(self.mentions):
            raise ValueError("duplicate reference mention ID")
        if len(visual_ids) != len(self.visual_assistance):
            raise ValueError("duplicate visual-assistance reference ID")
        for mention in self.mentions:
            statement = statements.get(mention.reference_statement_id)
            if statement is None:
                raise ValueError("reference mention points to a missing statement")
            if mention.page_index != self.page_index or statement.page_index != self.page_index:
                raise ValueError("reference occurrence is assigned to the wrong page")
            excerpt = statement.exact_evidence_excerpt
            if mention.char_end > len(excerpt):
                raise ValueError("reference mention span exceeds audited excerpt")
            if excerpt[mention.char_start : mention.char_end] != mention.raw_text:
                raise ValueError("reference raw_text differs from audited excerpt span")
        if any(item.page_index != self.page_index for item in self.visual_assistance):
            raise ValueError("visual-assistance region is assigned to the wrong page")
        return self


class SemanticReferenceAnnotation(SemanticEvaluationModel):
    annotation_schema_version: Literal[2] = 2
    annotation_version: Literal["v2"] = "v2"
    annotator_type: Literal["ai_visual_audit"]
    annotation_method: Literal["visual_pdf_semantic_reaudit"]
    document_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prior_physical_ir_exposure: Literal[True]
    prior_structural_ir_exposure: Literal[True]
    prior_semantic_extractor_exposure: Literal[True]
    semantic_extractor_candidates_seen: Literal[True]
    semantic_extractor_output_used_as_reference_truth: Literal[False]
    independent_or_blind_ground_truth: Literal[False]
    pages: tuple[SemanticReferencePage, ...]

    @field_validator("pages", mode="before")
    @classmethod
    def coerce_pages(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_annotation(self) -> Self:
        page_indexes = [page.page_index for page in self.pages]
        if len(set(page_indexes)) != len(page_indexes):
            raise ValueError("duplicate audited page index")
        statement_ids = [
            statement.reference_statement_id for page in self.pages for statement in page.statements
        ]
        mention_ids = [
            mention.reference_mention_id for page in self.pages for mention in page.mentions
        ]
        if len(set(statement_ids)) != len(statement_ids):
            raise ValueError("duplicate reference statement ID across annotation")
        if len(set(mention_ids)) != len(mention_ids):
            raise ValueError("duplicate reference mention ID across annotation")
        return self


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def generate_semantic_reference_candidates(root: Path) -> dict[str, Any]:
    """Export non-authoritative parser/extractor suggestions for an explicit visual audit."""
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
    audited_pages_by_document: dict[str, set[int]] = {}
    for source_path in sorted((root / "data/structural_annotations").glob("*.v4.json")):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if "audited_pages" in source:
            audited_pages_by_document[str(source["document_id"])] = {
                int(page["page_index"]) for page in source["audited_pages"]
            }
    records: list[dict[str, Any]] = []
    for document_id, representations in sorted(candidates_by_document.items()):
        audited_pages = audited_pages_by_document[document_id]
        for parser, semantic in sorted(representations):
            for statement in semantic.statements:
                page_index = statement.evidence_anchors[0].page_index
                if page_index not in audited_pages:
                    continue
                records.append(
                    {
                        "document_id": document_id,
                        "parser": parser,
                        "page_index": page_index,
                        "runtime_statement_id": statement.id,
                        "runtime_structural_node_id": statement.structural_node_id,
                        "candidate_excerpt": statement.text,
                        "mentions": [
                            {
                                "runtime_mention_id": mention.id,
                                "kind": mention.kind.value,
                                "raw_text": mention.raw_text,
                                "normalized_value": mention.normalized_value,
                            }
                            for mention in semantic.mentions
                            if mention.statement_id == statement.id
                        ],
                    }
                )
    return {
        "candidate_schema_version": 1,
        "authoritative_reference_truth": False,
        "warning": "Parser/Semantic extractor suggestions only. Evaluation must never load this artifact as reference truth.",
        "fixed_page_count": sum(len(value) for value in audited_pages_by_document.values()),
        "records": records,
    }


def write_semantic_reference_candidates(root: Path) -> dict[str, Any]:
    candidates = generate_semantic_reference_candidates(root)
    output = root / "data/semantic_candidates/reference_candidates.v1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(candidates))
    return candidates


def load_semantic_reference_annotations(root: Path) -> dict[str, SemanticReferenceAnnotation]:
    """Load only committed authoritative v2 files; never generate or load candidates."""
    result: dict[str, SemanticReferenceAnnotation] = {}
    source_paths = sorted((root / "data/semantic_annotations").glob("*.v2.json"))
    for source_path in source_paths:
        if source_path.name.startswith("reference_semantic_audit"):
            continue
        annotation = SemanticReferenceAnnotation.model_validate_json(source_path.read_bytes())
        if annotation.document_id in result:
            raise ValueError("duplicate Semantic reference v2 document")
        result[annotation.document_id] = annotation
    if len(result) != 6 or sum(len(value.pages) for value in result.values()) != 46:
        raise ValueError("Semantic reference v2 must contain the exact six-document, 46-page set")
    fixed_identities: dict[str, tuple[str, str, set[int]]] = {}
    for path in sorted((root / "data/structural_annotations").glob("*.v4.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if "audited_pages" in value:
            fixed_identities[str(value["document_id"])] = (
                str(value["version_id"]),
                str(value["source_sha256"]),
                {int(page["page_index"]) for page in value["audited_pages"]},
            )
    for document_id, annotation in result.items():
        identity = fixed_identities.get(document_id)
        if identity is None:
            raise ValueError("Semantic reference v2 document is absent from fixed #008 audit")
        version_id, source_sha256, fixed_pages = identity
        if (
            annotation.version_id != version_id
            or annotation.source_artifact_sha256 != source_sha256
        ):
            raise ValueError("Semantic reference v2 source identity differs from fixed #008 audit")
        if {page.page_index for page in annotation.pages} != fixed_pages:
            raise ValueError("Semantic reference v2 page set differs from fixed #008 audit")
    audit_path = root / "data/semantic_annotations/reference_semantic_audit.v2.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("audit_schema_version") != 2
        or audit.get("reference_version") != "v2"
        or audit.get("reference_schema") != 2
        or audit.get("pages") != 46
    ):
        raise ValueError("Semantic reference audit metadata is missing or incompatible")
    expected_page_records = {
        (annotation.document_id, page.page_index): (
            page.render_sha256,
            len(page.statements),
            Counter(mention.kind.value for mention in page.mentions),
            len(page.visual_assistance),
        )
        for annotation in result.values()
        for page in annotation.pages
    }
    observed_page_records: dict[tuple[str, int], tuple[str, int, Counter[str], int]] = {}
    for record in audit.get("page_records", []):
        key = (str(record["document_id"]), int(record["page_index"]))
        if key in observed_page_records or record.get("visually_reaudited") is not True:
            raise ValueError("Semantic reference audit has duplicate or unaudited page record")
        observed_page_records[key] = (
            str(record["render_sha256"]),
            int(record["statement_count"]),
            Counter(record["mention_count_by_kind"]),
            int(record["visual_assistance_region_count"]),
        )
    if observed_page_records != expected_page_records:
        raise ValueError("Semantic reference audit page records differ from reference v2")
    mention_counts = Counter(
        mention.kind.value
        for annotation in result.values()
        for page in annotation.pages
        for mention in page.mentions
    )
    if Counter(audit.get("mentions_by_kind", {})) != mention_counts:
        raise ValueError("Semantic reference audit mention counts differ from reference v2")
    if audit.get("statements") != sum(
        len(page.statements) for annotation in result.values() for page in annotation.pages
    ):
        raise ValueError("Semantic reference audit statement count differs from reference v2")
    if audit.get("visual_assistance_regions") != sum(
        len(page.visual_assistance) for annotation in result.values() for page in annotation.pages
    ):
        raise ValueError("Semantic reference audit visual-region count differs from reference v2")
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


def _mention_key(page: int, kind: str, raw: str) -> tuple[int, str, str]:
    return page, kind, raw


def _prediction_order(item: SemanticMention) -> tuple[int, str, int, int, str]:
    anchor = item.evidence_anchors[0]
    return (
        anchor.page_index,
        item.statement_id,
        int(getattr(anchor, "char_start", -1)),
        int(getattr(anchor, "char_end", -1)),
        item.id,
    )


def _evaluate_document(
    document: SemanticDocument, annotation: SemanticReferenceAnnotation
) -> dict[str, Any]:
    audited_pages = {page.page_index for page in annotation.pages}
    reference_statements = [statement for page in annotation.pages for statement in page.statements]
    reference_statement_by_id = {
        statement.reference_statement_id: statement for statement in reference_statements
    }
    reference_statement_order = {
        statement.reference_statement_id: index
        for index, statement in enumerate(reference_statements)
    }
    reference_mentions = [mention for page in annotation.pages for mention in page.mentions]
    predictions = [
        mention
        for mention in document.mentions
        if mention.evidence_anchors[0].page_index in audited_pages
    ]
    references_by_key: dict[tuple[int, str, str], list[SemanticReferenceMention]] = {}
    predictions_by_key: dict[tuple[int, str, str], list[SemanticMention]] = {}
    for reference_item in reference_mentions:
        references_by_key.setdefault(
            _mention_key(
                reference_item.page_index,
                reference_item.kind.value,
                reference_item.raw_text,
            ),
            [],
        ).append(reference_item)
    for prediction_item in predictions:
        predictions_by_key.setdefault(
            _mention_key(
                prediction_item.evidence_anchors[0].page_index,
                prediction_item.kind.value,
                prediction_item.raw_text,
            ),
            [],
        ).append(prediction_item)
    for reference_values in references_by_key.values():
        reference_values.sort(
            key=lambda item: (
                reference_statement_order[item.reference_statement_id],
                item.char_start,
                item.char_end,
                item.reference_mention_id,
            )
        )
    for prediction_values in predictions_by_key.values():
        prediction_values.sort(key=_prediction_order)
    pairs: list[tuple[SemanticReferenceMention, SemanticMention]] = []
    unmatched_references: list[SemanticReferenceMention] = []
    unmatched_predictions: list[SemanticMention] = []
    for key in sorted(set(references_by_key) | set(predictions_by_key)):
        references = references_by_key.get(key, [])
        predicted = predictions_by_key.get(key, [])
        common = min(len(references), len(predicted))
        pairs.extend(zip(references[:common], predicted[:common], strict=True))
        unmatched_references.extend(references[common:])
        unmatched_predictions.extend(predicted[common:])
    statements_by_id = {statement.id: statement for statement in document.statements}
    kinds: dict[str, dict[str, int | float | None]] = {}
    for kind in SemanticMentionKind:
        kind_pairs = sum(reference.kind == kind for reference, _ in pairs)
        kinds[kind.value] = _metric(
            kind_pairs,
            sum(item.kind == kind for item in unmatched_predictions),
            sum(item.kind == kind for item in unmatched_references),
        )
    overall = _metric(len(pairs), len(unmatched_predictions), len(unmatched_references))
    exact_span = 0
    for reference, prediction in pairs:
        anchor = prediction.evidence_anchors[0]
        statement = statements_by_id[prediction.statement_id]
        statement_anchor = statement.evidence_anchors[0]
        if (
            anchor.anchor_type == "text"
            and statement_anchor.anchor_type == "text"
            and statement.text
            == reference_statement_by_id[reference.reference_statement_id].exact_evidence_excerpt
            and anchor.char_start - statement_anchor.char_start == reference.char_start
            and anchor.char_end - statement_anchor.char_start == reference.char_end
        ):
            exact_span += 1
    reference_statement_counter = Counter(
        (item.page_index, item.exact_evidence_excerpt) for item in reference_statements
    )
    prediction_statement_counter = Counter(
        (item.evidence_anchors[0].page_index, item.text)
        for item in document.statements
        if item.evidence_anchors[0].page_index in audited_pages
    )
    statement_hits = sum((reference_statement_counter & prediction_statement_counter).values())
    normalized_pairs = [
        (reference, prediction)
        for reference, prediction in pairs
        if reference.normalized_value is not None
    ]
    normalized_exact = sum(
        reference.normalized_value == prediction.normalized_value
        for reference, prediction in normalized_pairs
    )
    legal_pairs = [
        (reference, prediction)
        for reference, prediction in pairs
        if reference.kind == SemanticMentionKind.LEGAL_REFERENCE
    ]
    legal_fields = (
        "instrument_type",
        "instrument_number_raw",
        "instrument_number_normalized",
        "article",
        "clause",
        "point",
        "scope",
    )
    legal_field_results: dict[str, dict[str, int | float | None]] = {}
    for field in legal_fields:
        exact = sum(
            reference.legal_reference is not None
            and prediction.legal_reference is not None
            and getattr(reference.legal_reference, field)
            == getattr(prediction.legal_reference, field)
            for reference, prediction in legal_pairs
        )
        legal_field_results[field] = {
            "exact": exact,
            "eligible": len(legal_pairs),
            "rate": exact / len(legal_pairs) if legal_pairs else None,
        }
    legal_exact = sum(
        reference.legal_reference == prediction.legal_reference
        for reference, prediction in legal_pairs
    )
    return {
        "mention_metrics": overall,
        "mention_metrics_by_kind": kinds,
        "exact_evidence_span_match": {
            "exact": exact_span,
            "eligible": len(pairs),
            "rate": exact_span / len(pairs) if pairs else None,
        },
        "normalized_value_exact_match": {
            "exact": normalized_exact,
            "eligible": len(normalized_pairs),
            "rate": normalized_exact / len(normalized_pairs) if normalized_pairs else None,
        },
        "legal_reference_component_exact_match": {
            "exact": legal_exact,
            "eligible": len(legal_pairs),
            "rate": legal_exact / len(legal_pairs) if legal_pairs else None,
            "by_component": legal_field_results,
        },
        "statement_evidence_coverage": {
            "covered": statement_hits,
            "reference": len(reference_statements),
            "rate": (statement_hits / len(reference_statements) if reference_statements else None),
        },
        "false_positive_trace": [
            {
                "document_id": document.document_id,
                "page_index": item.evidence_anchors[0].page_index,
                "semantic_kind": item.kind.value,
                "raw_evidence": item.raw_text,
                "normalized_value": item.normalized_value,
                "runtime_mention_id": item.id,
                "reason": "unresolved",
            }
            for item in sorted(unmatched_predictions, key=_prediction_order)
        ],
        "false_negative_trace": [
            {
                "document_id": document.document_id,
                "page_index": item.page_index,
                "semantic_kind": item.kind.value,
                "raw_evidence": item.raw_text,
                "normalized_value": item.normalized_value,
                "reference_mention_id": item.reference_mention_id,
                "reason": "unresolved",
            }
            for item in sorted(unmatched_references, key=lambda value: value.reference_mention_id)
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

    legal_components: dict[str, Any] = {}
    for field in (
        "instrument_type",
        "instrument_number_raw",
        "instrument_number_normalized",
        "article",
        "clause",
        "point",
        "scope",
    ):
        exact = sum(
            int(item["legal_reference_component_exact_match"]["by_component"][field]["exact"])
            for item in results
        )
        eligible = sum(
            int(item["legal_reference_component_exact_match"]["by_component"][field]["eligible"])
            for item in results
        )
        legal_components[field] = {
            "exact": exact,
            "eligible": eligible,
            "rate": exact / eligible if eligible else None,
        }
    legal_overall: dict[str, Any] = rate(
        "legal_reference_component_exact_match", "exact", "eligible"
    )
    legal_overall["by_component"] = legal_components
    return {
        "mention_metrics": overall,
        "mention_metrics_by_kind": by_kind,
        "exact_evidence_span_match": rate("exact_evidence_span_match", "exact", "eligible"),
        "normalized_value_exact_match": rate("normalized_value_exact_match", "exact", "eligible"),
        "legal_reference_component_exact_match": legal_overall,
        "statement_evidence_coverage": rate("statement_evidence_coverage", "covered", "reference"),
    }


def _task_family(task: VLMTaskType) -> str:
    if task in {VLMTaskType.REGION_TRANSCRIPTION, VLMTaskType.OCR_RECOVERY}:
        return "transcription"
    if task in {VLMTaskType.TABLE_TEXT_RECOVERY, VLMTaskType.TABLE_HEADER_RECOVERY}:
        return "table"
    return "figure"


def _bbox_overlap(
    reference: tuple[int, int, int, int], prediction: BoundingBox
) -> tuple[float, float]:
    rx0, ry0, rx1, ry1 = reference
    px0, py0, px1, py1 = prediction.x0, prediction.y0, prediction.x1, prediction.y1
    intersection = max(0.0, min(rx1, px1) - max(rx0, px0)) * max(0.0, min(ry1, py1) - max(ry0, py0))
    reference_area = float((rx1 - rx0) * (ry1 - ry0))
    prediction_area = max(0.0, px1 - px0) * max(0.0, py1 - py0)
    union = reference_area + prediction_area - intersection
    return (
        intersection / union if union else 0.0,
        intersection / reference_area if reference_area else 0.0,
    )


def _evaluate_selector(
    selection: VLMSelectionResult, annotation: SemanticReferenceAnnotation
) -> dict[str, Any]:
    references = sorted(
        (item for page in annotation.pages for item in page.visual_assistance),
        key=lambda item: item.reference_id,
    )
    predictions = sorted(selection.requests, key=lambda item: item.request_id)
    available = set(range(len(predictions)))
    matched: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for reference in references:
        candidates: list[tuple[float, float, int]] = []
        for index in available:
            prediction = predictions[index]
            if prediction.page_index != reference.page_index or _task_family(
                prediction.task_type
            ) != _task_family(reference.task_type):
                continue
            iou, containment = _bbox_overlap(reference.bbox_normalized_1000, prediction.bbox)
            if iou >= 0.25 or containment >= 0.50:
                candidates.append((iou, containment, index))
        if not candidates:
            false_negatives.append(
                {
                    "reference_id": reference.reference_id,
                    "page_index": reference.page_index,
                    "task_family": _task_family(reference.task_type),
                }
            )
            continue
        _, _, index = max(
            candidates,
            key=lambda item: (item[0], item[1], predictions[item[2]].request_id),
        )
        available.remove(index)
        prediction = predictions[index]
        iou, containment = _bbox_overlap(reference.bbox_normalized_1000, prediction.bbox)
        matched.append(
            {
                "reference_id": reference.reference_id,
                "request_id": prediction.request_id,
                "page_index": reference.page_index,
                "task_family": _task_family(reference.task_type),
                "iou": iou,
                "reference_containment": containment,
            }
        )
    false_positives = [
        {
            "request_id": predictions[index].request_id,
            "page_index": predictions[index].page_index,
            "task_family": _task_family(predictions[index].task_type),
        }
        for index in sorted(available)
    ]
    budget_exhaustion = Counter(
        item.reason.value
        for item in selection.non_selections
        if item.reason.value in {"page_budget_exhausted", "document_budget_exhausted"}
    )
    eligible_selection_reasons = Counter(item.selection_reason.value for item in selection.requests)
    eligible_selection_reasons.update(
        item.selection_reason.value for item in selection.non_selections
    )
    return {
        **_metric(len(matched), len(false_positives), len(false_negatives)),
        "matching_protocol": {
            "unit": "one-to-one audited region/task-family",
            "iou_threshold": 0.25,
            "reference_containment_threshold": 0.50,
        },
        "matched_regions": matched,
        "false_positive_regions": false_positives,
        "false_negative_regions": false_negatives,
        "selected_requests": len(predictions),
        "eligible_requests": len(predictions) + len(selection.non_selections),
        "selected_requests_per_page": dict(
            sorted(Counter(item.page_index for item in predictions).items())
        ),
        "selection_reason_histogram": dict(sorted(eligible_selection_reasons.items())),
        "selected_selection_reason_histogram": dict(
            sorted(Counter(item.selection_reason.value for item in predictions).items())
        ),
        "non_selection_reason_histogram": dict(
            sorted(Counter(item.reason.value for item in selection.non_selections).items())
        ),
        "budget_exhaustion_counts": dict(sorted(budget_exhaustion.items())),
    }


def _aggregate_selector(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric = _metric(
        sum(int(item["tp"]) for item in results),
        sum(int(item["fp"]) for item in results),
        sum(int(item["fn"]) for item in results),
    )
    selected = sum(int(item["selected_requests"]) for item in results)
    eligible = sum(int(item["eligible_requests"]) for item in results)
    selected_by_document: Counter[str] = Counter()
    for item in results:
        selected_by_document[str(item["document_id"])] += int(item["selected_requests"])
    return {
        **metric,
        "matching_protocol": {
            "unit": "one-to-one audited region/task-family per parser representation",
            "iou_threshold": 0.25,
            "reference_containment_threshold": 0.50,
        },
        "selected_requests": selected,
        "eligible_requests": eligible,
        "requests_per_parser_representation": selected / len(results) if results else None,
        "selected_requests_per_document": dict(sorted(selected_by_document.items())),
        "selection_reason_histogram": dict(
            sorted(
                sum(
                    (Counter(item["selection_reason_histogram"]) for item in results),
                    Counter(),
                ).items()
            )
        ),
        "selected_selection_reason_histogram": dict(
            sorted(
                sum(
                    (Counter(item["selected_selection_reason_histogram"]) for item in results),
                    Counter(),
                ).items()
            )
        ),
        "non_selection_reason_histogram": dict(
            sorted(
                sum(
                    (Counter(item["non_selection_reason_histogram"]) for item in results),
                    Counter(),
                ).items()
            )
        ),
        "budget_exhaustion_counts": dict(
            sorted(
                sum(
                    (Counter(item["budget_exhaustion_counts"]) for item in results),
                    Counter(),
                ).items()
            )
        ),
        "per_representation": results,
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
    selector_evaluations: list[dict[str, Any]] = []
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
        selection = select_visual_evidence(
            physical, structural=structural, budget=SelectionBudget()
        )
        selector_evaluation = _evaluate_selector(selection, annotations[document_id])
        selector_evaluation.update({"document_id": document_id, "parser": parser})
        selector_evaluations.append(selector_evaluation)
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
    selector_metric = _aggregate_selector(selector_evaluations)
    aggregate = _aggregate(evaluations)
    by_parser = {
        parser: _aggregate([x for x in evaluations if x["parser"] == parser])
        for parser in ("marker", "mineru")
    }
    evidence: dict[str, Any] = {
        "validation_schema_version": 2,
        "validation_protocol": "semantic_ir_v1_fixed_46_page_visual_reference_v2",
        "semantic_ir_version": 1,
        "semantic_profile": "vi_legal_planning_semantic_v1",
        "runtime_inputs": [
            "PhysicalDocumentV1",
            "StructuralDocument v1",
            "validated VLMObservation records",
        ],
        "reference": {
            "version": "v2",
            "schema": 2,
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
            "requests_per_source_document": selector_metric["selected_requests"] / 6,
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

This evaluates ten parser representations over the exact fixed 46-page #008 audit set. Reference v2/schema 2 was rebuilt from a visual review of rendered source-PDF pages. It is AI-authored, discloses prior Physical IR, Structural IR, and Semantic extractor exposure, and is not human, blind, or independent ground truth. Runtime extractor output is never loaded as reference truth.

## Reference provenance

Semantic reference v2/schema 2 covers six documents and 46 fixed pages. Each occurrence has its own audited statement/mention identity based on the source PDF, page, excerpt occurrence, and character span—never parser block IDs or runtime Semantic IDs. Candidate suggestions were visible during audit, so metrics remain diagnostic rather than an independent estimate of real-world accuracy. Counts by kind: `{json.dumps(evidence["reference"]["mention_counts_by_kind"], ensure_ascii=False, sort_keys=True)}`.

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

- Selector audited-region/task-family P/R/F1: {selector["precision"]!r} / {selector["recall"]!r} / {selector["f1"]!r}.
- Selected requests: {selector["selected_requests"]}; eligible requests: {selector["eligible_requests"]}; requests/representation: {selector["requests_per_parser_representation"]!r}; requests/source document across retained representations: {selector["requests_per_source_document"]!r}.
- Matching is deterministic and one-to-one, requiring the same page and task family plus IoU >= 0.25 or audited-reference containment >= 0.50.
- Eligible selection reasons: `{json.dumps(selector["selection_reason_histogram"], sort_keys=True)}`; selected reasons: `{json.dumps(selector["selected_selection_reason_histogram"], sort_keys=True)}`; non-selection reasons: `{json.dumps(selector["non_selection_reason_histogram"], sort_keys=True)}`; budget exhaustion: `{json.dumps(selector["budget_exhaustion_counts"], sort_keys=True)}`.
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

Statements are exact direct-content spans, not paraphrases. Mentions use a controlled deterministic Vietnamese taxonomy. Detection matches occurrences one-to-one by page, kind, and raw text; normalized and legal-component correctness are measured only after raw occurrence matching. Failures remain `unresolved` unless evidence establishes a cause. Parser aggregates are representation-weighted: six source documents produce ten parser representations. Selector labels are audited PDF regions with task families, but remain AI-authored diagnostic evidence. No RAG, KG, cross-document citation resolution, or entity resolution is implemented.
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
    "generate_semantic_reference_candidates",
    "load_semantic_reference_annotations",
    "normalized_edit_distance",
    "render_semantic_report",
    "write_semantic_ir_v1_validation",
    "write_semantic_reference_candidates",
]
