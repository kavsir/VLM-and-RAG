"""Deterministic selective-VLM candidate policy and budgets."""

import hashlib
import json
import unicodedata
from dataclasses import dataclass

from vlm_rag.physical_ir.models import BlockDisposition
from vlm_rag.physical_ir.v1 import (
    BlockKindV1,
    PhysicalBlockV1,
    PhysicalDocumentV1,
    TextExtractionMethod,
    VisualAssetStorage,
)
from vlm_rag.vlm.models import (
    NonSelectionReason,
    RetainedVisualAsset,
    SelectionReason,
    VisualEvidenceRequest,
    VLMNonSelection,
    VLMSelectionResult,
    VLMTaskType,
)


@dataclass(frozen=True, slots=True)
class SelectionBudget:
    max_requests_per_page: int = 2
    max_requests_per_document: int = 12

    def __post_init__(self) -> None:
        if self.max_requests_per_page < 0 or self.max_requests_per_document < 0:
            raise ValueError("selection budgets must be non-negative")


_PRIORITY = {
    SelectionReason.EMPTY_OR_TEXT_DEFICIENT_VISUAL_REGION: 10,
    SelectionReason.TABLE_MISSING_STRUCTURE: 20,
    SelectionReason.OCR_CORRUPTION_SIGNAL: 30,
    SelectionReason.OCR_LOW_CONFIDENCE: 40,
    SelectionReason.UNKNOWN_TEXT_EXTRACTION: 50,
    SelectionReason.FIGURE_OR_IMAGE: 60,
}


def _corruption_ratio(text: str) -> float:
    if not text:
        return 1.0
    suspicious = sum(
        character == "\ufffd"
        or (unicodedata.category(character) == "Cc" and character not in "\n\r\t")
        for character in text
    )
    return suspicious / len(text)


def _candidate(block: PhysicalBlockV1) -> tuple[SelectionReason, VLMTaskType] | None:
    if block.disposition != BlockDisposition.CONTENT:
        return None
    if block.kind in {BlockKindV1.FIGURE, BlockKindV1.IMAGE}:
        if not block.text:
            return (
                SelectionReason.EMPTY_OR_TEXT_DEFICIENT_VISUAL_REGION,
                VLMTaskType.FIGURE_OR_IMAGE_OBSERVATION,
            )
        return SelectionReason.FIGURE_OR_IMAGE, VLMTaskType.FIGURE_OR_IMAGE_OBSERVATION
    if block.kind == BlockKindV1.TABLE and block.table_structure is None:
        return SelectionReason.TABLE_MISSING_STRUCTURE, VLMTaskType.TABLE_TEXT_RECOVERY
    if _corruption_ratio(block.text) >= 0.02:
        return SelectionReason.OCR_CORRUPTION_SIGNAL, VLMTaskType.OCR_RECOVERY
    if block.text_extraction is not None:
        if (
            block.text_extraction.method == TextExtractionMethod.OCR
            and block.text_extraction.confidence is not None
            and block.text_extraction.confidence < 0.75
        ):
            return SelectionReason.OCR_LOW_CONFIDENCE, VLMTaskType.OCR_RECOVERY
        if block.text_extraction.method == TextExtractionMethod.UNKNOWN:
            return SelectionReason.UNKNOWN_TEXT_EXTRACTION, VLMTaskType.REGION_TRANSCRIPTION
    return None


def _retained_asset(block: PhysicalBlockV1) -> RetainedVisualAsset | None:
    asset = block.visual_asset
    if asset is None or asset.storage_kind != VisualAssetStorage.RELATIVE_FILE:
        return None
    assert asset.relative_path is not None
    assert asset.sha256 is not None and asset.byte_size is not None and asset.media_type is not None
    return RetainedVisualAsset(
        relative_path=asset.relative_path,
        sha256=asset.sha256,
        byte_size=asset.byte_size,
        media_type=asset.media_type,
    )


def _request_id(
    document: PhysicalDocumentV1,
    block: PhysicalBlockV1,
    reason: SelectionReason,
    task: VLMTaskType,
) -> str:
    payload = json.dumps(
        [
            document.document_id,
            document.version_id,
            document.source_artifact_sha256,
            block.page_index,
            block.id,
            reason.value,
            task.value,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"vlm-request-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def select_visual_evidence(
    document: PhysicalDocumentV1, *, budget: SelectionBudget | None = None
) -> VLMSelectionResult:
    """Select requests before invocation with stable priorities and budget traces."""
    budget = budget or SelectionBudget()
    candidates: list[VisualEvidenceRequest] = []
    for page in document.pages:
        for block in page.blocks:
            evidence = _candidate(block)
            if evidence is None:
                continue
            reason, task = evidence
            candidates.append(
                VisualEvidenceRequest(
                    request_id=_request_id(document, block, reason, task),
                    document_id=document.document_id,
                    version_id=document.version_id,
                    source_artifact_sha256=document.source_artifact_sha256,
                    page_index=block.page_index,
                    bbox=block.bbox,
                    source_block_ids=(block.id,),
                    task_type=task,
                    selection_reason=reason,
                    priority=_PRIORITY[reason],
                    retained_visual_asset=_retained_asset(block),
                )
            )
    candidates.sort(
        key=lambda item: (item.priority, item.page_index, item.source_block_ids, item.request_id)
    )
    selected: list[VisualEvidenceRequest] = []
    non_selected: list[VLMNonSelection] = []
    per_page: dict[int, int] = {}
    for request in candidates:
        if len(selected) >= budget.max_requests_per_document:
            non_selection_reason = NonSelectionReason.DOCUMENT_BUDGET_EXHAUSTED
        elif per_page.get(request.page_index, 0) >= budget.max_requests_per_page:
            non_selection_reason = NonSelectionReason.PAGE_BUDGET_EXHAUSTED
        else:
            selected.append(request)
            per_page[request.page_index] = per_page.get(request.page_index, 0) + 1
            continue
        non_selected.append(
            VLMNonSelection(
                request_id=request.request_id,
                page_index=request.page_index,
                source_block_ids=request.source_block_ids,
                task_type=request.task_type,
                selection_reason=request.selection_reason,
                priority=request.priority,
                reason=non_selection_reason,
            )
        )
    return VLMSelectionResult(requests=tuple(selected), non_selections=tuple(non_selected))


__all__ = ["SelectionBudget", "select_visual_evidence"]
