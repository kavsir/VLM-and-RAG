"""Fail-closed normalization of immutable raw VLM output."""

import hashlib
import json
from collections.abc import Mapping

from pydantic import ValidationError

from vlm_rag.semantic_ir.models import (
    SemanticProvenanceKind,
    VisualObservation,
    VisualSemanticAnchor,
)
from vlm_rag.vlm.evidence import ResolvedVisualEvidence
from vlm_rag.vlm.models import (
    RawVLMResponse,
    VLMObservationRecord,
    VLMRequestRecord,
    VLMTaskType,
    canonical_request_record_sha256,
)


class VLMObservationError(ValueError):
    """Raised when raw VLM output cannot become a supported observation."""


def normalize_vlm_response(
    request_record: VLMRequestRecord,
    response: RawVLMResponse,
    image: ResolvedVisualEvidence,
) -> VLMObservationRecord:
    request = request_record.request
    if hashlib.sha256(image.data).hexdigest() != image.sha256 or len(image.data) != image.byte_size:
        raise VLMObservationError("resolved visual evidence integrity mismatch")
    asset = request.retained_visual_asset
    if asset is not None and (
        image.sha256 != asset.sha256
        or image.byte_size != asset.byte_size
        or image.media_type != asset.media_type
    ):
        raise VLMObservationError("resolved visual evidence differs from retained asset identity")
    if response.finish_state != "completed":
        raise VLMObservationError("cannot normalize failed VLM response")
    if response.request_id != request.request_id:
        raise VLMObservationError("raw response/request identity mismatch")
    if response.model_id != request_record.model_id:
        raise VLMObservationError("raw response/request model identity mismatch")
    if response.provider_protocol != request_record.provider_protocol:
        raise VLMObservationError("raw response/request provider protocol mismatch")
    if (
        image.sha256 != request_record.image_sha256
        or image.byte_size != request_record.image_byte_size
    ):
        raise VLMObservationError("resolved image/request record identity mismatch")
    if (
        request.source_physical_ir_sha256 is None
        or request.source_structural_ir_sha256 is None
        or request.structural_node_id is None
        or request.structural_canonical_path is None
    ):
        raise VLMObservationError("semantic VLM normalization requires structural request context")
    try:
        raw: object = json.loads(response.raw_response)
    except json.JSONDecodeError as exc:
        raise VLMObservationError(f"malformed VLM JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise VLMObservationError("VLM JSON root must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != 1:
        raise VLMObservationError("unsupported VLM observation schema")
    if payload.get("request_id") != request.request_id:
        raise VLMObservationError("VLM payload/request identity mismatch")
    if payload.get("task_type") != request.task_type.value:
        raise VLMObservationError("VLM payload task mismatch")
    allowed = {"schema_version", "request_id", "task_type"}
    if request.task_type in {VLMTaskType.REGION_TRANSCRIPTION, VLMTaskType.OCR_RECOVERY}:
        allowed.add("transcription")
        payload_name = "transcription"
        provenance = SemanticProvenanceKind.VLM_TRANSCRIPTION
    elif request.task_type in {
        VLMTaskType.TABLE_TEXT_RECOVERY,
        VLMTaskType.TABLE_HEADER_RECOVERY,
    }:
        allowed.add("table_rows")
        payload_name = "table_rows"
        provenance = SemanticProvenanceKind.VLM_TABLE_RECOVERY
    else:
        allowed.add("description")
        payload_name = "description"
        provenance = SemanticProvenanceKind.VLM_VISUAL_OBSERVATION
    if set(payload) != allowed or payload_name not in payload:
        raise VLMObservationError("unsupported or missing VLM evidence claim")
    anchor = VisualSemanticAnchor(
        source_artifact_sha256=request.source_artifact_sha256,
        page_index=request.page_index,
        bbox=request.bbox,
        render_or_asset_sha256=image.sha256,
        byte_size=image.byte_size,
        media_type=image.media_type,
    )
    try:
        return VLMObservationRecord.model_validate(
            {
                "observation_schema_version": 2,
                "request_id": request.request_id,
                "task_type": request.task_type.value,
                "document_id": request.document_id,
                "version_id": request.version_id,
                "source_artifact_sha256": request.source_artifact_sha256,
                "source_physical_ir_sha256": request.source_physical_ir_sha256,
                "source_structural_ir_sha256": request.source_structural_ir_sha256,
                "structural_node_id": request.structural_node_id,
                "structural_canonical_path": request.structural_canonical_path,
                "request_record_sha256": canonical_request_record_sha256(request_record),
                "model_id": request_record.model_id,
                "provider_protocol": request_record.provider_protocol,
                "prompt_version": request_record.prompt_version,
                "raw_response_sha256": response.raw_response_sha256,
                "image_sha256": image.sha256,
                "source_block_ids": request.source_block_ids,
                payload_name: payload[payload_name],
                "evidence_anchor": anchor.model_dump(mode="json"),
                "provenance": provenance.value,
            }
        )
    except ValidationError as exc:
        raise VLMObservationError(f"invalid normalized VLM observation:\n{exc}") from exc


def to_visual_observation(record: VLMObservationRecord) -> VisualObservation:
    text = record.transcription if record.transcription is not None else record.description
    return VisualObservation(
        id=f"visual-observation-{record.request_id}",
        request_id=record.request_id,
        document_id=record.document_id,
        version_id=record.version_id,
        source_artifact_sha256=record.source_artifact_sha256,
        source_physical_ir_sha256=record.source_physical_ir_sha256,
        source_structural_ir_sha256=record.source_structural_ir_sha256,
        structural_node_id=record.structural_node_id,
        structural_canonical_path=record.structural_canonical_path,
        request_record_sha256=record.request_record_sha256,
        model_id=record.model_id,
        provider_protocol=record.provider_protocol,
        prompt_version=record.prompt_version,
        raw_response_sha256=record.raw_response_sha256,
        image_sha256=record.image_sha256,
        task_type=record.task_type.value,
        source_block_ids=record.source_block_ids,
        text=text,
        table_rows=record.table_rows,
        evidence_anchor=record.evidence_anchor,
        provenance=record.provenance,
    )


__all__ = ["VLMObservationError", "normalize_vlm_response", "to_visual_observation"]
