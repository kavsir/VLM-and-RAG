"""Selective VLM evidence boundary public API."""

from vlm_rag.vlm.client import (
    ReplayEvidence,
    ReplayFixture,
    ReplayVLMClient,
    VLMClient,
    execute_vlm_request,
    load_replay_evidence,
)
from vlm_rag.vlm.evidence import (
    PDFCropRenderer,
    ResolvedVisualEvidence,
    VerifiedAssetResolver,
    VerifiedPDFResolver,
    VisualEvidenceIntegrityError,
    VisualEvidenceResolver,
)
from vlm_rag.vlm.models import (
    NonSelectionReason,
    RawVLMResponse,
    RenderSpecification,
    RetainedVisualAsset,
    SelectionReason,
    VisualEvidenceRequest,
    VLMNonSelection,
    VLMObservationRecord,
    VLMRequestRecord,
    VLMSelectionResult,
    VLMTaskType,
    canonical_request_sha256,
)
from vlm_rag.vlm.normalization import (
    VLMObservationError,
    normalize_vlm_response,
    to_visual_observation,
)
from vlm_rag.vlm.prompt import PROMPT_VERSION, build_visual_evidence_prompt
from vlm_rag.vlm.selector import SelectionBudget, select_visual_evidence

__all__ = [
    "PROMPT_VERSION",
    "NonSelectionReason",
    "PDFCropRenderer",
    "RawVLMResponse",
    "RenderSpecification",
    "ReplayEvidence",
    "ReplayFixture",
    "ReplayVLMClient",
    "ResolvedVisualEvidence",
    "RetainedVisualAsset",
    "SelectionBudget",
    "SelectionReason",
    "VLMClient",
    "VLMNonSelection",
    "VLMObservationError",
    "VLMObservationRecord",
    "VLMRequestRecord",
    "VLMSelectionResult",
    "VLMTaskType",
    "VerifiedAssetResolver",
    "VerifiedPDFResolver",
    "VisualEvidenceIntegrityError",
    "VisualEvidenceRequest",
    "VisualEvidenceResolver",
    "build_visual_evidence_prompt",
    "canonical_request_sha256",
    "execute_vlm_request",
    "load_replay_evidence",
    "normalize_vlm_response",
    "select_visual_evidence",
    "to_visual_observation",
]
