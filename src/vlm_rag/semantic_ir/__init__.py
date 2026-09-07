"""Domain Semantic IR v1 public API."""

from vlm_rag.semantic_ir.builder import (
    SemanticSourceIntegrityError,
    build_semantic_document,
    validate_semantic_sources,
)
from vlm_rag.semantic_ir.extraction import MentionCandidate, extract_mention_candidates
from vlm_rag.semantic_ir.models import (
    LegalReferenceComponents,
    LegalReferenceScope,
    ObjectSemanticAnchor,
    QuantityComponents,
    SemanticDocument,
    SemanticEvidenceAnchor,
    SemanticMention,
    SemanticMentionKind,
    SemanticProfile,
    SemanticProvenanceKind,
    SemanticStatement,
    TextSemanticAnchor,
    VisualObservation,
    VisualSemanticAnchor,
)
from vlm_rag.semantic_ir.serialization import (
    SemanticIRSerializationError,
    semantic_document_from_dict,
    semantic_document_from_json,
    semantic_document_to_dict,
    semantic_document_to_json,
)

__all__ = [
    "LegalReferenceComponents",
    "LegalReferenceScope",
    "MentionCandidate",
    "ObjectSemanticAnchor",
    "QuantityComponents",
    "SemanticDocument",
    "SemanticEvidenceAnchor",
    "SemanticIRSerializationError",
    "SemanticMention",
    "SemanticMentionKind",
    "SemanticProfile",
    "SemanticProvenanceKind",
    "SemanticSourceIntegrityError",
    "SemanticStatement",
    "TextSemanticAnchor",
    "VisualObservation",
    "VisualSemanticAnchor",
    "build_semantic_document",
    "extract_mention_candidates",
    "semantic_document_from_dict",
    "semantic_document_from_json",
    "semantic_document_to_dict",
    "semantic_document_to_json",
    "validate_semantic_sources",
]
