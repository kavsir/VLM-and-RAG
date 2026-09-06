"""Vietnamese Legal/Planning Structural IR v1 public API."""

from vlm_rag.structural_ir.models import (
    AnchorRole,
    PhysicalAnchor,
    RecognitionEvidence,
    RecognitionMethod,
    StructuralDocument,
    StructuralNode,
    StructuralNodeKind,
    StructuralProfile,
)
from vlm_rag.structural_ir.serialization import (
    StructuralIRSerializationError,
    structural_document_from_dict,
    structural_document_from_json,
    structural_document_to_dict,
    structural_document_to_json,
)
from vlm_rag.structural_ir.validation import (
    StructuralPhysicalIntegrityError,
    reconstruction_by_block,
    validate_against_physical,
)
from vlm_rag.structural_ir.vietnamese import VietnameseStructuralExtractor, ordinal_key

__all__ = [
    "AnchorRole",
    "PhysicalAnchor",
    "RecognitionEvidence",
    "RecognitionMethod",
    "StructuralDocument",
    "StructuralIRSerializationError",
    "StructuralNode",
    "StructuralNodeKind",
    "StructuralPhysicalIntegrityError",
    "StructuralProfile",
    "VietnameseStructuralExtractor",
    "ordinal_key",
    "reconstruction_by_block",
    "structural_document_from_dict",
    "structural_document_from_json",
    "structural_document_to_dict",
    "structural_document_to_json",
    "validate_against_physical",
]
