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
from vlm_rag.structural_ir.ordinals import (
    OrdinalSystem,
    article_ordinal_key,
    clause_ordinal_key,
    formal_container_ordinal_key,
    generic_decimal_ordinal_key,
    generic_letter_ordinal_key,
    point_ordinal_key,
    roman_ordinal_key,
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
from vlm_rag.structural_ir.vietnamese import (
    StructuralDiagnostic,
    VietnameseStructuralExtractor,
)

__all__ = [
    "AnchorRole",
    "OrdinalSystem",
    "PhysicalAnchor",
    "RecognitionEvidence",
    "RecognitionMethod",
    "StructuralDiagnostic",
    "StructuralDocument",
    "StructuralIRSerializationError",
    "StructuralNode",
    "StructuralNodeKind",
    "StructuralPhysicalIntegrityError",
    "StructuralProfile",
    "VietnameseStructuralExtractor",
    "article_ordinal_key",
    "clause_ordinal_key",
    "formal_container_ordinal_key",
    "generic_decimal_ordinal_key",
    "generic_letter_ordinal_key",
    "point_ordinal_key",
    "reconstruction_by_block",
    "roman_ordinal_key",
    "structural_document_from_dict",
    "structural_document_from_json",
    "structural_document_to_dict",
    "structural_document_to_json",
    "validate_against_physical",
]
