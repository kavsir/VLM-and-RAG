"""Physical Document Intermediate Representation (v0) models and serialization."""

from vlm_rag.physical_ir.models import (
    BlockDisposition,
    BlockKind,
    BlockProvenance,
    BoundingBox,
    PhysicalBlock,
    PhysicalDocument,
    PhysicalIRModel,
    PhysicalPage,
)
from vlm_rag.physical_ir.serialization import (
    PhysicalIRSerializationError,
    physical_document_from_dict,
    physical_document_from_json,
    physical_document_to_dict,
    physical_document_to_json,
)

__all__ = [
    "BlockDisposition",
    "BlockKind",
    "BlockProvenance",
    "BoundingBox",
    "PhysicalBlock",
    "PhysicalDocument",
    "PhysicalIRModel",
    "PhysicalIRSerializationError",
    "PhysicalPage",
    "physical_document_from_dict",
    "physical_document_from_json",
    "physical_document_to_dict",
    "physical_document_to_json",
]
