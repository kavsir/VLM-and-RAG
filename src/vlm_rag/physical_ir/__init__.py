"""Versioned Physical Document Intermediate Representation APIs."""

from vlm_rag.physical_ir.loading import (
    PhysicalIRVersionError,
    load_physical_document,
)
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
from vlm_rag.physical_ir.serialization_v1 import (
    PhysicalIRV1SerializationError,
    physical_document_v1_from_dict,
    physical_document_v1_from_json,
    physical_document_v1_to_dict,
    physical_document_v1_to_json,
)
from vlm_rag.physical_ir.v1 import (
    BlockKindV1,
    BlockProvenanceV1,
    PhysicalBlockV1,
    PhysicalDocumentV1,
    PhysicalPageV1,
    TableCell,
    TableStructure,
    TextExtractionEvidence,
    TextExtractionMethod,
    VisualAssetEvidence,
    VisualAssetStorage,
)

__all__ = [
    "BlockDisposition",
    "BlockKind",
    "BlockKindV1",
    "BlockProvenance",
    "BlockProvenanceV1",
    "BoundingBox",
    "PhysicalBlock",
    "PhysicalBlockV1",
    "PhysicalDocument",
    "PhysicalDocumentV1",
    "PhysicalIRModel",
    "PhysicalIRSerializationError",
    "PhysicalIRV1SerializationError",
    "PhysicalIRVersionError",
    "PhysicalPage",
    "PhysicalPageV1",
    "TableCell",
    "TableStructure",
    "TextExtractionEvidence",
    "TextExtractionMethod",
    "VisualAssetEvidence",
    "VisualAssetStorage",
    "load_physical_document",
    "physical_document_from_dict",
    "physical_document_from_json",
    "physical_document_to_dict",
    "physical_document_to_json",
    "physical_document_v1_from_dict",
    "physical_document_v1_from_json",
    "physical_document_v1_to_dict",
    "physical_document_v1_to_json",
]
