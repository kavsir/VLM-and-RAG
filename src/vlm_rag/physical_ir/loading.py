"""Explicit version dispatch for Physical Document IR JSON."""

import json
from collections.abc import Mapping

from vlm_rag.physical_ir.models import PhysicalDocument
from vlm_rag.physical_ir.serialization import physical_document_from_dict
from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_from_dict
from vlm_rag.physical_ir.v1 import PhysicalDocumentV1

PhysicalDocumentAny = PhysicalDocument | PhysicalDocumentV1


class PhysicalIRVersionError(ValueError):
    """Raised when a document declares no supported wire/schema version."""


def load_physical_document_from_dict(data: Mapping[str, object]) -> PhysicalDocumentAny:
    """Load wire version 1 as v0 and wire version 2 as v1 without upgrading."""
    version = data.get("physical_ir_version")
    if type(version) is not int or version not in {1, 2}:
        raise PhysicalIRVersionError(
            f"unsupported physical_ir_version {version!r}; expected 1 or 2"
        )
    if version == 1:
        return physical_document_from_dict(data)
    return physical_document_v1_from_dict(data)


def load_physical_document_from_json(json_str: str) -> PhysicalDocumentAny:
    """Decode JSON and dispatch strictly using ``physical_ir_version``."""
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise PhysicalIRVersionError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PhysicalIRVersionError("Physical IR JSON root must be an object")
    return load_physical_document_from_dict(raw)


def load_physical_document(value: Mapping[str, object] | str) -> PhysicalDocumentAny:
    """Dispatch a mapping or JSON string without guessing or upgrading versions."""
    if isinstance(value, str):
        return load_physical_document_from_json(value)
    return load_physical_document_from_dict(value)


__all__ = [
    "PhysicalDocumentAny",
    "PhysicalIRVersionError",
    "load_physical_document",
    "load_physical_document_from_dict",
    "load_physical_document_from_json",
]
