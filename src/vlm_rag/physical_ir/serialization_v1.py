"""Deterministic serialization for Physical Document IR v1 (wire version 2)."""

import json
from collections.abc import Mapping

from pydantic import ValidationError

from vlm_rag.physical_ir.v1 import PhysicalDocumentV1


class PhysicalIRV1SerializationError(ValueError):
    """Raised when wire-version-2 Physical IR cannot be decoded."""


def physical_document_v1_to_dict(document: PhysicalDocumentV1) -> dict[str, object]:
    """Convert a v1 document to JSON-compatible primitives in field order."""
    return document.model_dump(mode="json")


def physical_document_v1_to_json(document: PhysicalDocumentV1, *, indent: int | None = 2) -> str:
    """Serialize deterministically as UTF-8-ready JSON with a terminal LF."""
    payload = physical_document_v1_to_dict(document)
    try:
        return json.dumps(payload, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"
    except ValueError as exc:
        raise PhysicalIRV1SerializationError(
            f"Physical IR v1 contains a non-finite JSON number: {exc}"
        ) from exc


def physical_document_v1_from_dict(data: Mapping[str, object]) -> PhysicalDocumentV1:
    """Validate a mapping strictly as Physical IR v1/wire version 2."""
    try:
        return PhysicalDocumentV1.model_validate(dict(data))
    except ValidationError as exc:
        raise PhysicalIRV1SerializationError(f"invalid Physical IR v1 document:\n{exc}") from exc


def physical_document_v1_from_json(json_str: str) -> PhysicalDocumentV1:
    """Decode JSON and validate strictly as Physical IR v1/wire version 2."""
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise PhysicalIRV1SerializationError(f"invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PhysicalIRV1SerializationError("Physical IR v1 JSON root must be an object")
    return physical_document_v1_from_dict(raw)


__all__ = [
    "PhysicalIRV1SerializationError",
    "physical_document_v1_from_dict",
    "physical_document_v1_from_json",
    "physical_document_v1_to_dict",
    "physical_document_v1_to_json",
]
