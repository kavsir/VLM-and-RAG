"""Deterministic serialization and parsing for Physical Document IR."""

import json
from collections.abc import Mapping

from pydantic import ValidationError

from vlm_rag.physical_ir.models import PhysicalDocument


class PhysicalIRSerializationError(ValueError):
    """Raised when Physical Document IR cannot be serialized or parsed."""


def physical_document_to_dict(document: PhysicalDocument) -> dict[str, object]:
    """Convert a PhysicalDocument into a plain JSON-compatible dictionary."""
    return document.model_dump(mode="json")


def physical_document_to_json(document: PhysicalDocument, *, indent: int | None = 2) -> str:
    """Serialize a PhysicalDocument to a deterministic UTF-8 JSON string."""
    data = physical_document_to_dict(document)
    return json.dumps(data, ensure_ascii=False, indent=indent) + "\n"


def physical_document_from_dict(data: Mapping[str, object]) -> PhysicalDocument:
    """Validate and construct a PhysicalDocument from a dictionary."""
    try:
        return PhysicalDocument.model_validate(data)
    except ValidationError as exc:
        raise PhysicalIRSerializationError(
            f"failed to validate PhysicalDocument data:\n{exc}"
        ) from exc


def physical_document_from_json(json_str: str) -> PhysicalDocument:
    """Parse and validate a PhysicalDocument from a JSON string."""
    try:
        raw: object = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise PhysicalIRSerializationError(f"invalid JSON for PhysicalDocument: {exc}") from exc

    if not isinstance(raw, Mapping):
        raise PhysicalIRSerializationError("PhysicalDocument JSON must contain an object")

    return physical_document_from_dict(raw)


__all__ = [
    "PhysicalIRSerializationError",
    "physical_document_from_dict",
    "physical_document_from_json",
    "physical_document_to_dict",
    "physical_document_to_json",
]
