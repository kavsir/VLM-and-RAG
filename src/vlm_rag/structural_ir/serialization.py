"""Deterministic JSON serialization for Structural IR v1."""

import json
from collections.abc import Mapping

from pydantic import ValidationError

from vlm_rag.structural_ir.models import StructuralDocument


class StructuralIRSerializationError(ValueError):
    """Raised when Structural IR v1 cannot be encoded or decoded strictly."""


def structural_document_to_dict(document: StructuralDocument) -> dict[str, object]:
    """Convert a StructuralDocument to JSON-compatible primitives in field order."""
    return document.model_dump(mode="json")


def structural_document_to_json(document: StructuralDocument, *, indent: int | None = 2) -> str:
    """Serialize deterministic UTF-8-ready JSON with LF and a terminal newline."""
    try:
        return (
            json.dumps(
                structural_document_to_dict(document),
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
            )
            + "\n"
        )
    except ValueError as exc:
        raise StructuralIRSerializationError(f"cannot serialize Structural IR v1: {exc}") from exc


def structural_document_from_dict(data: Mapping[str, object]) -> StructuralDocument:
    """Validate an object as exactly Structural IR version 1."""
    version = data.get("structural_ir_version")
    if version != 1:
        raise StructuralIRSerializationError(
            f"unsupported structural_ir_version {version!r}; expected 1"
        )
    try:
        return StructuralDocument.model_validate(dict(data))
    except ValidationError as exc:
        raise StructuralIRSerializationError(f"invalid Structural IR v1:\n{exc}") from exc


def structural_document_from_json(payload: str) -> StructuralDocument:
    """Decode a JSON object and validate Structural IR version 1."""
    try:
        value: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise StructuralIRSerializationError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise StructuralIRSerializationError("Structural IR JSON root must be an object")
    return structural_document_from_dict(value)


__all__ = [
    "StructuralIRSerializationError",
    "structural_document_from_dict",
    "structural_document_from_json",
    "structural_document_to_dict",
    "structural_document_to_json",
]
