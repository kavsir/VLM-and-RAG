"""Deterministic JSON serialization for Domain Semantic IR v1."""

import json
from collections.abc import Mapping

from pydantic import ValidationError

from vlm_rag.semantic_ir.models import SemanticDocument


class SemanticIRSerializationError(ValueError):
    """Raised when Semantic IR v1 cannot be encoded or decoded strictly."""


def semantic_document_to_dict(document: SemanticDocument) -> dict[str, object]:
    return document.model_dump(mode="json")


def semantic_document_to_json(document: SemanticDocument, *, indent: int | None = 2) -> str:
    try:
        return (
            json.dumps(
                semantic_document_to_dict(document),
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
            )
            + "\n"
        )
    except ValueError as exc:
        raise SemanticIRSerializationError(f"cannot serialize Semantic IR v1: {exc}") from exc


def semantic_document_from_dict(data: Mapping[str, object]) -> SemanticDocument:
    if data.get("semantic_ir_version") != 1:
        raise SemanticIRSerializationError(
            f"unsupported semantic_ir_version {data.get('semantic_ir_version')!r}; expected 1"
        )
    try:
        return SemanticDocument.model_validate(dict(data))
    except ValidationError as exc:
        raise SemanticIRSerializationError(f"invalid Semantic IR v1:\n{exc}") from exc


def semantic_document_from_json(payload: str) -> SemanticDocument:
    try:
        value: object = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SemanticIRSerializationError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticIRSerializationError("Semantic IR JSON root must be an object")
    return semantic_document_from_dict(value)


__all__ = [
    "SemanticIRSerializationError",
    "semantic_document_from_dict",
    "semantic_document_from_json",
    "semantic_document_to_dict",
    "semantic_document_to_json",
]
