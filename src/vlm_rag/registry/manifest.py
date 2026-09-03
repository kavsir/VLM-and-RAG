"""Offline loading and validation for document manifests."""

from collections.abc import Mapping
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from vlm_rag.registry.models import DocumentManifest


class ManifestValidationError(ValueError):
    """Raised when a manifest cannot be parsed or violates its schema."""


def load_manifest(path: Path) -> DocumentManifest:
    """Read and validate a YAML manifest without network access."""
    try:
        raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestValidationError(f"cannot read manifest {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestValidationError(f"invalid YAML in manifest {path}: {exc}") from exc
    except ValueError as exc:
        raise ManifestValidationError(
            f"invalid YAML scalar (including a possible date) in manifest {path}: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise ManifestValidationError(f"manifest {path} must contain a YAML mapping")

    try:
        return DocumentManifest.model_validate(raw)
    except ValidationError as exc:
        raise ManifestValidationError(f"manifest {path} failed validation:\n{exc}") from exc


__all__ = ["ManifestValidationError", "load_manifest"]
