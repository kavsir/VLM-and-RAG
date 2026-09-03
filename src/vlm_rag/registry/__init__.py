"""Golden document registry domain model and validation helpers."""

from vlm_rag.registry.artifacts import (
    ArtifactError,
    VerifiedArtifact,
    fetch_artifact,
    verify_artifact,
)
from vlm_rag.registry.manifest import ManifestValidationError, load_manifest
from vlm_rag.registry.models import (
    DocumentIdentity,
    DocumentManifest,
    DocumentVersion,
    FileArtifact,
    Sha256Digest,
    SourceReference,
)

__all__ = [
    "ArtifactError",
    "DocumentIdentity",
    "DocumentManifest",
    "DocumentVersion",
    "FileArtifact",
    "ManifestValidationError",
    "Sha256Digest",
    "SourceReference",
    "VerifiedArtifact",
    "fetch_artifact",
    "load_manifest",
    "verify_artifact",
]
