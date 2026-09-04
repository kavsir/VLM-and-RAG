"""Typed domain records for versioned source documents."""

from datetime import date
from pathlib import PurePath
from typing import Annotated, Literal, Self
from urllib.parse import urlparse

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")]
VersionIdentifier = Annotated[str, StringConstraints(pattern=r"^v[1-9][0-9]*$")]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class RegistryModel(BaseModel):
    """Shared strict, immutable behavior for registry records."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)


class DocumentIdentity(RegistryModel):
    """Stable identity and official descriptive metadata for a document."""

    id: Identifier
    document_number: str = Field(min_length=1)
    normalized_document_number: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: str = Field(min_length=1)
    issuer: str = Field(min_length=1)


class DocumentVersion(RegistryModel):
    """A dated version belonging to one stable document identity."""

    id: VersionIdentifier
    document_id: Identifier
    issued_on: date
    effective_on: date | None = None


class SourceReference(RegistryModel):
    """Provenance captured from a publisher's metadata page and byte source."""

    source_type: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    signer: str = Field(min_length=1)
    landing_page_url: str = Field(min_length=1)
    asset_url: str = Field(min_length=1)
    retrieved_at: AwareDatetime

    @field_validator("landing_page_url", "asset_url")
    @classmethod
    def require_https_url(cls, value: str) -> str:
        """Require complete HTTPS provenance URLs."""
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("must be an absolute HTTPS URL")
        return value

    @model_validator(mode="after")
    def require_distinct_source_urls(self) -> Self:
        """Keep descriptive landing-page provenance separate from artifact bytes."""
        if self.landing_page_url == self.asset_url:
            raise ValueError("landing_page_url and asset_url must be distinct")
        return self


class FileArtifact(RegistryModel):
    """Expected immutable properties of one downloaded file artifact."""

    filename: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    byte_size: PositiveInt
    sha256: Sha256Digest

    @field_validator("filename")
    @classmethod
    def require_plain_filename(cls, value: str) -> str:
        """Reject paths so a manifest cannot redirect local file placement."""
        if PurePath(value).name != value or "/" in value or "\\" in value:
            raise ValueError("must be a filename without directory components")
        return value


class DocumentManifest(RegistryModel):
    """Versioned manifest tying identity, version, source, and bytes together."""

    manifest_schema_version: Literal[1]
    document: DocumentIdentity
    version: DocumentVersion
    source: SourceReference
    artifact: FileArtifact

    @model_validator(mode="after")
    def require_matching_document_identity(self) -> Self:
        """Ensure the version cannot silently refer to another document."""
        if self.version.document_id != self.document.id:
            raise ValueError("version.document_id must match document.id")
        return self


__all__ = [
    "DocumentIdentity",
    "DocumentManifest",
    "DocumentVersion",
    "FileArtifact",
    "Sha256Digest",
    "SourceReference",
]
