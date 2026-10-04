"""Verified visual-byte resolution boundary; Semantic construction never opens PDFs."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from vlm_rag.vlm.models import RenderSpecification, VisualEvidenceRequest


class VisualEvidenceIntegrityError(ValueError):
    """Raised before invocation when visual bytes or paths fail verification."""


@dataclass(frozen=True, slots=True)
class ResolvedVisualEvidence:
    data: bytes
    sha256: str
    byte_size: int
    media_type: str
    width: int | None = None
    height: int | None = None


class VisualEvidenceResolver(Protocol):
    def resolve(self, request: VisualEvidenceRequest) -> ResolvedVisualEvidence: ...


class PDFCropRenderer(Protocol):
    def render(
        self,
        pdf_bytes: bytes,
        *,
        page_index: int,
        bbox: object,
        specification: RenderSpecification,
    ) -> ResolvedVisualEvidence: ...


def _verified_bytes(path: Path, expected_sha: str, expected_size: int | None = None) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise VisualEvidenceIntegrityError("visual source SHA-256 mismatch")
    if expected_size is not None and len(data) != expected_size:
        raise VisualEvidenceIntegrityError("visual source byte-size mismatch")
    return data


def _verified_resolution(value: ResolvedVisualEvidence) -> ResolvedVisualEvidence:
    if hashlib.sha256(value.data).hexdigest() != value.sha256:
        raise VisualEvidenceIntegrityError("resolved visual evidence SHA-256 mismatch")
    if len(value.data) != value.byte_size:
        raise VisualEvidenceIntegrityError("resolved visual evidence byte-size mismatch")
    if not value.media_type:
        raise VisualEvidenceIntegrityError("resolved visual evidence requires media type")
    if value.width is not None and value.width <= 0:
        raise VisualEvidenceIntegrityError("resolved visual evidence width must be positive")
    if value.height is not None and value.height <= 0:
        raise VisualEvidenceIntegrityError("resolved visual evidence height must be positive")
    return value


class VerifiedPDFResolver:
    """Verify source PDF bytes, then delegate rendering to an injected implementation."""

    def __init__(self, source_pdf: Path, renderer: PDFCropRenderer) -> None:
        self._source_pdf = source_pdf
        self._renderer = renderer

    def resolve(self, request: VisualEvidenceRequest) -> ResolvedVisualEvidence:
        pdf_bytes = _verified_bytes(self._source_pdf, request.source_artifact_sha256)
        return _verified_resolution(
            self._renderer.render(
                pdf_bytes,
                page_index=request.page_index,
                bbox=request.bbox,
                specification=request.render_specification,
            )
        )


class VerifiedAssetResolver:
    """Resolve an already retained visual asset within one explicit trusted root."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def resolve(self, request: VisualEvidenceRequest) -> ResolvedVisualEvidence:
        asset = request.retained_visual_asset
        if asset is None:
            raise VisualEvidenceIntegrityError("request has no retained visual asset")
        path = (self._root / asset.relative_path).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise VisualEvidenceIntegrityError("visual asset escapes trusted root") from exc
        data = _verified_bytes(path, asset.sha256, asset.byte_size)
        return _verified_resolution(
            ResolvedVisualEvidence(
                data=data,
                sha256=asset.sha256,
                byte_size=len(data),
                media_type=asset.media_type,
            )
        )


__all__ = [
    "PDFCropRenderer",
    "ResolvedVisualEvidence",
    "VerifiedAssetResolver",
    "VerifiedPDFResolver",
    "VisualEvidenceIntegrityError",
    "VisualEvidenceResolver",
]
