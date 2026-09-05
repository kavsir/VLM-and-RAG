"""Evidence-preserving visual asset helpers shared by v1 normalizers."""

import base64
import binascii
import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath

from vlm_rag.physical_ir.v1 import VisualAssetEvidence, VisualAssetStorage


class VisualAssetSourceError(ValueError):
    """Raised when parser-provided visual evidence is unsafe or corrupt."""


def _media_type(payload: bytes) -> str | None:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        return "image/webp"
    return None


def embedded_visual_asset(encoded: str) -> VisualAssetEvidence:
    """Validate and fingerprint parser-embedded base64 bytes without materializing them."""
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise VisualAssetSourceError("embedded visual asset is not valid base64") from exc
    if not payload:
        raise VisualAssetSourceError("embedded visual asset is empty")
    media_type = _media_type(payload)
    if media_type is None:
        raise VisualAssetSourceError("embedded visual asset has an unsupported byte signature")
    return VisualAssetEvidence(
        storage_kind=VisualAssetStorage.EMBEDDED_RAW,
        media_type=media_type,
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


def relative_visual_asset(
    raw_directory: Path, referring_directory: Path, parser_path: str
) -> VisualAssetEvidence:
    """Validate a parser-relative path and fingerprint bytes when the file is retained."""
    posix = PurePosixPath(parser_path)
    windows = PureWindowsPath(parser_path)
    if (
        not parser_path
        or "\\" in parser_path
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise VisualAssetSourceError(f"unsafe parser visual path {parser_path!r}")
    raw_resolved = raw_directory.resolve()
    candidate = referring_directory.joinpath(*posix.parts).resolve()
    if not candidate.is_relative_to(raw_resolved):
        raise VisualAssetSourceError(f"parser visual path escapes raw directory: {parser_path!r}")
    if not candidate.is_file():
        return VisualAssetEvidence(storage_kind=VisualAssetStorage.UNAVAILABLE)
    payload = candidate.read_bytes()
    if not payload:
        raise VisualAssetSourceError(f"retained visual asset is empty: {parser_path!r}")
    return VisualAssetEvidence(
        storage_kind=VisualAssetStorage.RELATIVE_FILE,
        relative_path=candidate.relative_to(raw_resolved).as_posix(),
        media_type=_media_type(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
    )


__all__ = ["VisualAssetSourceError", "embedded_visual_asset", "relative_visual_asset"]
