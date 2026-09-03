"""Checksum validation and atomic retrieval of registered file artifacts."""

import hashlib
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request
from urllib.parse import urlparse

from vlm_rag.registry.models import FileArtifact, SourceReference

_CHUNK_SIZE = 1024 * 1024


class ArtifactError(RuntimeError):
    """Raised when artifact retrieval or verification fails."""


@dataclass(frozen=True, slots=True)
class VerifiedArtifact:
    """Observed properties of bytes that matched a registry manifest."""

    path: Path
    byte_size: int
    sha256: str
    final_url: str | None = None


def verify_artifact(path: Path, expected: FileArtifact) -> VerifiedArtifact:
    """Stream a local file and require its size and SHA-256 to match the manifest."""
    digest = hashlib.sha256()
    byte_size = 0

    try:
        with path.open("rb") as file_handle:
            while chunk := file_handle.read(_CHUNK_SIZE):
                digest.update(chunk)
                byte_size += len(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot read artifact {path}: {exc}") from exc

    sha256 = digest.hexdigest()
    mismatches: list[str] = []
    if byte_size != expected.byte_size:
        mismatches.append(f"size expected {expected.byte_size}, got {byte_size}")
    if sha256 != expected.sha256:
        mismatches.append(f"SHA-256 expected {expected.sha256}, got {sha256}")
    if mismatches:
        raise ArtifactError(f"artifact {path} failed verification: {'; '.join(mismatches)}")

    return VerifiedArtifact(path=path, byte_size=byte_size, sha256=sha256)


def fetch_artifact(
    source: SourceReference,
    expected: FileArtifact,
    destination: Path,
    *,
    timeout_seconds: float = 60.0,
) -> VerifiedArtifact:
    """Download an HTTPS artifact atomically and require its registered digest and size."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".part",
        )
        os.close(file_descriptor)
    except OSError as exc:
        raise ArtifactError(f"cannot prepare artifact destination {destination}: {exc}") from exc

    temporary_path = Path(temporary_name)
    final_url: str | None = None

    http_request = request.Request(
        source.asset_url,
        headers={
            "Accept": expected.media_type,
            "User-Agent": "VLM-and-RAG/0.1 (+https://github.com/kavsir/VLM-and-RAG)",
        },
    )

    try:
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:
                status = response.getcode()
                if status is None or not 200 <= status < 300:
                    raise ArtifactError(f"artifact request returned HTTP status {status}")

                final_url = response.geturl()
                if urlparse(final_url).scheme != "https":
                    raise ArtifactError(f"artifact request resolved to non-HTTPS URL: {final_url}")

                byte_size = 0
                with temporary_path.open("wb") as file_handle:
                    while chunk := response.read(_CHUNK_SIZE):
                        byte_size += len(chunk)
                        if byte_size > expected.byte_size:
                            raise ArtifactError(
                                "artifact download exceeded the expected size "
                                f"of {expected.byte_size} bytes"
                            )
                        file_handle.write(chunk)
        except error.HTTPError as exc:
            raise ArtifactError(f"artifact request returned HTTP status {exc.code}") from exc
        except error.URLError as exc:
            raise ArtifactError(f"artifact request failed: {exc.reason}") from exc
        except OSError as exc:
            raise ArtifactError(f"artifact download failed: {exc}") from exc

        verified = verify_artifact(temporary_path, expected)
        try:
            os.replace(temporary_path, destination)
        except OSError as exc:
            raise ArtifactError(
                f"cannot publish verified artifact to {destination}: {exc}"
            ) from exc
        return VerifiedArtifact(
            path=destination,
            byte_size=verified.byte_size,
            sha256=verified.sha256,
            final_url=final_url,
        )
    finally:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)


__all__ = ["ArtifactError", "VerifiedArtifact", "fetch_artifact", "verify_artifact"]
