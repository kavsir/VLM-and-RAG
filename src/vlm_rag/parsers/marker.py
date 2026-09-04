"""Controlled subprocess adapter for an external Marker runtime."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vlm_rag.registry import DocumentManifest, verify_artifact

MARKER_NAME = "marker"
MARKER_VERSION = "2.0.0"
MARKER_MODE = "fast"
MARKER_BACKEND = "fast-no-ocr"
MARKER_OUTPUT_FORMAT = "json"

MarkerArtifactKind = Literal["document_json", "metadata_json", "image", "other"]


class MarkerError(RuntimeError):
    """Base error for the external Marker boundary."""


class MarkerExecutableNotFoundError(MarkerError):
    """Raised when an external executable cannot be resolved."""


class MarkerProbeError(MarkerError):
    """Raised when package metadata cannot prove the expected Marker version."""


class MarkerTimeoutError(MarkerError):
    """Raised when an external process exceeds its timeout."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class MarkerExecutionError(MarkerError):
    """Raised when Marker exits unsuccessfully."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(
            f"Marker exited with status {exit_code}: {stderr.strip() or stdout.strip()}"
        )
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class MarkerOutputError(MarkerError):
    """Raised when a Marker run directory or its outputs are invalid."""


@dataclass(frozen=True, slots=True)
class MarkerProbe:
    """Observed identity of the configured external Marker runtime."""

    parser: str
    version: str
    executable_path: Path
    python_path: Path
    python_version: str


@dataclass(frozen=True, slots=True)
class MarkerArtifact:
    """Integrity evidence for one retained Marker artifact."""

    relative_path: str
    kind: MarkerArtifactKind
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "relative_path": self.relative_path,
            "kind": self.kind,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class MarkerExecutionMetadata:
    """Process-level evidence captured for a successful parse."""

    command: tuple[str, ...]
    started_at: str
    completed_at: str
    duration_seconds: float
    exit_code: int
    stdout_log: str
    stderr_log: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation."""
        return {
            "command": list(self.command),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
        }


@dataclass(frozen=True, slots=True)
class MarkerRun:
    """Structured record of one verified Marker parse run."""

    schema_version: int
    status: Literal["succeeded"]
    parser: str
    parser_version: str
    backend: str
    mode: str
    disable_ocr: bool
    output_format: str
    executable_path: str
    python_path: str
    python_version: str
    device: str
    document_id: str
    version_id: str
    input_sha256: str
    output_directory: str
    execution: MarkerExecutionMetadata
    artifacts: tuple[MarkerArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible run manifest."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "backend": self.backend,
            "mode": self.mode,
            "disable_ocr": self.disable_ocr,
            "output_format": self.output_format,
            "executable_path": self.executable_path,
            "python_path": self.python_path,
            "python_version": self.python_version,
            "device": self.device,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "input_sha256": self.input_sha256,
            "output_directory": self.output_directory,
            "execution": self.execution.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class MarkerAdapter:
    """Invoke Marker 2.0.0 through a narrow, reproducible filesystem boundary."""

    def __init__(
        self,
        *,
        executable: str | Path = "marker_single",
        python_executable: str | Path = "python",
        expected_version: str = MARKER_VERSION,
        timeout_seconds: float = 14_400.0,
    ) -> None:
        if expected_version != MARKER_VERSION:
            raise ValueError(
                f"unsupported Marker version {expected_version!r}; expected {MARKER_VERSION!r}"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self._executable = os.fspath(executable)
        self._python_executable = os.fspath(python_executable)
        self.expected_version = expected_version
        self.timeout_seconds = timeout_seconds

    def probe(self) -> MarkerProbe:
        """Resolve executables and probe Marker via installed-package metadata."""
        executable_path = self._resolve_executable(self._executable, "Marker")
        python_path = self._resolve_executable(self._python_executable, "Python")
        script = (
            "import importlib.metadata,sys;"
            "print(importlib.metadata.version('marker-pdf'));"
            "print(sys.version.split()[0])"
        )
        completed = self._run_process((os.fspath(python_path), "-c", script))
        if completed.returncode != 0:
            raise MarkerProbeError(
                "Marker package version probe exited with status "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != 2:
            raise MarkerProbeError(f"unrecognized Marker package probe output: {lines!r}")
        version, python_version = lines
        if version != self.expected_version:
            raise MarkerProbeError(
                f"Marker version mismatch: expected {self.expected_version}, got {version}"
            )
        if not python_version.startswith("3.12."):
            raise MarkerProbeError(
                f"Marker requires the reviewed Python 3.12 environment, got {python_version}"
            )
        return MarkerProbe(
            parser=MARKER_NAME,
            version=version,
            executable_path=executable_path,
            python_path=python_path,
            python_version=python_version,
        )

    def run(
        self,
        *,
        manifest: DocumentManifest,
        input_pdf: Path,
        output_root: Path,
    ) -> MarkerRun:
        """Verify the source, run Marker, and persist complete execution evidence."""
        verified_input = verify_artifact(input_pdf, manifest.artifact)
        probe = self.probe()
        run_directory = output_root / probe.version / MARKER_BACKEND
        raw_directory = run_directory / "raw"
        self._prepare_run_directory(run_directory, raw_directory)
        command = self.build_command(probe.executable_path, input_pdf, raw_directory)

        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        completed = self._run_process(command)
        duration_seconds = time.perf_counter() - started_clock
        completed_at = datetime.now(UTC)
        if completed.returncode != 0:
            raise MarkerExecutionError(completed.returncode, completed.stdout, completed.stderr)

        stdout_path = run_directory / "stdout.log"
        stderr_path = run_directory / "stderr.log"
        self._write_bytes(stdout_path, completed.stdout.encode("utf-8"))
        self._write_bytes(stderr_path, completed.stderr.encode("utf-8"))
        artifacts = discover_marker_artifacts(raw_directory)
        discover_marker_document_json(raw_directory)
        execution = MarkerExecutionMetadata(
            command=command,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration_seconds,
            exit_code=completed.returncode,
            stdout_log=stdout_path.name,
            stderr_log=stderr_path.name,
        )
        run = MarkerRun(
            schema_version=1,
            status="succeeded",
            parser=probe.parser,
            parser_version=probe.version,
            backend=MARKER_BACKEND,
            mode=MARKER_MODE,
            disable_ocr=True,
            output_format=MARKER_OUTPUT_FORMAT,
            executable_path=os.fspath(probe.executable_path),
            python_path=os.fspath(probe.python_path),
            python_version=probe.python_version,
            device="cpu",
            document_id=manifest.document.id,
            version_id=manifest.version.id,
            input_sha256=verified_input.sha256,
            output_directory=os.fspath(run_directory.resolve()),
            execution=execution,
            artifacts=artifacts,
        )
        run_path = run_directory / "run.json"
        self._write_json(run_path, run.to_dict())
        retained = _discover_retained_artifacts(run_directory, artifacts)
        self._write_json(
            run_directory / "artifact_manifest.json",
            {"schema_version": 1, "artifacts": retained},
        )
        return run

    @staticmethod
    def build_command(executable: Path, input_pdf: Path, raw_directory: Path) -> tuple[str, ...]:
        """Build the exact Marker fast/no-OCR JSON command as an argument tuple."""
        return (
            os.fspath(executable),
            os.fspath(input_pdf.resolve()),
            "--mode",
            MARKER_MODE,
            "--disable_ocr",
            "--output_format",
            MARKER_OUTPUT_FORMAT,
            "--output_dir",
            os.fspath(raw_directory.resolve()),
            "--JSONRenderer_keep_pageheader_in_output",
            "--JSONRenderer_keep_pagefooter_in_output",
        )

    @staticmethod
    def _resolve_executable(value: str, label: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute() or candidate.parent != Path():
            if not candidate.is_file():
                raise MarkerExecutableNotFoundError(
                    f"{label} executable does not exist: {candidate}"
                )
            return candidate.resolve()
        resolved = shutil.which(value)
        if resolved is None:
            raise MarkerExecutableNotFoundError(
                f"{label} executable {value!r} was not found on PATH"
            )
        return Path(resolved).resolve()

    def _run_process(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                capture_output=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise MarkerExecutableNotFoundError(
                f"external executable disappeared before execution: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MarkerTimeoutError(
                f"Marker command timed out after {self.timeout_seconds:g} seconds",
                stdout=_coerce_process_output(exc.stdout),
                stderr=_coerce_process_output(exc.stderr),
            ) from exc

    @staticmethod
    def _prepare_run_directory(run_directory: Path, raw_directory: Path) -> None:
        try:
            if run_directory.exists() and any(run_directory.iterdir()):
                raise MarkerOutputError(
                    f"parser run directory is not empty; refusing to mix outputs: {run_directory}"
                )
            raw_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MarkerOutputError(f"cannot prepare parser output directory: {exc}") from exc

    @staticmethod
    def _write_bytes(path: Path, content: bytes) -> None:
        try:
            path.write_bytes(content)
        except OSError as exc:
            raise MarkerOutputError(f"cannot write parser evidence {path}: {exc}") from exc

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, object]) -> None:
        temporary_path: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            os.close(descriptor)
            temporary_path = Path(name)
            temporary_path.write_bytes(
                (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            )
            os.replace(temporary_path, path)
        except OSError as exc:
            raise MarkerOutputError(f"cannot write parser manifest {path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)


def discover_marker_artifacts(raw_directory: Path) -> tuple[MarkerArtifact, ...]:
    """Hash every regular file under the immutable raw Marker output directory."""
    if not raw_directory.is_dir():
        raise MarkerOutputError(f"raw Marker output directory does not exist: {raw_directory}")
    artifacts: list[MarkerArtifact] = []
    try:
        for path in sorted(item for item in raw_directory.rglob("*") if item.is_file()):
            relative_path = path.relative_to(raw_directory).as_posix()
            size, digest = _hash_file(path)
            artifacts.append(
                MarkerArtifact(
                    relative_path=relative_path,
                    kind=_classify_artifact(path),
                    byte_size=size,
                    sha256=digest,
                )
            )
    except OSError as exc:
        raise MarkerOutputError(f"cannot inspect raw Marker outputs: {exc}") from exc
    return tuple(artifacts)


def discover_marker_document_json(raw_directory: Path) -> Path:
    """Return the sole Marker document tree, excluding metadata sidecars."""
    candidates: list[Path] = []
    for path in sorted(raw_directory.rglob("*.json")):
        if path.name.casefold().endswith("_meta.json"):
            continue
        try:
            value: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MarkerOutputError(f"cannot read Marker JSON candidate {path}: {exc}") from exc
        if _is_marker_document_tree(value):
            candidates.append(path)
    if not candidates:
        raise MarkerOutputError(f"missing Marker document JSON below {raw_directory}")
    if len(candidates) != 1:
        relative = [path.relative_to(raw_directory).as_posix() for path in candidates]
        raise MarkerOutputError(f"ambiguous Marker document JSON below {raw_directory}: {relative}")
    return candidates[0]


def _is_marker_document_tree(value: object) -> bool:
    if isinstance(value, dict):
        return value.get("block_type") == "Document" and isinstance(value.get("children"), list)
    if isinstance(value, list):
        return bool(value) and all(
            isinstance(item, dict) and item.get("block_type") == "Page" for item in value
        )
    return False


def _classify_artifact(path: Path) -> MarkerArtifactKind:
    name = path.name.casefold()
    if name.endswith("_meta.json"):
        return "metadata_json"
    if path.suffix.casefold() == ".json":
        return "document_json"
    if path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    return "other"


def _discover_retained_artifacts(
    run_directory: Path, raw_artifacts: tuple[MarkerArtifact, ...]
) -> list[dict[str, object]]:
    retained: list[dict[str, object]] = []
    for artifact in raw_artifacts:
        item = artifact.to_dict()
        item["relative_path"] = f"raw/{artifact.relative_path}"
        retained.append(item)
    for name in ("stdout.log", "stderr.log", "run.json"):
        path = run_directory / name
        size, digest = _hash_file(path)
        retained.append(
            {"relative_path": name, "kind": "run_evidence", "byte_size": size, "sha256": digest}
        )
    return sorted(retained, key=lambda item: str(item["relative_path"]))


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return byte_size, digest.hexdigest()


def _coerce_process_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


__all__ = [
    "MARKER_BACKEND",
    "MARKER_MODE",
    "MARKER_NAME",
    "MARKER_OUTPUT_FORMAT",
    "MARKER_VERSION",
    "MarkerAdapter",
    "MarkerArtifact",
    "MarkerError",
    "MarkerExecutableNotFoundError",
    "MarkerExecutionError",
    "MarkerOutputError",
    "MarkerProbe",
    "MarkerProbeError",
    "MarkerRun",
    "MarkerTimeoutError",
    "discover_marker_artifacts",
    "discover_marker_document_json",
]
