"""Controlled subprocess adapter for an external MinerU runtime."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from vlm_rag.registry import DocumentManifest, verify_artifact

MINERU_NAME = "mineru"
MINERU_VERSION = "3.4.5"
MINERU_BACKEND = "pipeline"
_VERSION_PATTERN = re.compile(r"^mineru,\s+version\s+(?P<version>\S+)$", re.IGNORECASE)
_REQUIRED_ARTIFACT_KINDS = frozenset(
    {"markdown", "middle_json", "content_list_json", "content_list_v2_json"}
)

ArtifactKind = Literal[
    "markdown",
    "middle_json",
    "content_list_json",
    "content_list_v2_json",
    "image",
    "other",
]
RunStatus = Literal["succeeded"]


class MinerUError(RuntimeError):
    """Base error for the external MinerU boundary."""


class MinerUExecutableNotFoundError(MinerUError):
    """Raised when the configured MinerU executable cannot be resolved."""


class MinerUProbeError(MinerUError):
    """Raised when MinerU version probing fails or returns unexpected output."""


class MinerUTimeoutError(MinerUError):
    """Raised when the external MinerU process exceeds its timeout."""

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class MinerUExecutionError(MinerUError):
    """Raised when MinerU exits unsuccessfully."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        super().__init__(
            f"MinerU exited with status {exit_code}: {stderr.strip() or stdout.strip()}"
        )
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class MinerUOutputError(MinerUError):
    """Raised when a parse run directory or its expected outputs are invalid."""


@dataclass(frozen=True, slots=True)
class MinerUProbe:
    """Observed identity of the configured external executable."""

    parser: str
    version: str
    executable_path: Path


@dataclass(frozen=True, slots=True)
class ParserArtifact:
    """Integrity evidence for one retained raw parser output file."""

    relative_path: str
    kind: ArtifactKind
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
class ParserExecutionMetadata:
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
class ParserRun:
    """Structured record of one verified MinerU parse run."""

    schema_version: int
    status: RunStatus
    parser: str
    parser_version: str
    backend: str
    executable_path: str
    document_id: str
    version_id: str
    input_sha256: str
    output_directory: str
    execution: ParserExecutionMetadata
    artifacts: tuple[ParserArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the complete JSON-compatible run manifest."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "parser": self.parser,
            "parser_version": self.parser_version,
            "backend": self.backend,
            "executable_path": self.executable_path,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "input_sha256": self.input_sha256,
            "output_directory": self.output_directory,
            "execution": self.execution.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


class MinerUAdapter:
    """Invoke one pinned MinerU CLI through a narrow filesystem boundary."""

    def __init__(
        self,
        *,
        executable: str | Path = MINERU_NAME,
        backend: str = MINERU_BACKEND,
        expected_version: str = MINERU_VERSION,
        timeout_seconds: float = 14_400.0,
    ) -> None:
        if backend != MINERU_BACKEND:
            raise ValueError(f"unsupported MinerU backend {backend!r}; expected 'pipeline'")
        if expected_version != MINERU_VERSION:
            raise ValueError(
                f"unsupported MinerU version {expected_version!r}; expected {MINERU_VERSION!r}"
            )
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        self._executable = os.fspath(executable)
        self.backend = backend
        self.expected_version = expected_version
        self.timeout_seconds = timeout_seconds

    def probe(self) -> MinerUProbe:
        """Resolve MinerU and obtain its actual version from ``mineru --version``."""
        executable_path = self._resolve_executable()
        command = (os.fspath(executable_path), "--version")
        completed = self._run_process(command)
        if completed.returncode != 0:
            raise MinerUProbeError(
                "MinerU version probe exited with status "
                f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
            )

        output = completed.stdout.strip() or completed.stderr.strip()
        match = _VERSION_PATTERN.fullmatch(output)
        if match is None:
            raise MinerUProbeError(f"unrecognized MinerU version output: {output!r}")

        version = match.group("version")
        if version != self.expected_version:
            raise MinerUProbeError(
                f"MinerU version mismatch: expected {self.expected_version}, got {version}"
            )
        return MinerUProbe(parser=MINERU_NAME, version=version, executable_path=executable_path)

    def run(
        self,
        *,
        manifest: DocumentManifest,
        input_pdf: Path,
        output_root: Path,
    ) -> ParserRun:
        """Verify the input, invoke MinerU, validate outputs, and write ``run.json``."""
        verified_input = verify_artifact(input_pdf, manifest.artifact)
        probe = self.probe()
        run_directory = output_root / probe.version / self.backend
        raw_directory = run_directory / "raw"
        self._prepare_run_directory(run_directory, raw_directory)

        command = (
            os.fspath(probe.executable_path),
            "-p",
            os.fspath(input_pdf.resolve()),
            "-o",
            os.fspath(raw_directory.resolve()),
            "-b",
            self.backend,
        )
        started_at = datetime.now(UTC)
        started_clock = time.perf_counter()
        completed = self._run_process(command)
        duration_seconds = time.perf_counter() - started_clock
        completed_at = datetime.now(UTC)

        if completed.returncode != 0:
            raise MinerUExecutionError(
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )

        stdout_log = run_directory / "stdout.log"
        stderr_log = run_directory / "stderr.log"
        self._write_text(stdout_log, completed.stdout)
        self._write_text(stderr_log, completed.stderr)

        artifacts = discover_artifacts(raw_directory)
        self._validate_expected_outputs(artifacts)
        execution = ParserExecutionMetadata(
            command=command,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_seconds=duration_seconds,
            exit_code=completed.returncode,
            stdout_log=stdout_log.name,
            stderr_log=stderr_log.name,
        )
        run = ParserRun(
            schema_version=1,
            status="succeeded",
            parser=probe.parser,
            parser_version=probe.version,
            backend=self.backend,
            executable_path=os.fspath(probe.executable_path),
            document_id=manifest.document.id,
            version_id=manifest.version.id,
            input_sha256=verified_input.sha256,
            output_directory=os.fspath(run_directory.resolve()),
            execution=execution,
            artifacts=artifacts,
        )
        self._write_run_manifest(run_directory / "run.json", run)
        return run

    def _resolve_executable(self) -> Path:
        candidate = Path(self._executable)
        has_directory = candidate.is_absolute() or candidate.parent != Path()
        if has_directory:
            if not candidate.is_file():
                raise MinerUExecutableNotFoundError(
                    f"MinerU executable does not exist: {candidate}"
                )
            return candidate.resolve()

        resolved = shutil.which(self._executable)
        if resolved is None:
            raise MinerUExecutableNotFoundError(
                f"MinerU executable {self._executable!r} was not found on PATH"
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
            raise MinerUExecutableNotFoundError(
                f"MinerU executable disappeared before execution: {command[0]}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MinerUTimeoutError(
                f"MinerU command timed out after {self.timeout_seconds:g} seconds",
                stdout=_coerce_process_output(exc.stdout),
                stderr=_coerce_process_output(exc.stderr),
            ) from exc

    @staticmethod
    def _prepare_run_directory(run_directory: Path, raw_directory: Path) -> None:
        try:
            if run_directory.exists() and any(run_directory.iterdir()):
                raise MinerUOutputError(
                    f"parser run directory is not empty; refusing to mix outputs: {run_directory}"
                )
            raw_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise MinerUOutputError(f"cannot prepare parser output directory: {exc}") from exc

    @staticmethod
    def _validate_expected_outputs(artifacts: tuple[ParserArtifact, ...]) -> None:
        observed_kinds = {artifact.kind for artifact in artifacts}
        missing = sorted(_REQUIRED_ARTIFACT_KINDS - observed_kinds)
        if missing:
            raise MinerUOutputError(
                f"MinerU completed without required inspection outputs: {', '.join(missing)}"
            )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise MinerUOutputError(f"cannot write parser log {path}: {exc}") from exc

    @staticmethod
    def _write_run_manifest(path: Path, run: ParserRun) -> None:
        temporary_path: Path | None = None
        try:
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent,
                prefix=".run.",
                suffix=".json.tmp",
            )
            os.close(file_descriptor)
            temporary_path = Path(temporary_name)
            temporary_path.write_text(
                json.dumps(run.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_path, path)
        except OSError as exc:
            raise MinerUOutputError(f"cannot write parser run manifest {path}: {exc}") from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)


def discover_artifacts(raw_directory: Path) -> tuple[ParserArtifact, ...]:
    """Discover and hash every regular file below a raw MinerU output directory."""
    if not raw_directory.is_dir():
        raise MinerUOutputError(f"raw MinerU output directory does not exist: {raw_directory}")

    artifacts: list[ParserArtifact] = []
    try:
        files = sorted(path for path in raw_directory.rglob("*") if path.is_file())
        for path in files:
            relative_path = path.relative_to(raw_directory).as_posix()
            byte_size, sha256 = _hash_file(path)
            artifacts.append(
                ParserArtifact(
                    relative_path=relative_path,
                    kind=_classify_artifact(path, relative_path),
                    byte_size=byte_size,
                    sha256=sha256,
                )
            )
    except OSError as exc:
        raise MinerUOutputError(f"cannot inspect raw MinerU outputs: {exc}") from exc
    return tuple(artifacts)


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return byte_size, digest.hexdigest()


def _classify_artifact(path: Path, relative_path: str) -> ArtifactKind:
    name = path.name.casefold()
    parts = {part.casefold() for part in Path(relative_path).parts}
    if name.endswith("_content_list_v2.json"):
        return "content_list_v2_json"
    if name.endswith("_content_list.json"):
        return "content_list_json"
    if name.endswith("_middle.json"):
        return "middle_json"
    if path.suffix.casefold() == ".md":
        return "markdown"
    if "images" in parts:
        return "image"
    return "other"


def _coerce_process_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


__all__ = [
    "MINERU_BACKEND",
    "MINERU_NAME",
    "MINERU_VERSION",
    "MinerUAdapter",
    "MinerUError",
    "MinerUExecutableNotFoundError",
    "MinerUExecutionError",
    "MinerUOutputError",
    "MinerUProbe",
    "MinerUProbeError",
    "MinerUTimeoutError",
    "ParserArtifact",
    "ParserExecutionMetadata",
    "ParserRun",
    "discover_artifacts",
]
