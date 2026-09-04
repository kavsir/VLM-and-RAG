"""Contract tests for the external MinerU subprocess boundary."""

import hashlib
import json
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from vlm_rag.parsers import (
    MinerUAdapter,
    MinerUExecutableNotFoundError,
    MinerUExecutionError,
    MinerUOutputError,
    MinerUProbeError,
    MinerUTimeoutError,
    discover_artifacts,
)
from vlm_rag.registry import (
    ArtifactError,
    DocumentIdentity,
    DocumentManifest,
    DocumentVersion,
    FileArtifact,
    SourceReference,
)


def _manifest_for(payload: bytes) -> DocumentManifest:
    return DocumentManifest(
        manifest_schema_version=1,
        document=DocumentIdentity(
            id="test-document",
            document_number="1/QĐ-TEST",
            normalized_document_number="1/QD-TEST",
            title="Test document",
            document_type="Decision",
            issuer="Test issuer",
        ),
        version=DocumentVersion(
            id="v1",
            document_id="test-document",
            issued_on=date(2026, 5, 13),
            effective_on=date(2026, 5, 13),
        ),
        source=SourceReference(
            source_type="official_government_portal",
            publisher="Test publisher",
            signer="Test signer",
            landing_page_url="https://example.gov.test/document/1",
            asset_url="https://files.example.gov.test/document/1.pdf",
            retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
        ),
        artifact=FileArtifact(
            filename="source.pdf",
            media_type="application/pdf",
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def _successful_parse_outputs(command: list[str]) -> None:
    output_directory = Path(command[command.index("-o") + 1]) / "source" / "auto"
    (output_directory / "images").mkdir(parents=True)
    (output_directory / "source.md").write_text("# Parsed\n", encoding="utf-8")
    (output_directory / "source_middle.json").write_text('{"pdf_info": []}\n', encoding="utf-8")
    (output_directory / "source_content_list.json").write_text("[]\n", encoding="utf-8")
    (output_directory / "source_content_list_v2.json").write_text("[]\n", encoding="utf-8")
    (output_directory / "images" / "figure.png").write_bytes(b"tiny image")


def _mock_executable(monkeypatch: pytest.MonkeyPatch) -> Path:
    executable = Path("C:/Program Files/MinerU/mineru.exe")
    monkeypatch.setattr("vlm_rag.parsers.mineru.shutil.which", lambda _name: str(executable))
    return executable.resolve()


def test_missing_executable_has_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vlm_rag.parsers.mineru.shutil.which", lambda _name: None)

    with pytest.raises(MinerUExecutableNotFoundError, match="was not found on PATH"):
        MinerUAdapter(executable="missing-mineru").probe()


def test_version_probe_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    executable = _mock_executable(monkeypatch)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="mineru, version 3.4.5\n", stderr="")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    probe = MinerUAdapter().probe()

    assert probe.parser == "mineru"
    assert probe.version == "3.4.5"
    assert probe.executable_path == executable
    assert calls[0][0] == [str(executable), "--version"]
    assert calls[0][1]["shell"] is False


def test_version_probe_rejects_malformed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_executable(monkeypatch)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="MinerU latest\n", stderr="")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    with pytest.raises(MinerUProbeError, match="unrecognized MinerU version output"):
        MinerUAdapter().probe()


def test_command_is_argument_list_and_paths_with_spaces_are_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = _mock_executable(monkeypatch)
    payload = b"verified source"
    input_pdf = tmp_path / "input files" / "source document.pdf"
    input_pdf.parent.mkdir()
    input_pdf.write_bytes(payload)
    output_root = tmp_path / "parser output with spaces"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout="mineru, version 3.4.5\n", stderr=""
            )
        _successful_parse_outputs(command)
        return subprocess.CompletedProcess(command, 0, stdout="parse complete\n", stderr="")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    run = MinerUAdapter().run(
        manifest=_manifest_for(payload),
        input_pdf=input_pdf,
        output_root=output_root,
    )

    parse_command, parse_options = calls[1]
    assert parse_command == [
        str(executable),
        "-p",
        str(input_pdf.resolve()),
        "-o",
        str((output_root / "3.4.5" / "pipeline" / "raw").resolve()),
        "-b",
        "pipeline",
    ]
    assert parse_options["shell"] is False
    assert parse_options["capture_output"] is True
    assert parse_options["timeout"] == 14_400.0
    assert run.input_sha256 == hashlib.sha256(payload).hexdigest()
    assert run.execution.exit_code == 0
    assert {artifact.kind for artifact in run.artifacts} >= {
        "markdown",
        "middle_json",
        "content_list_json",
        "content_list_v2_json",
        "image",
    }
    run_data = json.loads(
        (output_root / "3.4.5" / "pipeline" / "run.json").read_text(encoding="utf-8")
    )
    assert run_data["document_id"] == "test-document"
    assert run_data["parser_version"] == "3.4.5"


def test_nonzero_parser_exit_is_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_executable(monkeypatch)
    payload = b"verified source"
    input_pdf = tmp_path / "source.pdf"
    input_pdf.write_bytes(payload)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout="mineru, version 3.4.5\n", stderr=""
            )
        return subprocess.CompletedProcess(command, 7, stdout="", stderr="parser failed")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    with pytest.raises(MinerUExecutionError, match="status 7") as error_info:
        MinerUAdapter().run(
            manifest=_manifest_for(payload),
            input_pdf=input_pdf,
            output_root=tmp_path / "output",
        )

    assert error_info.value.exit_code == 7
    assert error_info.value.stderr == "parser failed"


def test_parser_timeout_is_clear(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _mock_executable(monkeypatch)
    payload = b"verified source"
    input_pdf = tmp_path / "source.pdf"
    input_pdf.write_bytes(payload)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout="mineru, version 3.4.5\n", stderr=""
            )
        raise subprocess.TimeoutExpired(command, 1.0, output="partial", stderr="stalled")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    with pytest.raises(MinerUTimeoutError, match="timed out") as error_info:
        MinerUAdapter(timeout_seconds=1.0).run(
            manifest=_manifest_for(payload),
            input_pdf=input_pdf,
            output_root=tmp_path / "output",
        )

    assert error_info.value.stdout == "partial"
    assert error_info.value.stderr == "stalled"


def test_changed_source_is_rejected_before_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = b"registered source"
    input_pdf = tmp_path / "source.pdf"
    input_pdf.write_bytes(b"changed source")
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    with pytest.raises(ArtifactError, match="failed verification"):
        MinerUAdapter().run(
            manifest=_manifest_for(expected),
            input_pdf=input_pdf,
            output_root=tmp_path / "output",
        )

    assert calls == 0


def test_artifact_discovery_records_relative_path_size_and_sha256(tmp_path: Path) -> None:
    raw_directory = tmp_path / "raw"
    nested = raw_directory / "document" / "auto"
    nested.mkdir(parents=True)
    payload = b"artifact bytes"
    artifact_path = nested / "document_middle.json"
    artifact_path.write_bytes(payload)

    artifacts = discover_artifacts(raw_directory)

    assert len(artifacts) == 1
    assert artifacts[0].relative_path == "document/auto/document_middle.json"
    assert artifacts[0].kind == "middle_json"
    assert artifacts[0].byte_size == len(payload)
    assert artifacts[0].sha256 == hashlib.sha256(payload).hexdigest()


def test_success_exit_without_core_outputs_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _mock_executable(monkeypatch)
    payload = b"verified source"
    input_pdf = tmp_path / "source.pdf"
    input_pdf.write_bytes(payload)

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--version":
            return subprocess.CompletedProcess(
                command, 0, stdout="mineru, version 3.4.5\n", stderr=""
            )
        raw_directory = Path(command[command.index("-o") + 1])
        raw_directory.mkdir(parents=True, exist_ok=True)
        (raw_directory / "source.md").write_text("# Incomplete\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("vlm_rag.parsers.mineru.subprocess.run", fake_run)

    with pytest.raises(MinerUOutputError, match="required inspection outputs"):
        MinerUAdapter().run(
            manifest=_manifest_for(payload),
            input_pdf=input_pdf,
            output_root=tmp_path / "output",
        )


def test_only_pipeline_backend_is_allowed() -> None:
    with pytest.raises(ValueError, match="unsupported MinerU backend"):
        MinerUAdapter(backend="vlm-engine")
