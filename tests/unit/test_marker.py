"""Offline contracts for the external Marker adapter and v0 normalizer."""

import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from vlm_rag.normalizers.marker import MarkerNormalizationError, MarkerPhysicalNormalizer
from vlm_rag.parsers.marker import (
    MARKER_BACKEND,
    MARKER_DEVICE_POLICY,
    MarkerAdapter,
    MarkerExecutableNotFoundError,
    MarkerExecutionError,
    MarkerExecutionMetadata,
    MarkerOutputError,
    MarkerProbeError,
    MarkerRun,
    MarkerTimeoutError,
    discover_marker_document_json,
)
from vlm_rag.physical_ir import BlockDisposition, BlockKind
from vlm_rag.registry import (
    ArtifactError,
    DocumentIdentity,
    DocumentManifest,
    DocumentVersion,
    FileArtifact,
    SourceReference,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "marker"


def _probe_output(
    *,
    version: str = "2.0.0",
    python_version: str = "3.12.10",
    device: str = "cpu",
    configured_device: str = "cpu",
) -> str:
    return (
        json.dumps(
            {
                "version": version,
                "python_version": python_version,
                "device": device,
                "configured_device": configured_device,
            }
        )
        + "\n"
    )


def _manifest_for(payload: bytes) -> DocumentManifest:
    return DocumentManifest(
        manifest_schema_version=1,
        document=DocumentIdentity(
            id="test-document",
            document_number="1/QD-TEST",
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
            source_type="official",
            publisher="Test publisher",
            signer="Test signer",
            landing_page_url="https://example.gov.test/document",
            asset_url="https://files.example.gov.test/document.pdf",
            retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
        ),
        artifact=FileArtifact(
            filename="source.pdf",
            media_type="application/pdf",
            byte_size=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
    )


def _make_executables(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "Marker Runtime" / "marker_single.exe"
    python = tmp_path / "Marker Runtime" / "python.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"")
    python.write_bytes(b"")
    return executable, python


def _write_successful_output(command: list[str]) -> None:
    raw_directory = Path(command[command.index("--output_dir") + 1])
    output = raw_directory / "source"
    output.mkdir(parents=True)
    (output / "source.json").write_bytes((FIXTURES / "sample.json").read_bytes())
    (output / "source_meta.json").write_bytes((FIXTURES / "sample_meta.json").read_bytes())


def _write_normalizer_run(raw_directory: Path, **overrides: object) -> None:
    value: dict[str, object] = {
        "parser": "marker",
        "parser_version": "2.0.0",
        "backend": "fast-no-ocr",
        "mode": "fast",
        "disable_ocr": True,
        "output_format": "json",
        "document_id": "test-document",
        "version_id": "v1",
        "input_sha256": hashlib.sha256(b"source").hexdigest(),
    }
    value.update(overrides)
    (raw_directory.parent / "run.json").write_text(json.dumps(value), encoding="utf-8")


def _make_raw(tmp_path: Path) -> tuple[Path, DocumentManifest]:
    raw = tmp_path / "run" / "raw"
    output = raw / "source"
    output.mkdir(parents=True)
    (output / "source.json").write_bytes((FIXTURES / "sample.json").read_bytes())
    (output / "source_meta.json").write_bytes((FIXTURES / "sample_meta.json").read_bytes())
    _write_normalizer_run(raw)
    return raw, _manifest_for(b"source")


def _load_tree(raw: Path) -> dict[str, object]:
    path = raw / "source" / "source.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_tree(raw: Path, value: object, name: str = "source.json") -> Path:
    path = raw / "source" / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_missing_marker_executable_has_domain_error(tmp_path: Path) -> None:
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    with pytest.raises(MarkerExecutableNotFoundError, match="does not exist"):
        MarkerAdapter(executable=tmp_path / "missing-marker.exe", python_executable=python).probe()


def test_package_version_probe_uses_importlib_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable, python = _make_executables(tmp_path)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, _probe_output(), "")

    monkeypatch.setattr("vlm_rag.parsers.marker.subprocess.run", fake_run)
    probe = MarkerAdapter(executable=executable, python_executable=python).probe()
    assert probe.version == "2.0.0"
    assert probe.python_version == "3.12.10"
    assert probe.device == "cpu"
    assert calls[0][0][0] == str(python.resolve())
    assert "importlib.metadata.version('marker-pdf')" in calls[0][0][2]
    assert "torch" in calls[0][0][2]
    assert calls[0][1]["shell"] is False
    environment = calls[0][1]["env"]
    assert isinstance(environment, dict)
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["HIP_VISIBLE_DEVICES"] == ""
    assert environment["ROCR_VISIBLE_DEVICES"] == ""
    assert environment["TORCH_DEVICE"] == "cpu"


def test_probe_rejects_wrong_python_minor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    executable, python = _make_executables(tmp_path)
    monkeypatch.setattr(
        "vlm_rag.parsers.marker.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, _probe_output(python_version="3.13.7"), ""
        ),
    )
    with pytest.raises(MarkerProbeError, match=r"Python 3\.12"):
        MarkerAdapter(executable=executable, python_executable=python).probe()


def test_probe_rejects_an_accelerator_after_device_isolation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable, python = _make_executables(tmp_path)
    monkeypatch.setattr(
        "vlm_rag.parsers.marker.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 0, _probe_output(device="cuda"), ""
        ),
    )
    with pytest.raises(MarkerProbeError, match="CPU-only policy"):
        MarkerAdapter(executable=executable, python_executable=python).probe()


def test_explicit_executables_must_share_one_environment(tmp_path: Path) -> None:
    executable = tmp_path / "marker-env" / "marker_single.exe"
    python = tmp_path / "python-env" / "python.exe"
    executable.parent.mkdir()
    python.parent.mkdir()
    executable.write_bytes(b"")
    python.write_bytes(b"")
    with pytest.raises(MarkerProbeError, match="same environment directory"):
        MarkerAdapter(executable=executable, python_executable=python).probe()


def test_path_resolved_executables_must_share_one_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "marker-env" / "marker_single.exe"
    python = tmp_path / "python-env" / "python.exe"
    marker.parent.mkdir()
    python.parent.mkdir()
    marker.write_bytes(b"")
    python.write_bytes(b"")
    resolved = {"marker_single": str(marker), "python": str(python)}
    monkeypatch.setattr("vlm_rag.parsers.marker.shutil.which", lambda name: resolved.get(name))
    with pytest.raises(MarkerProbeError, match="same environment directory"):
        MarkerAdapter().probe()


def test_safe_command_and_paths_with_spaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable, python = _make_executables(tmp_path)
    source = tmp_path / "input files" / "source document.pdf"
    source.parent.mkdir()
    source.write_bytes(b"source")
    output = tmp_path / "output with spaces"
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, _probe_output(), "")
        _write_successful_output(command)
        return subprocess.CompletedProcess(command, 0, "complete\n", "warning\n")

    monkeypatch.setattr("vlm_rag.parsers.marker.subprocess.run", fake_run)
    run = MarkerAdapter(executable=executable, python_executable=python).run(
        manifest=_manifest_for(b"source"), input_pdf=source, output_root=output
    )
    parse_command, options = calls[1]
    assert parse_command == [
        str(executable.resolve()),
        str(source.resolve()),
        "--mode",
        "fast",
        "--disable_ocr",
        "--output_format",
        "json",
        "--output_dir",
        str((output / "2.0.0" / MARKER_BACKEND / "raw").resolve()),
        "--JSONRenderer_keep_pageheader_in_output",
        "--JSONRenderer_keep_pagefooter_in_output",
    ]
    assert options["shell"] is False
    assert options["capture_output"] is True
    assert calls[0][1]["env"] == options["env"]
    assert run.mode == "fast" and run.disable_ocr is True
    assert run.device == "cpu" and run.device_policy == MARKER_DEVICE_POLICY
    assert dict(run.device_environment) == {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "TORCH_DEVICE": "cpu",
    }
    run_dir = output / "2.0.0" / MARKER_BACKEND
    manifest_data = json.loads((run_dir / "artifact_manifest.json").read_text())
    retained = {item["relative_path"] for item in manifest_data["artifacts"]}
    assert retained >= {"raw/source/source.json", "run.json", "stdout.log", "stderr.log"}


def test_nonzero_exit_and_timeout_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable, python = _make_executables(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    responses: list[str] = ["nonzero", "timeout"]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0, _probe_output(), "")
        behavior = responses.pop(0)
        if behavior == "nonzero":
            return subprocess.CompletedProcess(command, 7, "partial", "failed")
        raise subprocess.TimeoutExpired(command, 1, output="partial", stderr="stalled")

    monkeypatch.setattr("vlm_rag.parsers.marker.subprocess.run", fake_run)
    adapter = MarkerAdapter(executable=executable, python_executable=python)
    with pytest.raises(MarkerExecutionError, match="status 7") as nonzero:
        adapter.run(
            manifest=_manifest_for(b"source"),
            input_pdf=source,
            output_root=tmp_path / "failed",
        )
    assert nonzero.value.stdout == "partial" and nonzero.value.stderr == "failed"
    failed_run = tmp_path / "failed" / "2.0.0" / MARKER_BACKEND
    assert (failed_run / "stdout.log").read_text(encoding="utf-8") == "partial"
    assert (failed_run / "stderr.log").read_text(encoding="utf-8") == "failed"
    with pytest.raises(MarkerTimeoutError, match="timed out") as timeout:
        adapter.run(
            manifest=_manifest_for(b"source"),
            input_pdf=source,
            output_root=tmp_path / "timed-out",
        )
    assert timeout.value.stdout == "partial" and timeout.value.stderr == "stalled"
    timed_out_run = tmp_path / "timed-out" / "2.0.0" / MARKER_BACKEND
    assert (timed_out_run / "stdout.log").read_text(encoding="utf-8") == "partial"
    assert (timed_out_run / "stderr.log").read_text(encoding="utf-8") == "stalled"


def test_checksum_mismatch_prevents_every_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable, python = _make_executables(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"changed")
    calls = 0

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("vlm_rag.parsers.marker.subprocess.run", fake_run)
    with pytest.raises(ArtifactError, match="failed verification"):
        MarkerAdapter(executable=executable, python_executable=python).run(
            manifest=_manifest_for(b"source"),
            input_pdf=source,
            output_root=tmp_path / "output",
        )
    assert calls == 0


def test_document_json_discovery_missing_ambiguous_and_sidecar(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source_meta.json").write_bytes((FIXTURES / "sample_meta.json").read_bytes())
    with pytest.raises(MarkerOutputError, match="missing Marker document JSON"):
        discover_marker_document_json(raw)
    (raw / "one.json").write_bytes((FIXTURES / "sample.json").read_bytes())
    assert discover_marker_document_json(raw).name == "one.json"
    (raw / "two.json").write_bytes((FIXTURES / "sample.json").read_bytes())
    with pytest.raises(MarkerOutputError, match="ambiguous Marker document JSON"):
        discover_marker_document_json(raw)


def test_document_json_discovery_does_not_accept_a_bare_page_list(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    document = json.loads((FIXTURES / "sample.json").read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    (raw / "pages.json").write_text(json.dumps(document["children"]), encoding="utf-8")
    with pytest.raises(MarkerOutputError, match="missing Marker document JSON"):
        discover_marker_document_json(raw)


def test_observed_mapping_html_heading_geometry_order_and_indices(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    document = MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)
    assert document.page_count == 1
    assert document.pages[0].width == 200.0 and document.pages[0].height == 400.0
    blocks = document.pages[0].blocks
    assert len(blocks) == 6
    assert [block.reading_order for block in blocks] == list(range(6))
    assert [block.provenance.source_raw_index for block in blocks] == list(range(6))
    assert [block.id for block in blocks] == [
        f"test-document_v1_p0000_b{index:04d}" for index in range(6)
    ]
    assert (blocks[0].kind, blocks[0].disposition) == (
        BlockKind.HEADER,
        BlockDisposition.DISCARDED,
    )
    assert (blocks[1].kind, blocks[1].disposition) == (
        BlockKind.TEXT,
        BlockDisposition.CONTENT,
    )
    assert blocks[1].text == "Hello world & Hà Nội\nnext line"
    assert blocks[1].bbox.model_dump() == {
        "x0": 100.0,
        "y0": 100.0,
        "x1": 500.0,
        "y1": 500.0,
        "coordinate_system": "normalized_1000",
    }
    assert blocks[2].kind == BlockKind.TITLE and blocks[2].heading_level == 3
    assert (blocks[3].kind, blocks[3].disposition, blocks[3].text) == (
        BlockKind.UNKNOWN,
        BlockDisposition.DISCARDED,
        "1",
    )
    assert blocks[4].kind == BlockKind.UNKNOWN
    assert blocks[4].disposition == BlockDisposition.CONTENT
    assert blocks[5].kind == BlockKind.UNKNOWN
    assert blocks[5].text == "First\nSecond"


def test_heading_level_is_not_fabricated_from_section_hierarchy(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[2], dict)
    children[2]["html"] = "<p>Heading without tag</p>"
    _write_tree(raw, tree)
    block = MarkerPhysicalNormalizer().normalize(raw, manifest=manifest).pages[0].blocks[2]
    assert block.kind == BlockKind.TITLE and block.heading_level is None


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("document_id", "other-document", "document_id"),
        ("version_id", "v2", "version_id"),
        ("source_artifact_sha256", "0" * 64, "source_artifact_sha256"),
        ("parser_version", "1.9.0", "parser_version"),
        ("parser_backend", "fast", "parser_backend"),
        ("mode", "balanced", "mode"),
        ("disable_ocr", False, "disable_ocr"),
        ("output_format", "html", "output_format"),
    ],
)
def test_provenance_conflicts_are_rejected(
    tmp_path: Path, field: str, bad_value: object, message: str
) -> None:
    raw, manifest = _make_raw(tmp_path)
    with pytest.raises(MarkerNormalizationError, match=message):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest, **{field: bad_value})


def test_manifest_run_object_run_json_and_arguments_must_all_agree(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    run = MarkerRun(
        schema_version=1,
        status="succeeded",
        parser="marker",
        parser_version="2.0.0",
        backend="fast-no-ocr",
        mode="fast",
        disable_ocr=True,
        output_format="json",
        executable_path="marker_single",
        python_path="python",
        python_version="3.12.10",
        device="cpu",
        device_policy=MARKER_DEVICE_POLICY,
        device_environment=(("CUDA_VISIBLE_DEVICES", ""),),
        document_id="test-document",
        version_id="v1",
        input_sha256=manifest.artifact.sha256,
        output_directory=str(raw.parent),
        execution=MarkerExecutionMetadata(
            command=("marker_single",),
            started_at="2026-09-04T00:00:00+00:00",
            completed_at="2026-09-04T00:00:01+00:00",
            duration_seconds=1.0,
            exit_code=0,
            stdout_log="stdout.log",
            stderr_log="stderr.log",
        ),
        artifacts=(),
    )
    document = MarkerPhysicalNormalizer().normalize(
        raw,
        manifest=manifest,
        run=run,
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256=manifest.artifact.sha256,
        parser="marker",
        parser_version="2.0.0",
        parser_backend="fast-no-ocr",
        mode="fast",
        disable_ocr=True,
        output_format="json",
    )
    assert document.document_id == "test-document"
    with pytest.raises(MarkerNormalizationError, match="document_id"):
        MarkerPhysicalNormalizer().normalize(
            raw, manifest=manifest, run=replace(run, document_id="other")
        )


def test_global_raw_index_continues_across_pages(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    second = json.loads(json.dumps(pages[0]))
    second["id"] = "/page/1/Page/100"
    children = second["children"]
    assert isinstance(children, list)
    second["children"] = children[:1]
    pages.append(second)
    _write_tree(raw, tree)
    document = MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)
    assert document.pages[1].blocks[0].provenance.source_raw_index == 6


def test_missing_provenance_and_malformed_run_json_fail(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    (raw.parent / "run.json").unlink()
    with pytest.raises(MarkerNormalizationError, match="parser_version"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)
    (raw.parent / "run.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(MarkerNormalizationError, match="cannot read JSON"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


def test_malformed_root_page_and_geometry_are_rejected(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    _write_tree(raw, {"block_type": "NotDocument", "children": []})
    with pytest.raises(MarkerNormalizationError, match="missing Marker document JSON"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)

    _write_tree(raw, {"block_type": "Document", "children": [{"block_type": "Text"}]})
    with pytest.raises(MarkerNormalizationError, match="not a Page"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)

    _write_tree(raw, json.loads((FIXTURES / "sample.json").read_text(encoding="utf-8")))
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    pages[0]["bbox"] = [10.0, 20.0, 10.0, 420.0]
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="non-positive dimensions"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


def test_bbox_outside_page_is_rejected_without_clipping(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[1], dict)
    children[1]["bbox"] = [-10.0, 60.0, 110.0, 220.0]
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="outside page bounds"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


@pytest.mark.parametrize(
    ("coordinate", "value", "bbox_field", "expected"),
    [
        (0, 10.0 - 0.5e-6, "x0", 0.0),
        (1, 20.0 - 0.5e-6, "y0", 0.0),
        (2, 210.0 + 0.5e-6, "x1", 1000.0),
        (3, 420.0 + 0.5e-6, "y1", 1000.0),
    ],
)
def test_bbox_roundoff_within_tolerance_snaps_to_nonzero_origin_page_bounds(
    tmp_path: Path, coordinate: int, value: float, bbox_field: str, expected: float
) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[1], dict)
    children[1]["bbox"] = [10.0, 20.0, 210.0, 420.0]
    children[1]["bbox"][coordinate] = value
    _write_tree(raw, tree)
    bbox = MarkerPhysicalNormalizer().normalize(raw, manifest=manifest).pages[0].blocks[1].bbox
    assert getattr(bbox, bbox_field) == expected


@pytest.mark.parametrize(
    ("coordinate", "value"),
    [
        (0, 10.0 - 2e-6),
        (1, 20.0 - 2e-6),
        (2, 210.0 + 2e-6),
        (3, 420.0 + 2e-6),
    ],
)
def test_bbox_beyond_tolerance_is_rejected_on_every_page_edge(
    tmp_path: Path, coordinate: int, value: float
) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[1], dict)
    children[1]["bbox"] = [10.0, 20.0, 210.0, 420.0]
    children[1]["bbox"][coordinate] = value
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="outside page bounds"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


def test_content_ref_cycles_fail_with_a_domain_error(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[5], dict)
    list_group = children[5]
    list_group["html"] = "<content-ref src='/page/0/ListGroup/5'></content-ref>"
    list_group["children"] = [dict(list_group)]
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="cyclic Marker content-ref"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


def test_content_ref_depth_is_bounded_with_a_domain_error(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    blocks = pages[0]["children"]
    assert isinstance(blocks, list) and isinstance(blocks[5], dict)
    child: dict[str, object] = {
        "id": "/nested/65",
        "html": "<li>leaf</li>",
        "children": None,
    }
    for index in reversed(range(65)):
        next_id = str(child["id"])
        child = {
            "id": f"/nested/{index}",
            "html": f"<content-ref src='{next_id}'></content-ref>",
            "children": [child],
        }
    blocks[5]["html"] = "<content-ref src='/nested/0'></content-ref>"
    blocks[5]["children"] = [child]
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="nesting exceeds 64 levels"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


def test_content_ref_unresolved_and_duplicate_ids_remain_errors(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    tree = _load_tree(raw)
    pages = tree["children"]
    assert isinstance(pages, list) and isinstance(pages[0], dict)
    children = pages[0]["children"]
    assert isinstance(children, list) and isinstance(children[5], dict)
    list_group = children[5]
    list_group["html"] = "<content-ref src='/missing'></content-ref>"
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="unresolved Marker content-ref"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)

    tree = json.loads((FIXTURES / "sample.json").read_text(encoding="utf-8"))
    pages = tree["children"]
    list_group = pages[0]["children"][5]
    list_group["children"].append(dict(list_group["children"][0]))
    _write_tree(raw, tree)
    with pytest.raises(MarkerNormalizationError, match="duplicate nested Marker child id"):
        MarkerPhysicalNormalizer().normalize(raw, manifest=manifest)


@pytest.mark.parametrize(
    "relative_target",
    [
        "raw/physical_ir.json",
        "raw/source/source.json",
        "run.json",
        "stdout.log",
        "stderr.log",
        "artifact_manifest.json",
    ],
)
def test_raw_and_run_evidence_cannot_be_overwritten(tmp_path: Path, relative_target: str) -> None:
    raw, manifest = _make_raw(tmp_path)
    target = raw.parent / relative_target
    before = {path: path.read_bytes() for path in raw.rglob("*") if path.is_file()}
    target_before = target.read_bytes() if target.exists() else None
    with pytest.raises(MarkerNormalizationError, match=r"inside the raw|overwrite"):
        MarkerPhysicalNormalizer().normalize_to_file(raw, target, manifest=manifest)
    assert {path: path.read_bytes() for path in raw.rglob("*") if path.is_file()} == before
    if target_before is not None:
        assert target.read_bytes() == target_before


def test_authoritative_source_artifact_cannot_be_overwritten(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source")
    before = source.read_bytes()
    before_sha256 = hashlib.sha256(before).hexdigest()
    with pytest.raises(MarkerNormalizationError, match="authoritative source artifact"):
        MarkerPhysicalNormalizer().normalize_to_file(
            raw,
            source,
            source_artifact_path=source,
            manifest=manifest,
        )
    assert source.read_bytes() == before
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before_sha256


def test_serialization_is_deterministic_utf8_lf(tmp_path: Path) -> None:
    raw, manifest = _make_raw(tmp_path)
    first = tmp_path / "normalized" / "first.json"
    second = tmp_path / "normalized" / "second.json"
    normalizer = MarkerPhysicalNormalizer()
    normalizer.normalize_to_file(raw, first, manifest=manifest)
    normalizer.normalize_to_file(raw, second, manifest=manifest)
    assert first.read_bytes() == second.read_bytes()
    assert b"\r\n" not in first.read_bytes()
    assert (
        hashlib.sha256(first.read_bytes()).hexdigest()
        == hashlib.sha256(second.read_bytes()).hexdigest()
    )
