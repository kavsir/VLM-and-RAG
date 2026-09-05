"""Unit tests for golden document manifests and artifact verification."""

import hashlib
from datetime import UTC, date, datetime
from io import BytesIO
from pathlib import Path
from urllib import request

import pytest
from pydantic import ValidationError

from vlm_rag.registry import (
    ArtifactError,
    DocumentIdentity,
    DocumentManifest,
    DocumentVersion,
    FileArtifact,
    ManifestValidationError,
    SourceReference,
    fetch_artifact,
    load_manifest,
    verify_artifact,
)
from vlm_rag.registry.__main__ import main

VALID_MANIFEST = """\
manifest_schema_version: 1
document:
  id: test-document
  document_number: "1/QĐ-TEST"
  normalized_document_number: 1/QD-TEST
  title: Test document
  document_type: Decision
  issuer: Test issuer
version:
  id: v1
  document_id: test-document
  issued_on: 2026-05-13
  effective_on: 2026-05-13
source:
  source_type: official_government_portal
  publisher: Test publisher
  signer: Test signer
  landing_page_url: https://example.gov.test/documents/1
  asset_url: https://files.example.gov.test/documents/1.pdf
  retrieved_at: 2026-09-03T00:00:00Z
artifact:
  filename: source.pdf
  media_type: application/pdf
  byte_size: 12
  sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
"""


def _write_manifest(tmp_path: Path, content: str = VALID_MANIFEST) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _artifact_for(payload: bytes) -> FileArtifact:
    return FileArtifact(
        filename="source.pdf",
        media_type="application/pdf",
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _source() -> SourceReference:
    return SourceReference(
        source_type="official_government_portal",
        publisher="Test publisher",
        signer="Test signer",
        landing_page_url="https://example.gov.test/documents/1",
        asset_url="https://files.example.gov.test/documents/1.pdf",
        retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
    )


class FakeResponse(BytesIO):
    """Minimal in-memory response used to keep retrieval tests offline."""

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return "https://files.example.gov.test/documents/1.pdf"


def test_valid_manifest_loads_offline(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path))

    assert isinstance(manifest, DocumentManifest)
    assert manifest.document.document_number == "1/QĐ-TEST"
    assert manifest.document.normalized_document_number == "1/QD-TEST"


def test_effective_on_accepts_dated_value_and_explicit_null(tmp_path: Path) -> None:
    dated = load_manifest(_write_manifest(tmp_path, VALID_MANIFEST))
    assert dated.version.effective_on == date(2026, 5, 13)

    explicit_null = VALID_MANIFEST.replace("  effective_on: 2026-05-13\n", "  effective_on: null\n")
    undated = load_manifest(_write_manifest(tmp_path, explicit_null))
    assert undated.version.effective_on is None


def test_effective_on_cannot_be_omitted(tmp_path: Path) -> None:
    missing = VALID_MANIFEST.replace("  effective_on: 2026-05-13\n", "")
    with pytest.raises(ManifestValidationError, match=r"(?s)effective_on.*Field required"):
        load_manifest(_write_manifest(tmp_path, missing))


def test_missing_required_field_fails_clearly(tmp_path: Path) -> None:
    content = VALID_MANIFEST.replace("  signer: Test signer\n", "")

    with pytest.raises(ManifestValidationError, match=r"(?s)source\.signer.*Field required"):
        load_manifest(_write_manifest(tmp_path, content))


def test_malformed_sha256_fails_clearly(tmp_path: Path) -> None:
    content = VALID_MANIFEST.replace("a" * 64, "not-a-sha256")

    with pytest.raises(ManifestValidationError, match="sha256"):
        load_manifest(_write_manifest(tmp_path, content))


def test_unsupported_schema_version_fails_clearly(tmp_path: Path) -> None:
    content = VALID_MANIFEST.replace("manifest_schema_version: 1", "manifest_schema_version: 2")

    with pytest.raises(ManifestValidationError, match="manifest_schema_version"):
        load_manifest(_write_manifest(tmp_path, content))


def test_invalid_date_fails_clearly(tmp_path: Path) -> None:
    content = VALID_MANIFEST.replace("issued_on: 2026-05-13", "issued_on: '2026-02-30'")

    with pytest.raises(ManifestValidationError, match="issued_on"):
        load_manifest(_write_manifest(tmp_path, content))


def test_local_fixture_checksum_succeeds(tmp_path: Path) -> None:
    payload = b"registered bytes"
    path = tmp_path / "source.pdf"
    path.write_bytes(payload)

    verified = verify_artifact(path, _artifact_for(payload))

    assert verified.byte_size == len(payload)
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()


def test_mutated_local_fixture_checksum_fails(tmp_path: Path) -> None:
    original = b"registered bytes"
    path = tmp_path / "source.pdf"
    path.write_bytes(b"mutated bytes")

    with pytest.raises(ArtifactError, match="failed verification"):
        verify_artifact(path, _artifact_for(original))


def test_document_identity_and_version_are_distinct() -> None:
    identity = DocumentIdentity(
        id="test-document",
        document_number="1/QĐ-TEST",
        normalized_document_number="1/QD-TEST",
        title="Test document",
        document_type="Decision",
        issuer="Test issuer",
    )
    version = DocumentVersion(
        id="v1",
        document_id=identity.id,
        issued_on=date(2026, 5, 13),
        effective_on=date(2026, 5, 13),
    )

    assert identity.id == version.document_id
    assert identity.id != version.id


def test_version_cannot_reference_another_document(tmp_path: Path) -> None:
    content = VALID_MANIFEST.replace("document_id: test-document", "document_id: other-document")

    with pytest.raises(ManifestValidationError, match=r"version\.document_id"):
        load_manifest(_write_manifest(tmp_path, content))


def test_landing_page_and_asset_urls_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="must be distinct"):
        SourceReference(
            source_type="official_government_portal",
            publisher="Test publisher",
            signer="Test signer",
            landing_page_url="https://example.gov.test/source",
            asset_url="https://example.gov.test/source",
            retrieved_at=datetime(2026, 9, 3, tzinfo=UTC),
        )


def test_fetch_streams_and_atomically_publishes_verified_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = b"downloaded registered bytes"

    def fake_urlopen(_request: request.Request, **_kwargs: object) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr("vlm_rag.registry.artifacts.request.urlopen", fake_urlopen)
    destination = tmp_path / "golden" / "source.pdf"

    verified = fetch_artifact(_source(), _artifact_for(payload), destination)

    assert destination.read_bytes() == payload
    assert verified.path == destination
    assert verified.final_url == "https://files.example.gov.test/documents/1.pdf"
    assert list(destination.parent.glob("*.part")) == []


def test_changed_remote_bytes_do_not_replace_known_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registered = b"registered bytes"

    def fake_urlopen(_request: request.Request, **_kwargs: object) -> FakeResponse:
        return FakeResponse(b"tampered bytes!!")

    monkeypatch.setattr("vlm_rag.registry.artifacts.request.urlopen", fake_urlopen)
    destination = tmp_path / "source.pdf"
    destination.write_bytes(registered)

    with pytest.raises(ArtifactError, match="failed verification"):
        fetch_artifact(_source(), _artifact_for(registered), destination)

    assert destination.read_bytes() == registered
    assert list(tmp_path.glob("*.part")) == []


def test_validate_cli_is_offline_and_reports_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["validate", str(_write_manifest(tmp_path))])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "document=test-document version=v1" in output
    assert "landing page:" in output
    assert "asset:" in output
