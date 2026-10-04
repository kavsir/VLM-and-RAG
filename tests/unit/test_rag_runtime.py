"""Offline adversarial tests for real transport, retrieval and VLM-to-RAG wiring."""

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from vlm_rag.__main__ import run
from vlm_rag.physical_ir import (
    BlockDisposition,
    BlockKindV1,
    BlockProvenanceV1,
    BoundingBox,
    PhysicalBlockV1,
    PhysicalDocumentV1,
    PhysicalPageV1,
    TextExtractionEvidence,
    TextExtractionMethod,
    physical_document_v1_to_json,
)
from vlm_rag.rag.answer import answer_question
from vlm_rag.rag.index import create_index, search
from vlm_rag.runtime.chat import ChatClient, ModelCallError, ModelSettings
from vlm_rag.runtime.pdf import PDFiumRenderer
from vlm_rag.runtime.vlm import enrich_document
from vlm_rag.semantic_ir import build_semantic_document, semantic_document_from_json
from vlm_rag.structural_ir import VietnameseStructuralExtractor
from vlm_rag.vlm.evidence import ResolvedVisualEvidence
from vlm_rag.vlm.models import RenderSpecification, VisualEvidenceRequest

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOioAAAAASUVORK5CYII="
)


def physical(text: str = "Hà Nội có quy hoạch đô thị.") -> PhysicalDocumentV1:
    block = PhysicalBlockV1(
        id="b0",
        page_index=0,
        reading_order=0,
        kind=BlockKindV1.TEXT,
        disposition=BlockDisposition.CONTENT,
        text=text,
        bbox=BoundingBox(x0=0.0, y0=0.0, x1=1000.0, y1=1000.0),
        provenance=BlockProvenanceV1(
            parser="fixture",
            parser_version="1",
            parser_backend="test",
            source_raw_artifact="fixture.json",
            source_raw_index=0,
            source_raw_type="text",
        ),
        text_extraction=TextExtractionEvidence(method=TextExtractionMethod.UNKNOWN),
    )
    return PhysicalDocumentV1(
        document_id="test-document",
        version_id="v1",
        source_artifact_sha256="a" * 64,
        parser="fixture",
        parser_version="1",
        parser_backend="test",
        page_count=1,
        pages=(PhysicalPageV1(page_index=0, width=100.0, height=100.0, blocks=(block,)),),
    )


def settings(monkeypatch: pytest.MonkeyPatch, *, budget: int = 2) -> ModelSettings:
    for key in tuple(__import__("os").environ):
        if key.startswith("VLM_RAG_MODEL_"):
            monkeypatch.delenv(key)
    return ModelSettings(
        _env_file=None,
        base_url="https://models.example/v1",
        vlm="vision",
        answer="answer",
        max_requests=budget,
    )


def envelope(content: object, *, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "provider-snapshot",
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(content)}}],
        },
    )


def test_index_is_reproducible_filtered_and_never_overwrites(tmp_path: Path) -> None:
    source = physical()
    document = build_semantic_document(source, VietnameseStructuralExtractor().extract(source))
    first, second = tmp_path / "first.db", tmp_path / "second.db"
    assert create_index(first, [document]) == {"documents": 1, "chunks": 1}
    create_index(second, [document])
    assert first.read_bytes() == second.read_bytes()
    hits = search(first, "ha noi")
    assert hits and hits[0].chunk.text == "Hà Nội có quy hoạch đô thị."
    assert hits[0].chunk.source_artifact_sha256 == "a" * 64
    assert hits[0].chunk.page_indexes == (0,)
    assert not search(first, "Hà Nội", document_id="other")
    assert not search(first, "Hà Nội", version_id="v2")
    before = first.read_bytes()
    with pytest.raises(FileExistsError):
        create_index(first, [document])
    assert first.read_bytes() == before
    with pytest.raises(ValueError, match="one representation"):
        create_index(tmp_path / "duplicates.db", [document, document])
    assert not (tmp_path / "duplicates.db").exists()
    assert not search(first, '" OR nonexistenttoken* --')


def test_answer_quotes_are_validated_and_no_hit_makes_no_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = physical()
    doc = build_semantic_document(source, VietnameseStructuralExtractor().extract(source))
    path = tmp_path / "index.db"
    create_index(path, [doc])
    hits = search(path, "Hà Nội")
    chunk = hits[0].chunk
    payload = {
        "status": "answered",
        "claims": [
            {
                "text": "Hà Nội có quy hoạch.",
                "citations": [{"chunk_id": chunk.id, "quote": "quy hoạch đô thị"}],
            }
        ],
    }
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return envelope(payload)

    client = ChatClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    try:
        result = answer_question("Quy hoạch Hà Nội?", hits, client=client)
        assert result.answer.status == "answered"
        assert (
            result.completion is not None
            and result.completion.reported_model == "provider-snapshot"
        )
        assert seen[0]["messages"][0]["role"] == "system"
        assert "untrusted" in seen[0]["messages"][0]["content"]
        assert seen[0]["max_tokens"] == 2048
        assert (
            answer_question("unknown?", [], client=client).answer.status == "insufficient_evidence"
        )
        assert client.requests_sent == 1
        payload["claims"][0]["citations"][0]["quote"] = "fabricated evidence"
        with pytest.raises(ValueError, match="non-verbatim"):
            answer_question("Quy hoạch?", hits, client=client)
    finally:
        client.close()


@pytest.mark.parametrize(
    "url",
    ["http://remote.example/v1", "https://u:p@host/v1", "https://host/v1?key=x", "file:///tmp/api"],
)
def test_endpoint_rejects_insecure_or_secret_urls(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = settings(monkeypatch).model_dump()
    config["base_url"] = url
    with pytest.raises(ValidationError):
        ModelSettings(_env_file=None, **config)


@pytest.mark.parametrize("status", [301, 401, 429, 500])
def test_transport_does_not_redirect_retry_or_echo_errors(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            text="sensitive-provider-error",
            headers={"location": "https://elsewhere.example"},
        )

    client = ChatClient(settings(monkeypatch, budget=1), transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ModelCallError, match=f"HTTP {status}") as error:
            client.complete("model", [])
        assert "sensitive" not in str(error.value)
        with pytest.raises(ModelCallError, match="budget"):
            client.complete("model", [])
        assert len(calls) == 1
    finally:
        client.close()


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls"])
def test_incomplete_outputs_fail_closed(finish: str, monkeypatch: pytest.MonkeyPatch) -> None:
    client = ChatClient(
        settings(monkeypatch), transport=httpx.MockTransport(lambda _: envelope({}, finish=finish))
    )
    try:
        with pytest.raises(ModelCallError, match="unfinished"):
            client.complete("model", [])
    finally:
        client.close()


class ImageResolver:
    def resolve(self, request: VisualEvidenceRequest) -> ResolvedVisualEvidence:
        assert request.structural_node_id
        return ResolvedVisualEvidence(
            data=PNG,
            sha256=hashlib.sha256(PNG).hexdigest(),
            byte_size=len(PNG),
            media_type="image/png",
        )


def test_vlm_to_index_to_grounded_answer_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = physical("uncertain source text")
    structural = VietnameseStructuralExtractor().extract(source)
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if payload["model"] == "vision":
            parts = payload["messages"][1]["content"]
            shape = json.loads(parts[0]["text"].split(": ", 1)[1])
            assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
            shape["transcription"] = "Hà Nội quy hoạch đến năm 2050."
            return envelope(shape)
        evidence = json.loads(payload["messages"][1]["content"])["evidence"][0]
        return envelope(
            {
                "status": "answered",
                "claims": [
                    {
                        "text": "Đến năm 2050.",
                        "citations": [
                            {"chunk_id": evidence["id"], "quote": "2050"},
                        ],
                    }
                ],
            }
        )

    client = ChatClient(settings(monkeypatch), transport=httpx.MockTransport(handler))
    try:
        output = tmp_path / "enriched"
        with pytest.raises(ValueError, match="opt-in"):
            enrich_document(
                source, structural, resolver=ImageResolver(), client=client, output=output
            )
        assert not output.exists() and client.requests_sent == 0
        target = enrich_document(
            source, structural, resolver=ImageResolver(), client=client, output=output, live=True
        )
        semantic = semantic_document_from_json(target.read_text(encoding="utf-8"))
        assert len(semantic.visual_observations) == 1
        assert (output / "request-0000/completion.json").exists()
        index = tmp_path / "index.db"
        create_index(index, [semantic])
        hits = search(index, "2050")
        assert hits[0].chunk.provenance == "vlm_transcription"
        assert hits[0].chunk.source_visual_observation_id
        result = answer_question("Hà Nội quy hoạch đến năm nào?", hits, client=client)
        assert result.answer.claims[0].citations[0].quote == "2050"
        assert len(calls) == 2
    finally:
        client.close()


def test_cli_offline_prepare_index_query(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    physical_path = tmp_path / "physical.json"
    physical_path.write_bytes(physical_document_v1_to_json(physical()).encode("utf-8"))
    prepared = tmp_path / "prepared"
    assert run(["prepare", "--physical", str(physical_path), "--output", str(prepared)]) == 0
    capsys.readouterr()
    index = tmp_path / "index.db"
    assert run(["index", str(prepared / "semantic.json"), "--output", str(index)]) == 0
    capsys.readouterr()
    assert run(["ask", "Hà Nội", "--index", str(index)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["mode"] == "extractive" and result["answer"]["claims"]
    assert result["completion"] is None


def test_pdf_crop_uses_top_left_coordinates() -> None:
    image_module = pytest.importorskip("PIL.Image")
    pytest.importorskip("pypdfium2")
    with image_module.new("RGB", (100, 100), "red") as image:
        image.paste("blue", (0, 50, 100, 100))
        buffer = io.BytesIO()
        image.save(buffer, format="PDF", resolution=72)
    renderer = PDFiumRenderer()
    crop = renderer.render(
        buffer.getvalue(),
        page_index=0,
        bbox=BoundingBox(x0=0.0, y0=0.0, x1=1000.0, y1=400.0),
        specification=RenderSpecification(dpi=72),
    )
    with image_module.open(io.BytesIO(crop.data)) as decoded:
        red, _, blue = decoded.getpixel((20, 20))
        assert red > 200 and blue < 50
        assert decoded.size == (100, 40)


def test_invalid_vlm_payload_retains_evidence_without_semantic_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = physical("uncertain text")
    structural = VietnameseStructuralExtractor().extract(source)
    client = ChatClient(
        settings(monkeypatch),
        transport=httpx.MockTransport(
            lambda _: envelope({"schema_version": 1, "request_id": "wrong-request"})
        ),
    )
    output = tmp_path / "failed-run"
    try:
        with pytest.raises(ValueError, match="identity mismatch"):
            enrich_document(
                source,
                structural,
                resolver=ImageResolver(),
                client=client,
                output=output,
                live=True,
            )
        assert (output / "failure.json").is_file()
        assert (output / "request-0000/raw.json").is_file()
        assert (output / "request-0000/completion.json").is_file()
        assert not (output / "semantic.json").exists()
        assert client.requests_sent == 1
    finally:
        client.close()


def test_zero_request_budget_never_calls_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("transport must not run with a zero request budget")

    client = ChatClient(settings(monkeypatch, budget=0), transport=httpx.MockTransport(forbidden))
    try:
        with pytest.raises(ModelCallError, match="budget"):
            client.complete("model", [])
        assert client.requests_sent == 0
    finally:
        client.close()


def test_unknown_citation_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = physical()
    doc = build_semantic_document(source, VietnameseStructuralExtractor().extract(source))
    path = tmp_path / "index.db"
    create_index(path, [doc])
    client = ChatClient(
        settings(monkeypatch),
        transport=httpx.MockTransport(
            lambda _: envelope(
                {
                    "status": "answered",
                    "claims": [
                        {
                            "text": "unsupported claim",
                            "citations": [
                                {"chunk_id": "invented", "quote": "Hà Nội"},
                            ],
                        }
                    ],
                }
            )
        ),
    )
    try:
        with pytest.raises(ValueError, match="unknown citation"):
            answer_question("Hà Nội?", search(path, "Hà Nội"), client=client)
    finally:
        client.close()


def test_cli_preserves_vietnamese_under_ascii_stdout(tmp_path: Path) -> None:
    source = physical()
    doc = build_semantic_document(source, VietnameseStructuralExtractor().extract(source))
    path = tmp_path / "index.db"
    create_index(path, [doc])
    result = subprocess.run(
        [sys.executable, "-m", "vlm_rag", "ask", "Hà Nội", "--index", str(path)],
        env={**os.environ, "PYTHONIOENCODING": "ascii"},
        capture_output=True,
        check=True,
    )
    answer = json.loads(result.stdout)
    assert "Hà Nội" in answer["answer"]["claims"][0]["text"]
