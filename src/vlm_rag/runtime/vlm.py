"""Real VLM invocation with retained images, raw responses and strict normalization."""

import base64
import hashlib
import json
from pathlib import Path

from vlm_rag.physical_ir.v1 import PhysicalDocumentV1
from vlm_rag.runtime.chat import ChatClient
from vlm_rag.semantic_ir import build_semantic_document, semantic_document_to_json
from vlm_rag.structural_ir.models import StructuralDocument
from vlm_rag.vlm.chat_prompt import CHAT_PROMPT_VERSION, task_instruction
from vlm_rag.vlm.evidence import ResolvedVisualEvidence, VisualEvidenceResolver
from vlm_rag.vlm.models import RawVLMResponse, VLMRequestRecord
from vlm_rag.vlm.normalization import normalize_vlm_response, to_visual_observation
from vlm_rag.vlm.prompt import build_visual_evidence_prompt
from vlm_rag.vlm.selector import SelectionBudget, select_visual_evidence


def write_new_json(path: Path, value: object) -> None:
    with path.open("xb") as stream:
        stream.write(
            (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )


def image_content(image: ResolvedVisualEvidence) -> dict[str, object]:
    signatures = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}
    signature = signatures.get(image.media_type)
    if (
        signature is None
        or not image.data.startswith(signature)
        or image.byte_size != len(image.data)
        or image.byte_size > 10_000_000
        or hashlib.sha256(image.data).hexdigest() != image.sha256
    ):
        raise ValueError("unsupported or invalid image evidence (PNG/JPEG, at most 10 MB)")
    encoded = base64.b64encode(image.data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{image.media_type};base64,{encoded}"}}


def enrich_document(
    physical: PhysicalDocumentV1,
    structural: StructuralDocument,
    *,
    resolver: VisualEvidenceResolver,
    client: ChatClient,
    output: Path,
    max_requests: int = 2,
    live: bool = False,
) -> Path:
    if not live:
        raise ValueError("VLM enrichment requires explicit live opt-in")
    if not 1 <= max_requests <= 1000:
        raise ValueError("max_requests must be 1..1000")
    baseline = build_semantic_document(physical, structural)
    selection = select_visual_evidence(
        physical,
        structural=structural,
        budget=SelectionBudget(
            max_requests_per_page=2,
            max_requests_per_document=max_requests,
        ),
    )
    if len(selection.requests) > client.settings.max_requests - client.requests_sent:
        raise ValueError("selection exceeds remaining model request budget")
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "selection.json", selection.model_dump(mode="json"))
    observations = []
    try:
        for index, request in enumerate(selection.requests):
            image = resolver.resolve(request)
            part = image_content(image)  # verify before any request
            folder = output / f"request-{index:04d}"
            folder.mkdir()
            with (folder / "image.bin").open("xb") as stream:
                stream.write(image.data)
            record = VLMRequestRecord(
                request=request,
                model_id=client.settings.vlm,
                provider_protocol="openai-chat-completions-v1",
                prompt_version=CHAT_PROMPT_VERSION,
                image_sha256=image.sha256,
                image_byte_size=image.byte_size,
                parameters={"output_limit": client.settings.max_output_tokens},
            )
            write_new_json(folder / "request.json", record.model_dump(mode="json"))
            completion = client.complete(
                client.settings.vlm,
                [
                    {"role": "system", "content": build_visual_evidence_prompt(request.task_type)},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": task_instruction(request.task_type, request.request_id),
                            },
                            part,
                        ],
                    },
                ],
            )
            write_new_json(folder / "completion.json", completion.model_dump(mode="json"))
            response = RawVLMResponse(
                request_id=request.request_id,
                model_id=record.model_id,
                provider_protocol=record.provider_protocol,
                raw_response=completion.content,
                raw_response_sha256=hashlib.sha256(completion.content.encode("utf-8")).hexdigest(),
                finish_state="completed",
                latency_ms=completion.latency_ms,
            )
            write_new_json(folder / "raw.json", response.model_dump(mode="json"))
            observation = normalize_vlm_response(record, response, image)
            write_new_json(folder / "observation.json", observation.model_dump(mode="json"))
            observations.append(to_visual_observation(observation))
        document = (
            build_semantic_document(physical, structural, visual_observations=tuple(observations))
            if observations
            else baseline
        )
        target = output / "semantic.json"
        with target.open("xb") as stream:
            stream.write(semantic_document_to_json(document).encode("utf-8"))
        write_new_json(
            output / "run.json",
            {
                "status": "completed",
                "executed_requests": len(observations),
                "semantic_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "endpoint": client.settings.base_url,
                "quality_claim": False,
            },
        )
        return target
    except Exception:
        write_new_json(
            output / "failure.json",
            {
                "status": "failed",
                "completed_observations": len(observations),
                "requests_attempted": client.requests_sent,
                "message": "See retained evidence; no completed semantic output is published.",
            },
        )
        raise
