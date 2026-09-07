"""Provider-neutral VLM protocol and deterministic offline replay implementation."""

from pathlib import Path
from typing import Protocol

from pydantic import Field, model_validator

from vlm_rag.vlm.models import RawVLMResponse, VLMModel, VLMRequestRecord, canonical_request_sha256


class VLMClient(Protocol):
    def invoke(self, request: VLMRequestRecord) -> RawVLMResponse: ...


class ReplayFixture(VLMModel):
    fixture_version: int = Field(ge=1, le=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response: RawVLMResponse


class ReplayEvidence(VLMModel):
    """Versioned, explicitly non-quality replay evidence envelope."""

    replay_evidence_schema_version: int = Field(ge=1, le=1)
    purpose: str = Field(pattern="^synthetic_contract_only$")
    quality_claim_permitted: bool = False
    request_record: VLMRequestRecord
    fixture: ReplayFixture

    @model_validator(mode="after")
    def validate_identity(self) -> "ReplayEvidence":
        request = self.request_record.request
        if self.quality_claim_permitted:
            raise ValueError("synthetic replay evidence cannot support a quality claim")
        if self.fixture.request_sha256 != canonical_request_sha256(request):
            raise ValueError("replay evidence request SHA-256 mismatch")
        if self.fixture.response.request_id != request.request_id:
            raise ValueError("replay evidence request identity mismatch")
        if self.fixture.response.model_id != self.request_record.model_id:
            raise ValueError("replay evidence model identity mismatch")
        return self


def load_replay_evidence(path: Path) -> ReplayEvidence:
    """Load a strict offline replay envelope without network or model access."""
    return ReplayEvidence.model_validate_json(path.read_bytes())


def execute_vlm_request(
    request: VLMRequestRecord,
    client: VLMClient,
    *,
    live_vlm: bool = False,
) -> RawVLMResponse:
    """Invoke replay by default; require explicit opt-in for any external client."""
    if not live_vlm and not isinstance(client, ReplayVLMClient):
        raise ValueError("external VLM execution requires explicit live_vlm=True")
    return client.invoke(request)


class ReplayVLMClient:
    """Return committed raw responses only when request identity matches exactly."""

    def __init__(self, fixtures: tuple[ReplayFixture, ...]) -> None:
        self._fixtures = {fixture.response.request_id: fixture for fixture in fixtures}
        if len(self._fixtures) != len(fixtures):
            raise ValueError("duplicate replay request_id")

    def invoke(self, request: VLMRequestRecord) -> RawVLMResponse:
        fixture = self._fixtures.get(request.request.request_id)
        if fixture is None:
            raise ValueError("no replay fixture for request")
        if fixture.request_sha256 != canonical_request_sha256(request.request):
            raise ValueError("replay fixture/request identity mismatch")
        if fixture.response.model_id != request.model_id:
            raise ValueError("replay model identity mismatch")
        return fixture.response


__all__ = [
    "ReplayEvidence",
    "ReplayFixture",
    "ReplayVLMClient",
    "VLMClient",
    "execute_vlm_request",
    "load_replay_evidence",
]
