"""Bounded, non-retrying OpenAI-compatible Chat Completions transport."""

import hashlib
import json
import time
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VLM_RAG_MODEL_", env_file=".env", extra="ignore")
    base_url: str
    api_key: SecretStr = SecretStr("")
    vlm: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    timeout_seconds: float = Field(default=60, gt=0, le=600)
    max_output_tokens: int = Field(default=2048, ge=1, le=16384)
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    max_requests: int = Field(default=0, ge=0, le=1000)

    @field_validator("base_url")
    @classmethod
    def safe_endpoint(cls, value: str) -> str:
        parsed = urlsplit(value)
        local = parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or (parsed.scheme != "https" and not (parsed.scheme == "http" and local))
        ):
            raise ValueError("endpoint requires HTTPS (HTTP allowed only on loopback), no secrets")
        return value.rstrip("/")


class ModelCallError(ValueError):
    """Sanitized failure: never include credentials or a provider error body."""


class Completion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    requested_model: str
    reported_model: str
    content: str
    raw_envelope: str
    response_sha256: str
    latency_ms: int
    usage: dict[str, int] = Field(default_factory=dict)


class ChatClient:
    """One caller-owned request budget shared by VLM and answer calls; no retries."""

    def __init__(self, settings: ModelSettings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.requests_sent = 0
        self._client = httpx.Client(
            timeout=settings.timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def complete(self, model: str, messages: list[dict[str, object]]) -> Completion:
        if self.requests_sent >= self.settings.max_requests:
            raise ModelCallError("request budget exhausted; configure max_requests explicitly")
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_object"},
            self.settings.token_parameter: self.settings.max_output_tokens,
        }
        headers = {"Content-Type": "application/json"}
        key = self.settings.api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.requests_sent += 1
        start = time.monotonic()
        try:
            with self._client.stream(
                "POST",
                self.settings.base_url + "/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    raise ModelCallError(f"model endpoint returned HTTP {response.status_code}")
                chunks = bytearray()
                for chunk in response.iter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > 4_000_000:
                        raise ModelCallError("model response exceeds 4 MB limit")
            raw = chunks.decode("utf-8")
            envelope = json.loads(raw)
            if not isinstance(envelope, dict):
                raise ModelCallError("invalid completion envelope")
            choices = envelope.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ModelCallError("expected exactly one completion choice")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
                raise ModelCallError("completion was refused, truncated, or unfinished")
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("refusal"):
                raise ModelCallError("completion refused or missing message")
            content = message.get("content")
            reported_model = envelope.get("model")
            if not isinstance(content, str) or not content.strip():
                raise ModelCallError("completion content must be non-empty text")
            if not isinstance(reported_model, str) or not reported_model:
                raise ModelCallError("completion must report model identity")
            raw_usage = envelope.get("usage", {})
            usage = (
                {
                    key: value
                    for key, value in raw_usage.items()
                    if type(value) is int and value >= 0
                }
                if isinstance(raw_usage, dict)
                else {}
            )
            return Completion(
                requested_model=model,
                reported_model=reported_model,
                content=content,
                raw_envelope=raw,
                response_sha256=hashlib.sha256(chunks).hexdigest(),
                latency_ms=int((time.monotonic() - start) * 1000),
                usage=usage,
            )
        except (httpx.HTTPError, UnicodeError, json.JSONDecodeError):
            raise ModelCallError("model transport or JSON decoding failed") from None
