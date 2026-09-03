"""Pytest configuration and global fixtures."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from vlm_rag.config import get_settings


@pytest.fixture(autouse=True)
def isolate_settings_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None, None, None]:
    """Isolate tests from ambient VLM_RAG_* variables and repository .env files."""
    get_settings.cache_clear()

    # Strip any ambient VLM_RAG_* variables
    for key in list(os.environ.keys()):
        if key.startswith("VLM_RAG_"):
            monkeypatch.delenv(key, raising=False)

    # Prevent reading repo-level .env by changing working directory to an isolated tmp_path
    monkeypatch.chdir(tmp_path)

    yield

    get_settings.cache_clear()
