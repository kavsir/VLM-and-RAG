"""Unit tests for typed configuration and environment isolation."""

from pathlib import Path

import pytest

from vlm_rag.config import Settings, get_settings


def test_default_settings() -> None:
    """Verify default configuration values when no environment overrides exist."""
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.app_name == "vlm-rag"
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.log_level == "INFO"


def test_explicit_settings_instantiation() -> None:
    """Verify settings can be instantiated with explicit keyword arguments."""
    custom = Settings(
        app_name="custom-vlm-rag",
        environment="test",
        debug=True,
        log_level="DEBUG",
    )
    assert custom.app_name == "custom-vlm-rag"
    assert custom.environment == "test"
    assert custom.debug is True
    assert custom.log_level == "DEBUG"


def test_environment_variable_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that VLM_RAG_* environment variables override default settings."""
    monkeypatch.setenv("VLM_RAG_APP_NAME", "override-app")
    monkeypatch.setenv("VLM_RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("VLM_RAG_DEBUG", "true")
    monkeypatch.setenv("VLM_RAG_LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.app_name == "override-app"
    assert settings.environment == "production"
    assert settings.debug is True
    assert settings.log_level == "WARNING"


def test_dotenv_file_loading(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify that a local .env file in the working directory is loaded."""
    env_file = tmp_path / ".env"
    env_file.write_text("VLM_RAG_APP_NAME=from-env-file\nVLM_RAG_DEBUG=true\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.app_name == "from-env-file"
    assert settings.debug is True


def test_regression_ambient_injected_var_does_not_break_isolated_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test: verify ambient VLM_RAG_* injection does not break isolation.

    Simulates an ambient variable injected externally (e.g. VLM_RAG_APP_NAME=ci-injected).
    Verifies that explicit isolation cleanses the environment and preserves expected defaults.
    """
    # 1. Simulate external injection
    monkeypatch.setenv("VLM_RAG_APP_NAME", "ci-injected")
    get_settings.cache_clear()

    # When not isolated, the ambient variable is picked up
    injected_settings = get_settings()
    assert injected_settings.app_name == "ci-injected"

    # 2. Now isolate environment and clear cache (as done by the isolate_settings_env fixture)
    monkeypatch.delenv("VLM_RAG_APP_NAME", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    # Defaults must be restored deterministically
    isolated_settings = get_settings()
    assert isolated_settings.app_name == "vlm-rag"
    assert isolated_settings.environment == "development"
