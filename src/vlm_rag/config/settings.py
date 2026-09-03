"""Typed application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EnvironmentType = Literal["development", "staging", "production", "test"]


class Settings(BaseSettings):
    """Core application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="VLM_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="vlm-rag", description="Application name")
    environment: EnvironmentType = Field(default="development", description="Execution environment")
    debug: bool = Field(default=False, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Standard logging level")


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
