"""Training configuration validated without importing the GPU stack."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    base_model: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    method: Literal["lora", "qlora"]
    train_manifest: Path
    validation_manifest: Path
    image_root: Path
    output: Path
    max_steps: int = Field(ge=1, le=100000)
    learning_rate: float = Field(gt=0, le=0.01)
    gradient_accumulation_steps: int = Field(ge=1, le=128)
    lora_rank: int = Field(ge=1, le=128)
    lora_alpha: int = Field(ge=1, le=256)
    max_pixels: int = Field(ge=3136, le=1048576)
    seed: int = Field(ge=0)
