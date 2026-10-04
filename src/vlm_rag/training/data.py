"""Verified, document-disjoint supervision with reference-corpus contamination guards."""

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vlm_rag.registry.models import Sha256Digest
from vlm_rag.vlm.chat_prompt import task_instruction
from vlm_rag.vlm.models import VLMTaskType
from vlm_rag.vlm.prompt import build_visual_evidence_prompt


class TrainingSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    sample_id: str = Field(min_length=1, max_length=200)
    document_id: str = Field(min_length=1, max_length=200)
    source_artifact_sha256: Sha256Digest
    page_index: int = Field(ge=0)
    image_path: str
    image_sha256: Sha256Digest
    image_byte_size: int = Field(gt=0, le=10_000_000)
    task_type: VLMTaskType
    label_status: Literal["human_verified", "synthetic_contract_only"]
    rights: Literal["owned", "licensed", "public_domain"]
    transcription: str | None = Field(default=None, min_length=1, max_length=6000)
    table_rows: list[list[str]] | None = None
    description: str | None = Field(default=None, min_length=1, max_length=6000)

    @model_validator(mode="after")
    def validate_task(self) -> Self:
        posix, windows = PurePosixPath(self.image_path), PureWindowsPath(self.image_path)
        if (
            not self.image_path
            or "\\" in self.image_path
            or posix.is_absolute()
            or windows.drive
            or ".." in posix.parts
            or posix.as_posix() != self.image_path
        ):
            raise ValueError("image_path must be a safe relative POSIX path")
        transcript = self.task_type in {VLMTaskType.REGION_TRANSCRIPTION, VLMTaskType.OCR_RECOVERY}
        table = self.task_type in {
            VLMTaskType.TABLE_TEXT_RECOVERY,
            VLMTaskType.TABLE_HEADER_RECOVERY,
        }
        if (self.transcription is not None) != transcript or (self.table_rows is not None) != table:
            raise ValueError("supervision payload must match task type")
        if (self.description is not None) != (not transcript and not table):
            raise ValueError("description is required only for visual observations")
        if self.table_rows is not None and (
            not self.table_rows
            or any(not row for row in self.table_rows)
            or len(json.dumps(self.table_rows, ensure_ascii=False)) > 6000
        ):
            raise ValueError("table supervision must contain bounded, non-empty rows")
        return self

    def target(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": 1,
            "request_id": self.sample_id,
            "task_type": self.task_type.value,
        }
        for field in ("transcription", "table_rows", "description"):
            value = getattr(self, field)
            if value is not None:
                payload[field] = value
        return payload

    def messages(self) -> list[dict[str, object]]:
        return [
            {"role": "system", "content": build_visual_evidence_prompt(self.task_type)},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": task_instruction(self.task_type, self.sample_id),
                    },
                    {"type": "image"},
                ],
            },
            {"role": "assistant", "content": json.dumps(self.target(), ensure_ascii=False)},
        ]


def load_samples(path: Path, image_root: Path) -> list[TrainingSample]:
    root = image_root.resolve()
    samples = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        sample = TrainingSample.model_validate_json(line)
        image = (root / sample.image_path).resolve()
        if not image.is_relative_to(root):
            raise ValueError(f"image escapes root at line {line_number}")
        if image.stat().st_size != sample.image_byte_size:
            raise ValueError(f"image size mismatch at line {line_number}")
        data = image.read_bytes()
        if hashlib.sha256(data).hexdigest() != sample.image_sha256:
            raise ValueError(f"image SHA mismatch at line {line_number}")
        if not data.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")):
            raise ValueError(f"expected PNG or JPEG at line {line_number}")
        samples.append(sample)
    if not samples or len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("manifest requires non-empty samples with unique IDs")
    return samples


def validate_splits(
    train: list[TrainingSample],
    validation: list[TrainingSample],
    reference_root: Path,
    *,
    for_training: bool = False,
) -> dict[str, object]:
    references = []
    for path in sorted(reference_root.glob("*.v2.json")):
        payload = json.loads(path.read_bytes())
        if "source_artifact_sha256" in payload and "document_id" in payload:
            references.append(payload)
    if not references:
        raise ValueError("reference guard is missing; cannot establish held-out corpus")
    reference_hashes = {item["source_artifact_sha256"] for item in references}
    reference_ids = {item["document_id"] for item in references}
    all_samples = train + validation
    if not train or not validation:
        raise ValueError("both train and validation must be non-empty")
    if len({sample.sample_id for sample in all_samples}) != len(all_samples):
        raise ValueError("sample IDs overlap between splits")
    for field in ("document_id", "source_artifact_sha256", "image_sha256"):
        if {getattr(sample, field) for sample in train} & {
            getattr(sample, field) for sample in validation
        }:
            raise ValueError(f"train/validation leakage by {field}")
    for sample in all_samples:
        if sample.source_artifact_sha256 in reference_hashes or sample.document_id in reference_ids:
            raise ValueError("held-out reference document cannot enter train or validation")
        if for_training and sample.label_status != "human_verified":
            raise ValueError(
                "actual training requires human-verified labels; synthetic is test-only"
            )
    return {
        "train_samples": len(train),
        "validation_samples": len(validation),
        "held_out_documents": len(references),
        "document_disjoint": True,
        "synthetic_samples": sum(s.label_status == "synthetic_contract_only" for s in all_samples),
        "note": "Hashes detect exact overlap, not recompressions; audit source identity.",
    }
