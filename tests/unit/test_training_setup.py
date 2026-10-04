"""CPU-only data guards; these tests never import a model or training framework."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from vlm_rag.training.config import TrainingConfig
from vlm_rag.training.data import TrainingSample, load_samples, validate_splits
from vlm_rag.vlm.chat_prompt import task_instruction

REFERENCE_ROOT = Path("data/semantic_annotations")


def sample(index: int = 1, **updates: object) -> TrainingSample:
    values: dict[str, object] = {
        "sample_id": f"sample-{index}",
        "document_id": f"owned-document-{index}",
        "source_artifact_sha256": str(index) * 64,
        "page_index": 0,
        "image_path": f"images/{index}.png",
        "image_sha256": str(index + 2) * 64,
        "image_byte_size": 20,
        "task_type": "ocr_recovery",
        "label_status": "human_verified",
        "rights": "owned",
        "transcription": "Hà Nội",
    }
    values.update(updates)
    return TrainingSample.model_validate(values)


def test_disjoint_training_and_prompt_contract() -> None:
    train, validation = sample(), sample(2)
    report = validate_splits([train], [validation], REFERENCE_ROOT, for_training=True)
    assert report["held_out_documents"] == 6
    messages = train.messages()
    instruction = messages[1]["content"][0]["text"]
    assert instruction == task_instruction(train.task_type, train.sample_id)
    assert json.loads(messages[-1]["content"])["transcription"] == "Hà Nội"


@pytest.mark.parametrize(
    "field", ["sample_id", "document_id", "source_artifact_sha256", "image_sha256"]
)
def test_split_leakage_is_rejected(field: str) -> None:
    train = sample()
    validation = sample(2, **{field: getattr(train, field)})
    with pytest.raises(ValueError, match=r"overlap|leakage"):
        validate_splits([train], [validation], REFERENCE_ROOT)


@pytest.mark.parametrize("field", ["document_id", "source_artifact_sha256"])
def test_reference_corpus_cannot_be_training_data(field: str) -> None:
    reference = json.loads((REFERENCE_ROOT / "hanoi_master_plan_100y.v2.json").read_bytes())
    contaminated = sample(**{field: reference[field]})
    with pytest.raises(ValueError, match="held-out"):
        validate_splits([contaminated], [sample(2)], REFERENCE_ROOT)


def test_synthetic_supervision_cannot_start_training(tmp_path: Path) -> None:
    synthetic = sample(label_status="synthetic_contract_only")
    assert validate_splits([synthetic], [sample(2)], REFERENCE_ROOT)["synthetic_samples"] == 1
    with pytest.raises(ValueError, match="human-verified"):
        validate_splits([synthetic], [sample(2)], REFERENCE_ROOT, for_training=True)
    with pytest.raises(ValueError, match="guard"):
        validate_splits([sample()], [sample(2)], tmp_path)


@pytest.mark.parametrize(
    "path", ["../outside.png", "C:/private.png", "/root/a.png", "images/../a.png"]
)
def test_training_image_paths_are_contained(path: str) -> None:
    with pytest.raises(ValidationError):
        sample(image_path=path)


def test_training_image_hash_and_task_are_validated(tmp_path: Path) -> None:
    data = b"\x89PNG\r\n\x1a\nsynthetic-contract-data"
    image = tmp_path / "image.png"
    image.write_bytes(data)
    item = sample(
        image_path="image.png",
        image_sha256=hashlib.sha256(data).hexdigest(),
        image_byte_size=len(data),
    )
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(item.model_dump_json() + "\n", encoding="utf-8")
    assert load_samples(manifest, tmp_path) == [item]
    image.write_bytes(data[:-1] + b"X")
    with pytest.raises(ValueError, match="SHA"):
        load_samples(manifest, tmp_path)
    with pytest.raises(ValidationError):
        sample(task_type="table_text_recovery")


def test_training_plan_is_offline_and_config_is_pinned() -> None:
    config = TrainingConfig.model_validate_json(
        Path("training/qwen25vl-3b.example.json").read_bytes()
    )
    assert len(config.revision) == 40
    result = subprocess.run(
        [sys.executable, "training/train.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["mode"] == "plan_only"
    assert not plan["training_started"] and not plan["weights_downloaded"]
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({**config.model_dump(), "revision": "main"})
