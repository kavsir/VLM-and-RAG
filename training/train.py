"""Plan on CPU; run single-GPU LoRA/QLoRA only with an explicit --run flag."""

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

from vlm_rag.training.config import TrainingConfig
from vlm_rag.training.data import load_samples, validate_splits

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/qwen25vl-3b.example.json")
    parser.add_argument(
        "--run", action="store_true", help="download base weights and train on CUDA"
    )
    args = parser.parse_args()
    config = TrainingConfig.model_validate_json(args.config.read_bytes())
    train_path = ROOT / config.train_manifest
    validation_path = ROOT / config.validation_manifest
    images = ROOT / config.image_root
    output = ROOT / config.output
    present = train_path.is_file() and validation_path.is_file()
    report = None
    if present:
        train = load_samples(train_path, images)
        validation = load_samples(validation_path, images)
        report = validate_splits(
            train,
            validation,
            ROOT / "data/semantic_annotations",
            for_training=args.run,
        )
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "weights_downloaded": False,
                    "training_started": False,
                    "config": config.model_dump(mode="json"),
                    "data_ready": present,
                    "validation": report,
                    "next": "Provide document-disjoint, human-verified data; use a CUDA GPU host.",
                },
                indent=2,
            )
        )
        return
    if not present:
        raise ValueError("train/validation manifests missing; no weights have been downloaded")
    if output.exists():
        raise ValueError(
            "choose a new output directory; existing checkpoints are never overwritten"
        )

    # These imports intentionally happen AFTER CPU-only preflight and explicit --run.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from PIL import Image
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise ValueError("training setup requires a CUDA GPU; CPU planning remains available")
    if torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3:
        raise ValueError("this starter recipe refuses GPUs below 12 GiB; use a larger GPU host")
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this starter supports one GPU; do not use distributed launch")
    set_seed(config.seed)
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quantization = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        if config.method == "qlora"
        else None
    )
    processor = AutoProcessor.from_pretrained(
        config.base_model,
        revision=config.revision,
        trust_remote_code=False,
        min_pixels=3136,
        max_pixels=config.max_pixels,
    )
    processor.tokenizer.padding_side = "right"
    model = AutoModelForImageTextToText.from_pretrained(
        config.base_model,
        revision=config.revision,
        trust_remote_code=False,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.config.use_cache = False
    if quantization is not None:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    class Samples(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            return self.samples[index]

    def collate(samples):
        # Batch size is fixed to one. Never truncate away image tokens or target text.
        if len(samples) != 1:
            raise ValueError("collator requires batch size one")
        sample = samples[0]
        data = (images / sample.image_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != sample.image_sha256:
            raise ValueError("training image changed after validation")
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > 25_000_000:
                raise ValueError("training image exceeds pixel budget")
            rgb = image.convert("RGB")
        try:
            messages = sample.messages()
            full_text = processor.apply_chat_template(messages, tokenize=False)
            prompt_text = processor.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )
            batch = processor(text=[full_text], images=[rgb], return_tensors="pt")
            prefix = processor(text=[prompt_text], images=[rgb], return_tensors="pt")
        finally:
            rgb.close()
        prefix_ids = prefix["input_ids"][0]
        if not torch.equal(batch["input_ids"][0, : len(prefix_ids)], prefix_ids):
            raise ValueError("chat template does not preserve the prompt prefix; cannot mask loss")
        labels = batch["input_ids"].clone()
        labels[:, : len(prefix_ids)] = -100
        labels[batch["attention_mask"] == 0] = -100
        if not (labels != -100).any():
            raise ValueError("sample has no assistant tokens to supervise")
        batch["labels"] = labels
        return batch

    # Fail on template/masking incompatibility before entering a costly optimizer loop.
    collate([train[0]])
    output.mkdir(parents=True, exist_ok=False)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    (output / "run-inputs.json").write_bytes(
        (
            json.dumps(
                {
                    "config": config.model_dump(mode="json"),
                    "git_head": head,
                    "validation": report,
                    "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                    "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                    "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__,
                    "loss": "assistant-completion-only; image and prompt tokens masked",
                },
                indent=2,
            )
            + "\n"
        ).encode()
    )
    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=TrainingArguments(
            output_dir=str(output),
            max_steps=config.max_steps,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            bf16=bf16,
            fp16=not bf16,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            remove_unused_columns=False,
            report_to="none",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,
            logging_steps=5,
            seed=config.seed,
            data_seed=config.seed,
            dataloader_num_workers=0,
        ),
        train_dataset=Samples(train),
        eval_dataset=Samples(validation),
        data_collator=collate,
    )
    training_result = trainer.train()
    trainer.save_metrics("train", training_result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    trainer.save_model(str(output / "adapter"))
    processor.save_pretrained(output / "adapter")


if __name__ == "__main__":
    main()
