# Self-trained document VLM: setup now, GPU execution later

This is a single-GPU LoRA/QLoRA starter, not a trained model. The current GTX 1650 (4 GiB)
host is used for data preparation, tests and RAG. Do not install or run the GPU environment
on it. No base weights have been downloaded and no GPU optimization step has been tested.

## What the model learns

Train the VLM to transcribe document crops, recover table text, or describe visible figures
using the same strict JSON prompt contract as inference. It is not trained to invent facts,
answer legal questions from memory, or build the RAG index. Retrieval uses preserved source
evidence; the answer model is separately configurable.

The starter selects `Qwen/Qwen2.5-VL-3B-Instruct` at immutable revision
`66285546d2b821cf421d4f5eb2576359d3770cd3`. This is a configurable initial choice, not a
benchmark winner. Review its [model card and license](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
before distributing derived weights. The recipe targets the language attention projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`); base vision weights remain frozen.

## Safe commands on the current machine

From repository root:

```bash
uv sync --locked
uv run python training/train.py
uv lock --project training --check
```

The plan imports no Torch/Transformers, starts no training, downloads no weights and reports
whether local data is ready. `training/uv.lock` resolves a separate Python 3.12 environment;
Torch, PEFT and bitsandbytes are absent from the main environment.

## Prepare your dataset

Keep images and manifests under ignored `data/training/`. Each JSONL line follows this schema
(replace placeholder hashes/sizes with the actual bytes):

```json
{
  "sample_id": "owned-doc-a-p0-crop1",
  "document_id": "owned-doc-a",
  "source_artifact_sha256": "<64 lowercase hex characters of source PDF>",
  "page_index": 0,
  "image_path": "doc-a/page-0000-crop1.png",
  "image_sha256": "<64 lowercase hex characters of image>",
  "image_byte_size": 12345,
  "task_type": "ocr_recovery",
  "label_status": "human_verified",
  "rights": "owned",
  "transcription": "Exact visible text, corrected against the image."
}
```

Supported tasks: `ocr_recovery`, `region_transcription`, `table_text_recovery`,
`table_header_recovery`, `figure_or_image_observation`. Table labels use `table_rows`
instead of `transcription`; visual descriptions use `description`. Rights must be
`owned`, `licensed`, or `public_domain` based on your actual permission.

Separate whole documents across train and validation, including all versions and crops.
Never split pages of one document across both. Keep the six existing benchmark documents
held out; neither their annotations nor runtime candidates are training truth. Labels must
be checked by a person. A `human_verified` flag records your assertion; software cannot
prove the visual correctness or legal rights of a label.

```bash
uv run python -m vlm_rag.training \
  --train data/training/train.jsonl \
  --validation data/training/validation.jsonl \
  --images data/training/images
```

Validation checks exact image bytes and safe paths, payload/task consistency, source/hash/ID
overlap, and reference-document contamination. Hash checks cannot identify recompressed
duplicates or renamed documents with incorrectly supplied source metadata; audit lineage too.
Synthetic fixtures may validate for engineering tests but cannot pass actual training preflight.

## Later, on a suitable Linux CUDA host

Copy the repo and your private dataset. Keep the same commit and lockfile. Start with a
two-step smoke config before a longer run. This recipe refuses GPUs below 12 GiB as a
conservative guard; that threshold is not a guarantee that a particular batch fits.
Measure peak VRAM on the actual host and reduce image pixels/sequence size as needed.

```bash
uv sync --project training --locked
uv run --project training python training/train.py --config training/qwen25vl-3b.example.json
# Only this explicit command downloads base weights and starts optimization:
uv run --project training python training/train.py --config training/qwen25vl-3b.example.json --run
```

Set `method` to `lora` or `qlora`. QLoRA loads the base in NF4 with double quantization;
LoRA uses the full base precision and normally needs more memory. The collator masks all
prompt/image tokens and supervises only the assistant completion. It rejects chat-template
prefix mismatches rather than silently training on prompts; it never truncates image tokens.
Training outputs include adapter/processor files, manifest hashes, base revision, commit,
seed, hardware and train/eval metrics. No automatic Hub upload or experiment tracker is used.

## Serving and connecting to RAG

Use a separate GPU serving environment, not the training environment. A compatible vLLM
release can serve the base plus a language-backbone LoRA adapter. Follow its current
[LoRA serving guide](https://docs.vllm.ai/en/latest/features/lora/) and
[model support table](https://docs.vllm.ai/en/latest/models/supported_models/); verify the
adapter loads on your exact serving build before a corpus run. Serving has not been run here.

Example once the GPU server and adapter exist:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
  --revision 66285546d2b821cf421d4f5eb2576359d3770cd3 \
  --host 127.0.0.1 --enable-lora \
  --lora-modules document-vlm=/absolute/path/to/run/adapter
```

In the application's local `.env`, use the served name `document-vlm` for
`VLM_RAG_MODEL_VLM`, your endpoint for `VLM_RAG_MODEL_BASE_URL`, and a separately chosen
answer-capable model for `VLM_RAG_MODEL_ANSWER`. Use an SSH tunnel for a remote private GPU
endpoint; the client permits plain HTTP only on loopback. See the runtime runbook below.

Before promotion, compare the untouched base and your adapter on held-out source crops:
JSON/schema compliance, exact transcription and edit distance, table recovery, provenance,
latency, and downstream retrieval/citation quality. Low training loss does not prove better
document understanding. Do not alter reference labels to improve reported scores.

Implementation references: [PEFT quantized training](https://huggingface.co/docs/peft/v0.17.0/developer_guides/quantization),
[Transformers Trainer](https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/trainer).
