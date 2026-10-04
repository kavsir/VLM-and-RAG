# VLM + RAG runtime quickstart

The runtime works from validated Physical/Semantic IR. Existing MinerU/Marker adapters remain
the PDF parsing stage; this change does not replace them with a low-fidelity page-text parser.
It supports a self-hosted fine-tuned VLM through an OpenAI-compatible Chat Completions endpoint.
Follow [training setup](../../training/README.md) before attempting a real adapter experiment.

## Try retrieval now, no model or GPU

From the repository root, choose exactly one parser representation per document/version:

```bash
uv sync --locked
uv run python -m vlm_rag index \
  data/semantic_ir/hanoi_master_plan_100y/marker.v1.json \
  --output data/rag-demo/hanoi.sqlite
uv run python -m vlm_rag ask "quy hoạch Thủ đô Hà Nội" \
  --index data/rag-demo/hanoi.sqlite --output data/rag-demo/answer.json
```

Outputs are JSON. The default `extractive` mode returns matching source excerpts; it is not
a model-generated synthesis. Each hit has document/version, source and Semantic SHA, page
indexes (zero-based), Structural path, original anchors, statement-relative slice offsets,
and optional visual observation lineage. A `visual` anchor includes a normalized image box.
No-hit retrieval returns `insufficient_evidence` without any model request.

The index is immutable: use a new filename to rebuild. Existing files are never overwritten.
Supplying two parser representations for one document/version fails instead of double-counting.
Use `--document-id` and `--version-id` to scope a query. Scores are BM25 ranks, not confidence
probabilities. Unicode tokenization removes accents for lookup; quoted evidence preserves bytes.

## Prepare Semantic IR from a parser result

Use a wire-version-2 Physical IR file produced by the existing v1 normalizers:

```bash
uv run python -m vlm_rag prepare --physical /path/to/physical-v1.json \
  --output data/runtime/my-document-baseline
```

An existing Structural IR can be supplied with `--structural`; otherwise the deterministic
extractor creates it in memory. Source identity is validated before Semantic construction.
The new `semantic.json` can be passed to `index`.

## Later: use your trained VLM

Configure `.env` from `.env.example`. Do not commit keys. The endpoint must support images,
JSON-object output and the configured output-token parameter; no automatic provider-specific
fallback or retry is attempted. `VLM_RAG_MODEL_MAX_REQUESTS=0` disables all model calls.
Set a small explicit request cap for a smoke run. It applies to one process, not monthly billing.

```bash
uv sync --locked --extra pdf
uv run --extra pdf python -m vlm_rag prepare \
  --physical /path/to/physical-v1.json --pdf /path/to/verified-source.pdf \
  --output data/runtime/my-document-vlm --live --max-requests 2
uv run python -m vlm_rag index data/runtime/my-document-vlm/semantic.json \
  --output data/rag-demo/enriched.sqlite
uv run python -m vlm_rag ask "Câu hỏi của bạn" \
  --index data/rag-demo/enriched.sqlite --live --output data/rag-demo/model-answer.json
```

Alternatively use `--asset-root` for retained PNG/JPEG images whose hashes are already in
Physical IR. PDF bytes must match the source SHA before rendering. PDF crops use normalized
top-left coordinates on the displayed bitmap, not assumed parser page units. The renderer's
nominal DPI scale does not account for unusual PDF UserUnit factors.

Each VLM run retains selection decisions, exact submitted image bytes, request records,
completion envelope (reported model, usage, latency), raw content/hash and normalized observations.
Requested model alias and provider-reported snapshot are retained separately. A failed run
keeps its available evidence and `failure.json`; it does not produce a completed Semantic result.
Do not rerun blindly after failure: requests may already have been billed.

The answer validator rejects unknown chunk IDs and non-verbatim citation quotes. It cannot
prove that a paraphrased claim logically follows from a valid quote. Human review and held-out
answer-quality evaluation remain required. Figure descriptions and recovered tables remain
visual observations; only supported transcription statements enter this initial text index.

## Validation and current limits

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run --extra pdf pytest
uv build
```

Tests cover mock HTTP transport, PDF crop geometry, VLM normalization/lineage, indexing,
retrieval and grounded answers without a provider or GPU. Actual training, provider-specific
model behavior, serving compatibility and quality improvements are not yet validated.
This is a single-user CLI and lexical retrieval baseline, not a deployed multi-tenant service,
vector-search benchmark, hierarchy-expansion retriever or complete production RAG platform.

Protocol reference: [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
Rendering reference: [PDFium Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
