# VLM-and-RAG

Multimodal Document Intelligence + VLM + RAG + Hierarchical Retrieval + GraphRAG + Versioned Knowledge.

## Project Purpose

VLM-and-RAG is a research and production system for multimodal document intelligence, structured extraction, hierarchical retrieval, and versioned knowledge graphs.

### Architecture Principle

> **"Frameworks are replaceable. The domain model belongs to us."**

The core domain model, entities, and pipelines remain decoupled from specific orchestrators, vector databases, or vendor frameworks.

## Current Project Status

- **Phase**: Marker Adapter + Physical IR v0 Normalization (Issue #005)
- **Status**: The engineering foundation, golden-document registry, independent external MinerU
  and Marker adapters, and parser-independent Physical Document IR v0 normalization are
  implemented. Semantic extraction, RAG, and retrieval features are not yet implemented.

## Prerequisites

* **Python**: >= 3.12 (tested on Python 3.12 and 3.14)
* **Package Manager**: [uv](https://docs.astral.sh/uv/) (pinned to version `0.11.14` in CI)
* **Git**

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/kavsir/VLM-and-RAG.git
cd VLM-and-RAG
```

### 2. Environment Configuration

Copy the example environment configuration:

```bash
cp .env.example .env
```

### 3. Install Dependencies

Depending on whether you are developing locally or running in a reproducible/CI environment:

* **Normal Development (Resolution & Sync)**:
  Resolves and updates dependencies, updating `uv.lock` if dependencies changed:
  ```bash
  uv sync
  ```

* **Reproducible / CI Installation (Locked)**:
  Enforces exact versions from `uv.lock` and fails if `uv.lock` is stale or requires changes:
  ```bash
  uv sync --locked
  ```

This creates a managed `.venv` with the project installed in editable mode along with development tooling (`ruff`, `mypy`, `pytest`, `pre-commit`).

### 4. Set up Pre-commit Hooks

Pre-commit runs fast local checks using the project's uv-managed tools to prevent tool-version drift:

```bash
uv run pre-commit install
```

## Running Quality Checks

### Linting

Run Ruff lint analysis:

```bash
uv run ruff check .
```

To automatically apply safe fixes:

```bash
uv run ruff check --fix .
```

### Format Check

Verify code formatting conforms to project standards:

```bash
uv run ruff format --check .
```

To reformat code in-place:

```bash
uv run ruff format .
```

### Type Checking

Run Mypy static type analysis in strict mode:

```bash
uv run mypy src
```

### Testing

Run the test suite using Pytest:

```bash
uv run pytest
```

### Building Packages

Build source distribution (sdist) and binary wheel:

```bash
uv build
```

## Golden Document Registry

The committed manifest records official metadata and the exact expected PDF bytes separately. Its
validation is deterministic and offline:

```bash
uv run python -m vlm_rag.registry validate
```

Fetch the official PDF, follow redirects, and publish it locally only after its byte size and
SHA-256 match the manifest:

```bash
uv run python -m vlm_rag.registry fetch
```

The verified artifact is written to
`data/golden/hanoi_master_plan_100y/v1/source.pdf`. Downloaded source artifacts are ignored by Git;
the versioned YAML manifest under `data/manifests/` is committed. A checksum mismatch fails without
replacing an existing local artifact.

## Optional MinerU Parser Environment

MinerU is an optional external parser runtime. It is deliberately absent from the main project
dependencies because MinerU 3.4.5 requires Python `<3.14`, while this project supports and tests
Python 3.14. The adapter invokes only the public `mineru` executable and does not import MinerU
internals.

Keep the two environments separate. The normal `.venv` remains the project environment. On
Windows, create a dedicated Python 3.12 parser environment and install only the pinned pipeline
extra:

```powershell
uv venv .venv-mineru --python 3.12
uv pip install --python .venv-mineru\Scripts\python.exe "mineru[pipeline]==3.4.5" "six==1.17.0"
```

`six==1.17.0` is an explicit workaround for MinerU 3.4.5 pipeline code that imports `six` without
declaring it in the published pipeline dependencies.

On Windows hosts where the current account cannot create Hugging Face cache symlinks, select
MinerU's documented ModelScope source before the first model download:

```powershell
$env:MINERU_MODEL_SOURCE = "modelscope"
```

On POSIX systems, the equivalent executable is `.venv-mineru/bin/mineru`. Probe the actual external
runtime from the main project environment:

```powershell
uv run python -m vlm_rag.parsers --executable .venv-mineru\Scripts\mineru.exe probe
```

After obtaining the verified golden PDF with the registry fetch command, run the Hanoi pipeline
baseline explicitly:

```powershell
uv run python -m vlm_rag.parsers --executable .venv-mineru\Scripts\mineru.exe --timeout 14400 run
```

The adapter verifies `source.pdf` against the Issue #002 manifest before starting MinerU. Successful
run metadata is written to
`data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/run.json`; untouched MinerU
outputs are retained below its `raw/` directory. The entire generated `parser_runs/` tree is ignored
by Git. The measured structure, artifact hashes, and runtime from the first real Hanoi parse are
recorded in [`docs/research/mineru-hanoi-baseline.md`](docs/research/mineru-hanoi-baseline.md).

## Optional Marker Parser Environment

Marker 2.0.0 is the second independent external parser. Its Torch, Surya, and model dependencies
remain outside the main project and CI. Create the dedicated Python 3.12 environment and install the
exact reviewed distribution:

```powershell
uv venv .venv-marker --python 3.12
uv pip install --python .venv-marker\Scripts\python.exe "marker-pdf==2.0.0"
```

Verify the installed package through distribution metadata, without importing Marker internals:

```powershell
.venv-marker\Scripts\python.exe -c "import importlib.metadata; print(importlib.metadata.version('marker-pdf'))"
```

The installed `marker_single --help` must expose the flags used by the adapter. Probe the boundary
and then run the checksum-verified born-digital Hanoi baseline in `fast-no-ocr` mode:

```powershell
uv run python -m vlm_rag.parsers.marker_cli `
  --executable .venv-marker\Scripts\marker_single.exe `
  --python-executable .venv-marker\Scripts\python.exe probe

uv run python -m vlm_rag.parsers.marker_cli `
  --executable .venv-marker\Scripts\marker_single.exe `
  --python-executable .venv-marker\Scripts\python.exe `
  --timeout 14400 run
```

The adapter explicitly records `mode=fast`, `disable_ocr=true`, `output_format=json`, and the
`fast-no-ocr` experiment label. It requires the Marker executable and Python interpreter to resolve
from the same external environment, enforces a CPU-only subprocess environment, and records the
observed device and device policy. It does not use an LLM service or network from main application
code. Raw outputs and execution evidence are written below
`data/golden/hanoi_master_plan_100y/v1/parser_runs/marker/2.0.0/fast-no-ocr/` and ignored by Git.

Normalize the retained Marker document tree into the unchanged Physical IR v0:

```powershell
uv run python -m vlm_rag.physical_ir normalize-marker
```

See [ADR 0005](docs/adr/0005-marker-parser-boundary.md) and the
[Marker Hanoi baseline](docs/research/marker-hanoi-baseline.md) for the observed JSON boundary,
mapping rules, artifact hashes, metrics, and MinerU comparison.

## Physical Document IR v0

Physical Document IR provides an explicit, parser-independent representation of layout blocks, bounding
boxes, and reading order, decoupled from MinerU or any other parser runtime.

Normalize raw parser outputs into a validated `PhysicalDocument`:

```bash
uv run python -m vlm_rag.physical_ir normalize \
  --raw-dir data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/raw \
  --manifest data/manifests/hanoi_master_plan_100y.v1.yaml \
  --output data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/physical_ir_v0.json
```

Validate an existing serialized PhysicalDocument JSON file:

```bash
uv run python -m vlm_rag.physical_ir validate \
  data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/physical_ir_v0.json
```

See [ADR 0004](docs/adr/0004-physical-document-ir-v0.md) and the [Hanoi Normalization Report](docs/research/physical-ir-hanoi-normalization.md)
for schema design and baseline metrics.

## Branch & PR Workflow

1. Create a feature branch off `main` following Conventional Branch Naming:
   ```bash
   git checkout -b feat/<feature-name>
   # or fix/<issue-name>, chore/<task-name>
   ```
2. Commit your changes using [Conventional Commits](https://www.conventionalcommits.org/):
   * `feat(...)`: New features
   * `fix(...)`: Bug fixes
   * `docs(...)`: Documentation changes
   * `test(...)`: Adding or updating tests
   * `ci(...)`: CI/CD pipeline changes
   * `chore(...)`: Routine tooling or dependency updates
3. Verify all local checks pass before submitting:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest
   ```
4. Open a Pull Request targeting `main`. Pull requests to `main` are expected to pass CI before merge.
