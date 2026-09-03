# VLM-and-RAG

Multimodal Document Intelligence + VLM + RAG + Hierarchical Retrieval + GraphRAG + Versioned Knowledge.

## Project Purpose

VLM-and-RAG is a research and production system for multimodal document intelligence, structured extraction, hierarchical retrieval, and versioned knowledge graphs.

### Architecture Principle

> **"Frameworks are replaceable. The domain model belongs to us."**

The core domain model, entities, and pipelines remain decoupled from specific orchestrators, vector databases, or vendor frameworks.

## Current Project Status

- **Phase**: Golden Document Registry (Issue #002)
- **Status**: The engineering foundation and first versioned source-document manifest are implemented. Parsing, RAG, and retrieval features are not yet implemented.

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
