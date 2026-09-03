# 1. Bootstrap Repository and Tooling Foundation

Date: 2026-09-03

## Status

Accepted

## Context

The VLM-and-RAG project is beginning development. To ensure long-term maintainability, research reproducibility, and code quality, the repository requires a strict, minimal, and standardized engineering foundation before implementing domain logic, retrieval, or model pipelines.

Key constraints:
* Framework independence ("Frameworks are replaceable. The domain model belongs to us.")
* Fast and reproducible dependency management
* Automated static quality checks (linting, formatting, type checking, unit tests)
* Clean src-layout to prevent implicit import leaks

## Decision

1. **Python Version**: Standardize on Python >= 3.12 for modern typing, performance, and standard library improvements.
2. **Package & Environment Manager**: Adopt `uv` as the unified project and package manager.
3. **Build Backend**: Use `hatchling` pinned to exact version (`hatchling==1.32.0`) adhering to PEP 517 / PEP 621.
4. **Code Quality**:
   * `ruff` for both high-speed linting and formatting.
   * `mypy` for static type checking in strict mode (`strict = true`).
   * `pre-commit` configured with local hooks executing the project's uv-managed tools to prevent tool-version drift.
5. **Testing**: `pytest` for test discovery and execution, with deterministic test isolation from ambient environment variables.
6. **Configuration**: Use `pydantic-settings` for typed, environment-variable-driven configuration with explicit defaults and zero secrets.
7. **CI**: Minimal hardened GitHub Actions workflow executing checkout (with pinned SHA and unpersisted credentials), pinned uv installation, locked dependency sync, ruff lint/format check, strict mypy type check, and pytest matrix across supported Python versions on push and PR to main.

## Consequences

* Ensures uniform local development across Linux, macOS, and Windows.
* Enforces strict quality gates before any domain or model code is introduced.
* Prevents premature coupling to specific RAG or vector database frameworks.
