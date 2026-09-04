# 3. Keep Document Parsers Behind an External Runtime Boundary

Date: 2026-09-04

## Status

Accepted

## Context

MinerU 3.4.5 is the first parser evaluated against the registered Hanoi golden document. It requires
Python `>=3.10,<3.14`, while the main project supports Python 3.12 and 3.14. Parser installations can
also carry large model, inference, and hardware-specific dependencies that do not belong in the
domain package or its CPU-light CI environment.

The first parse exists to capture and inspect MinerU's real raw output. It must not predetermine the
Physical Document IR planned for Issue #004.

## Decision

MinerU runs as an explicitly invoked external executable in a separate Python 3.12 environment. The
main package does not import MinerU modules or add MinerU to `pyproject.toml`. A small adapter:

* accepts only the `pipeline` backend and the reviewed MinerU version 3.4.5;
* verifies the input bytes against the golden-document manifest before any subprocess call;
* invokes an explicit argument list with `shell=False`, a timeout, and captured output;
* records the executable's actual `mineru --version` response;
* preserves raw outputs and records their relative paths, sizes, and SHA-256 digests; and
* rejects missing executables, timeouts, failed processes, mixed run directories, and missing core
  inspection artifacts.

No generic parser plugin framework, parser factory, normalized block model, or Physical Document IR
is introduced.

## Consequences

* Python and heavy inference dependencies remain isolated from the main project and CI.
* MinerU can be upgraded, replaced, or moved to a worker process without changing domain records.
* GPU/runtime failures remain outside the application import path.
* Raw evidence can inform Issue #004 without coupling it to MinerU's private Python APIs.
* Developers must provision and manage the optional parser environment separately.

Docker and parser microservices remain possible future deployment choices, not current
requirements.
