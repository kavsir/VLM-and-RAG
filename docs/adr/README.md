# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records (ADRs) tracking significant architectural and technical decisions made throughout the project lifecycle.

## Format

Each record should be named sequentially: `NNNN-short-title.md` (e.g. `0001-bootstrap-python-tooling.md`) and follow this structure:

* **Status**: Proposed | Accepted | Deprecated | Superseded
* **Context**: What problem or requirement motivated this decision?
* **Decision**: What was decided and why?
* **Consequences**: What are the positive and negative outcomes or trade-offs?

## Records

* [0001 — Bootstrap Repository and Tooling Foundation](0001-bootstrap-python-tooling.md)
* [0002 — Separate Document Identity, Version, Source, and Artifact](0002-document-identity-and-versioning.md)
* [0003 — Keep Document Parsers Behind an External Runtime Boundary](0003-external-document-parser-boundary.md)
* [0004 — Physical Document Intermediate Representation (v0)](0004-physical-document-ir-v0.md)
* [0005 — Add Marker Through an External Runtime Boundary](0005-marker-parser-boundary.md)
