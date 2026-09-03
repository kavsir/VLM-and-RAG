# 2. Separate Document Identity, Version, Source, and Artifact

Date: 2026-09-03

## Status

Accepted

## Context

A source document can retain the same legal identity while a publisher replaces or republishes
its bytes. A landing page describes the document, while a downloadable URL supplies the bytes;
neither a URL nor a filename proves which bytes were retrieved. The first golden document must
therefore be reproducible without introducing parsing, retrieval, or model infrastructure.

## Decision

The committed YAML manifest uses schema version 1 and keeps four concepts explicit:

* `DocumentIdentity` records the stable official number, its normalized form, title, type, and
  issuer.
* `DocumentVersion` records a version identifier, its owning document identifier, and official
  issue/effective dates.
* `SourceReference` records publisher metadata, signer, retrieval time, and distinct HTTPS URLs
  for the descriptive landing page and downloadable asset.
* `FileArtifact` records the published filename, media type, exact byte size, and SHA-256 digest.

Models are strict and immutable. Loading is offline and rejects unknown fields, unsupported schema
versions, invalid dates, malformed digests, mismatched document/version identifiers, and conflated
source URLs. Fetching follows HTTPS redirects, streams to a temporary file, verifies bytes, and
atomically publishes only an exact match. Changed bytes fail rather than silently redefining `v1`.

## Consequences

* Every registered version is independently auditable from committed metadata.
* Large source documents remain local and reproducible instead of entering Git history.
* A replacement artifact requires an explicit new manifest version or a reviewed correction.
* YAML is parsed with the sole added runtime dependency, PyYAML; no document processing or RAG
  framework is introduced.
