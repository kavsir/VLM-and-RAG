# ADR 0008: Domain Semantic IR v1 and selective VLM evidence

- Status: Accepted for Issue #009
- Date: 2026-09-07

## Context

Structural IR owns hierarchy, canonical paths, and exact Physical IR ownership. Semantic extraction
needs statements and controlled mentions without making parser fields, a VLM provider, or future
retrieval infrastructure part of the domain contract.

## Decision

`SemanticDocument` v1 consumes only `PhysicalDocumentV1`, `StructuralDocument` v1, and validated
`VLMObservationRecord` values. It binds document/version/source identity plus SHA-256 hashes of the
canonical Physical and Structural serializations. It never consumes raw parser artifacts or
evaluation labels.

Statements are first-class exact direct-content spans, not paraphrases. They retain Structural
context, evidence anchors, controlled provenance, and owned mention IDs. Mentions cover only legal
references, document identifiers, explicit dates, quantities, and high-confidence authorities. Raw
text is immutable; deterministic normalized values are separate. Citations and entities remain
unresolved.

Text, object, and visual anchors are controlled evidence types. VLM-derived transcriptions pass
through the same deterministic mention extractor as native text. Table recovery and figure
descriptions remain visual observations; they do not patch Physical or Structural IR and figure
descriptions are not promoted to facts.

Selection precedes invocation and uses only Physical IR evidence. Stable priority, page/document
budgets, and non-selection reasons make the policy reproducible. PDF crops cross an injected
renderer boundary only after source-PDF SHA verification. Retained assets require a safe relative
path, hash, byte size, and media type. No renderer is a core dependency.

The provider-neutral VLM contract keeps immutable request metadata and raw responses separate from
strict normalized observations. Invalid data fails closed. Prompts mark document content as
untrusted and prohibit following instructions in it. Replay is the offline default; any non-replay
client requires explicit `live_vlm=True`. The committed replay envelope is synthetic contract
evidence and cannot support a quality claim.

## Consequences and limitations

The fixed 46-page reference v1 is an AI-assisted candidate audit with disclosed prior Physical,
Structural, and Semantic extractor exposure. It is not human, blind, or independent ground truth.
No real VLM experiment was available, so quality, latency, cost, transcription accuracy, and
improvement remain unestablished. Cross-document resolution, entity resolution, RAG, KG, and Issue
#010 are deliberately absent.
