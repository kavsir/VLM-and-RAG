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
context, evidence anchors, controlled provenance, owned mention IDs, and—when VLM-derived—the
exact visual-observation ID. Mentions cover only legal
references, document identifiers, explicit dates, quantities, and high-confidence authorities. Raw
text is immutable; deterministic normalized values are separate. Citations and entities remain
unresolved.

Text, object, and visual anchors are controlled evidence types. VLM-derived transcriptions pass
through the same deterministic mention extractor as native text. Table recovery and figure
descriptions remain visual observations; they do not patch Physical or Structural IR and figure
descriptions are not promoted to facts.

Selection precedes invocation and uses Physical IR evidence plus exact Structural IR ownership.
Every selected request is bound to one Structural node/path and the canonical Physical/Structural
hashes. Missing or ambiguous ownership fails closed instead of selecting a last-seen owner. Stable
priority, page/document budgets, and non-selection reasons make the policy reproducible. PDF crops cross an injected
renderer boundary only after source-PDF SHA verification. Retained assets require a safe relative
path, hash, byte size, and media type. No renderer is a core dependency.

Semantic construction independently rebuilds the direct-content owner index from Structural IR;
it does not trust selector history or a caller-declared owner. Every observation block must have
exactly one owner, and every block in a multi-block observation must resolve to the same declared
node/path. Whole-block evidence spanning multiple Structural nodes fails closed.

The provider-neutral VLM contract keeps immutable request metadata and raw responses separate from
strict normalized observations. Each observation records canonical request-record, raw-response,
image, Physical IR, and Structural IR hashes plus model, provider protocol, prompt version, and
Structural node/path. Semantic statements and mentions link back to that observation. Invalid data
fails closed. Prompts mark document content as untrusted and prohibit following instructions in
it. Replay is the offline default; any non-replay
client requires explicit `live_vlm=True`. The committed replay envelope is synthetic contract
evidence and cannot support a quality claim.

## Consequences and limitations

The fixed 46-page reference v2/schema 2 was rebuilt by visually auditing rendered source-PDF pages.
It records occurrence-level statement and mention IDs, exact excerpts/spans, render hashes, strict
typed details, and region/task visual-assistance labels. Runtime candidates are stored separately
and are never loaded as authoritative truth. The reference is AI-authored with disclosed prior
Physical, Structural, and Semantic extractor exposure; it is not human, blind, or independent
ground truth. Metrics pair occurrences one-to-one by page, kind, and raw evidence. Duplicate
occurrences use exact statement text and relative span as non-target evidence; unresolved ambiguity
is excluded from attribute submetrics unless every optimal pairing gives the same scored result.
Selector evaluation uses deterministic maximum-cardinality, then maximum-overlap, region/task
matching; it is not a coarse page-level proxy and arbitrary IDs do not affect metric values.

No real VLM experiment was available, so quality, latency, cost, transcription accuracy, and
improvement remain unestablished. Cross-document resolution, entity resolution, RAG, KG, and Issue
#010 are deliberately absent.
