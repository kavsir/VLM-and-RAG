# ADR 0007 — Vietnamese Legal/Planning Structural IR v1

- **Status:** Accepted
- **Date:** 2026-09-06
- **Issue:** #008

## Context

Physical IR v1 records what is physically present: ordered blocks, geometry, disposition,
text-extraction provenance, tables, and visual assets. Hierarchical retrieval needs a separate answer
to how that evidence is organized. Putting articles or planning outline levels into Physical IR would
mix parser observation with a domain-specific interpretation and make the physical contract unstable.

The retained corpus contains Vietnamese laws, administrative decisions, planning documents,
appendices, born-digital pages, and scanned pages. It also contains embedded legal citations and
numbered prose that must not automatically become structure.

## Decision

### Separate, domain-specific representation

Structural IR v1 is a new parser-independent wire model with `structural_ir_version=1` and profile
`vi_legal_planning_v1`. The deterministic extractor consumes `PhysicalDocumentV1` only and requires
its wire `physical_ir_version=2`. It never reads Marker or MinerU raw records, benchmark annotations,
official HTML, or model output.

Each result repeats document/version/source identity and binds to the exact deterministic serialized
Physical IR bytes through `source_physical_ir_sha256`. This is not an automatic v0-to-v1 upgrade.

### Node taxonomy and hierarchy

Every document begins with one `DOCUMENT` root. Controlled non-root kinds are `PART`, `CHAPTER`,
`SECTION`, `SUBSECTION`, `ARTICLE`, `CLAUSE`, `POINT`, `APPENDIX`, and `GENERIC_SECTION`.
Unclassified front matter remains direct root content; no speculative preamble node is invented.

Legal levels may be omitted. Articles can be children of the document, part, chapter, section,
subsection, or appendix. Clauses require an article and points require a clause. Appendices are
document-level. Corpus evidence also requires appendix-contained parts and sections. Generic outline
nodes may nest below an appendix, another generic node, or an active article/clause when a planning
outline is physically embedded in a decision article.

State transitions are deterministic. A new legal sibling closes its descendants; an appendix closes
the legal and generic stacks. A generic child does not erase the active article/clause, so a `1.1`
outline cannot cause the next `2.` clause to lose its legal context.

### Deterministic identity and path

Node IDs are zero-padded reading-order sequences. They are deterministic for identical normalized
input and order, not persistent identity across parser representations or document versions.
Ordinals preserve `ordinal_raw` and a kind-aware normalized `ordinal_key`. Article and clause Arabic
ordinals preserve lowercase amendment suffixes, POINT ordinals preserve Vietnamese letters (including
`c`, `d`, `i`, `l`, `m`, and `đ`), and only Roman-numbered structural kinds use strict Roman
conversion. Non-canonical Roman spellings such as `IIII`, `IC`, `VX`, `IIV`, and `MMMM` are rejected.

Canonical paths use controlled kind/ordinal segments and never raw parser IDs or occurrence suffixes.
A duplicate parent/kind/ordinal key is rejected before node creation, retained as BODY, and recorded as
`duplicate_structural_key_rejected`; the extractor does not invent `~2` identities. A path represents
the recovered hierarchical key for that exact physical input; cross-parser equality is measured rather
than assumed.

### Physical anchors and complete partition

Every anchor identifies a Physical IR `block_id` and `page_index`. Text roles (`MARKER`, `TITLE`,
`BODY`) also carry half-open Python string offsets into the exact `PhysicalBlockV1.text`. Detection
may use NFKC and controlled whitespace normalization on a copy, but stored offsets and source text
are unchanged.

All characters of every non-empty `CONTENT` block are partitioned exactly once across structural
anchors: no gap and no overlap. One physical block may therefore contribute to several nodes. An
empty `CONTENT` table, figure, or image has exactly one `OBJECT` anchor owned by the deepest active
node (or the root). `DISCARDED` and unknown-disposition blocks stay unanchored.

### Formal and generic detection

Line-start, case-insensitive Vietnamese markers (plus controlled unaccented variants) have highest
precedence: `PHẦN`, `CHƯƠNG`, `MỤC`, `TIỂU MỤC`, `ĐIỀU`, and `PHỤ LỤC`. A leading Arabic item is a
clause only while an article is active; a leading letter item is a point only while a clause is
active. Article/heading detection does not require the physical block to be a title.

Outside those contexts, heading-like Roman or dotted decimal outlines may become
`GENERIC_SECTION`. A simple Arabic generic heading requires stronger physical/typographic evidence.
A multi-component decimal in ordinary text requires either strong heading evidence or an exact active
numeric generic parent; short text alone is insufficient.
Letter items outside clauses remain body content. Immediate title continuation is accepted only for
a physical title or a short uppercase heading-like line.

The rules are anchored at line start. Embedded phrases such as “theo Điều 3”, “tại điểm a khoản 2”,
or “thực hiện Chương II” do not create nodes. Line-start prose references using continuations such as
“của”, “nêu trên”, and “kèm theo” are also rejected. Likely TOC pages are detected conservatively from
an exact `MỤC LỤC` heading plus dotted/page-number leader entries, or from multiple leader entries;
their text remains BODY and cannot reserve a structural key. No fuzzy edit distance or synthetic
confidence is used.

APPENDIX is terminal in profile v1. The retained corpus does not require returning from an appendix to
the outer legal hierarchy; a future profile must add evidence for such a transition rather than infer it.

### Evaluation and reproducibility

Reference annotation v2/schema 2 is an AI visual PDF structural re-audit, not human ground truth. It
discloses prior Physical IR and Structural extractor exposure, states that neither output was used as
reference truth, and contains no parser/block identity. The 46-page re-audit log, render configuration,
render SHA-256 values, corrections, and ambiguities are committed in
`data/structural_annotations/reference_structural_audit.v2.json`.

Matching requires exact audited page, kind, and ordinal. Singleton groups match directly; duplicate
groups match only through a uniquely resolvable exact parent canonical path. Ambiguous groups remain
unmatched, so occurrence shifting cannot create a false true-positive. Full parent-edge metrics include
document-root edges, all unmatched predicted/reference edges, and count a wrong parent as one FP plus
one FN. Zero-denominator metrics and empty-union Jaccard are `null`/N/A. Marker, MinerU, and the combined
parser-representation-weighted aggregate are reported separately.

All ten retained Physical IR inputs are extracted twice. The reasonably small outputs are committed
so a clean clone can load the schema, validate annotations, re-evaluate metrics offline, and render
the report without parser inference or source PDFs.

## Consequences

- Physical observations remain stable and replaceable parser adapters cannot define domain identity.
- Hierarchical consumers retain an exact path to every character and physical object.
- Conservative rules intentionally leave OCR-corrupted, visual-only, and ambiguous numbering cases
  unresolved.
- Cross-parser path differences are reported as consistency, not accuracy.
- Duplicate-first-wins remains deliberately conservative after TOC and prose suppression: later
  duplicate structural keys remain BODY and are exposed in diagnostics.
- No entity extraction, legal-reference resolution, semantic edge, retrieval, RAG, knowledge graph,
  VLM, or LLM behavior is introduced.

## Boundary to Issue #009

Issue #009 may use the committed ambiguity/failure evidence to decide where selective visual or
semantic processing is justified. It must not retroactively turn Structural IR v1 into a semantic
legal model. This ADR implements no part of Issue #009.
