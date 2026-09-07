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
nodes may nest below an appendix, another generic node, an active article/clause when a planning
outline is physically embedded in a decision article, or the deepest active compatible formal
container (`SUBSECTION`, `SECTION`, `CHAPTER`, then `PART`).

State transitions are deterministic. A new legal sibling closes its descendants; an appendix closes
the legal and generic stacks. A generic child may be rooted under a legal ARTICLE/CLAUSE, but once a
planning outline is active its decimal and letter children take precedence over stale legal
CLAUSE/POINT state. A sequential Arabic clause marker may explicitly re-enter the active article. A
generic outline selected under a formal non-legal container closes incompatible stale
ARTICLE/CLAUSE/POINT state.

### Deterministic identity and path

Node IDs are zero-padded reading-order sequences. They are deterministic for identical normalized
input and order, not persistent identity across parser representations or document versions.
Ordinals preserve `ordinal_raw` and a kind-aware normalized `ordinal_key`. One parser-independent
ordinal module is used by extraction, the Structural domain schema, and the reference schema, so an
internally path-consistent but semantically corrupt pair is invalid. Article and clause Arabic
ordinals preserve lowercase amendment suffixes, POINT ordinals preserve Vietnamese letters (including
`c`, `d`, `i`, `l`, `m`, and `đ`), and only Roman-numbered structural kinds use strict Roman
conversion. Non-canonical Roman spellings such as `IIII`, `IC`, `VX`, `IIV`, and `MMMM` are rejected.

Canonical paths use controlled kind/ordinal segments and never raw parser IDs or occurrence suffixes.
A non-root node without an ordinal must end in the exact `<kind>:unnumbered` segment (or
`generic:unnumbered`); marker/title text cannot invent another value.
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
`GENERIC_SECTION`. Under an evidenced non-legal APPENDIX/generic outline, uppercase-initial decimal
and letter items may become nested `GENERIC_SECTION` nodes using explicit Roman, decimal, or letter
recognition methods. A generic letter is never a legal POINT unless a compatible legal CLAUSE is active.
A simple Arabic generic heading otherwise requires stronger physical/typographic evidence.
A multi-component decimal in ordinary text requires either strong heading evidence or an exact active
numeric generic parent; short text alone is insufficient.
Letter items outside compatible legal or generic context remain body content. Immediate title
continuation is accepted only for a physical title or a short uppercase heading-like line.

The rules are anchored at line start. Embedded phrases such as “theo Điều 3”, “tại điểm a khoản 2”,
or “thực hiện Chương II” do not create nodes. Formal no-separator prose is rejected before
parser TITLE evidence is considered; a no-separator container title requires uppercase structural
form. An explicit `MỤC LỤC` starts a locally contiguous sequence of entries ending in dotted page
leaders. Pending fragments are committed only when a nearby entry terminator exists, the search is
bounded, and a genuine formal heading without a leader terminates the region. Without the explicit
heading, only individual strong marker-shaped leader events are suppressed, including multi-page TOC
continuations. TOC text remains BODY and cannot reserve a key; later same-page body structure is still
eligible. Corpus-evidenced inline segmentation may recover combined formal headings or a flattened
second legal clause/list, while retaining original half-open offsets with no gaps or overlap.

APPENDIX is terminal in profile v1. The retained corpus does not require returning from an appendix to
the outer legal hierarchy; a future profile must add evidence for such a transition rather than infer it.

### Evaluation and reproducibility

Reference annotation v4/schema 4 is an AI visual PDF structural re-audit, not human ground truth. It
discloses prior Physical IR and Structural extractor exposure, states that neither output was used as
reference truth, and contains no parser/block identity. V4 changes no visual truth from v3. A stable
`reference_instance_id` identifies the actual visual structural instance; repeated unscored context
records reuse it. A separate `reference_record_id` identifies each page-local annotation record.
`parent_reference_instance_id` binds the exact visual parent and must agree with
`parent_canonical_path`; QD23 Clause 1–4 records therefore explicitly belong to the first of its two
Article 2 instances. Repeated scored canonical paths remain legal, while record IDs are unique,
instance definitions (including the source marker page) are consistent, parents resolve, parent
markers cannot begin after their children, and instance cycles are forbidden. Page ordering does not
disambiguate duplicated canonical parents that begin on the same page; without other existing
non-title structural evidence, either explicit same-page assignment remains an acknowledged v4
limitation. No geometry, bounding-box containment, or reading-order inference is claimed. The 46-page
log and render hashes are committed in
`data/structural_annotations/reference_structural_audit.v4.json`.

Matching requires exact audited page, kind, and ordinal. Singleton groups match directly; duplicate
groups match only through exact parent canonical paths. Indistinguishable duplicates may contribute
the conservative `min(N, M)` detection count, but reference IDs never determine their pairing and all
such pairs are excluded from title accuracy. Residual ambiguous groups remain unmatched, so occurrence
shifting cannot create a false true-positive. Full parent-edge metrics include
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
- Duplicate-first-wins remains deliberately conservative after TOC, prose, and stale-stack
  correction: indistinguishable later Structural IR keys remain BODY and are exposed in diagnostics;
  visually genuine reference instances are never deleted to improve extractor scores.
- No entity extraction, legal-reference resolution, semantic edge, retrieval, RAG, knowledge graph,
  VLM, or LLM behavior is introduced.

## Boundary to Issue #009

Issue #009 may use the committed ambiguity/failure evidence to decide where selective visual or
semantic processing is justified. It must not retroactively turn Structural IR v1 into a semantic
legal model. This ADR implements no part of Issue #009.
