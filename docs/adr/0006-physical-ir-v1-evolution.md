# ADR 0006 — Evolve Physical Document IR to v1 without rewriting v0 evidence

- **Status:** Accepted
- **Date:** 2026-09-05

## Context

Issue #006 retained ten parser/document runs across six Vietnamese legal/planning documents. Their
raw evidence contains 55 Marker `Table` objects and 52 MinerU `table` objects that Physical IR v0
could represent only as UNKNOWN. The same evidence contains one Marker `Figure`, five native image
or picture objects, embedded or relative visual bytes, and text whose extraction method is not
represented in v0.

Physical IR normalizes parser observations; it does not correct those observations using an
evaluation label. In particular, the audited diagram on TT04/2023 PDF page 22 (page index 21) is a
Marker `Figure` but a MinerU `table`. Both raw classifications must survive truthfully.

The historical name and wire number differ: the research/model generation called Physical IR v0
already serializes `physical_ir_version: 1`. Rewriting that value would invalidate research
artifacts and ten deterministic normalization hashes.

## Decision

Physical IR v0 remains frozen in `physical_ir/models.py`, `physical_ir/serialization.py`, and the
existing Marker/MinerU normalizers. Physical IR v1 is additive and serializes wire/schema version
2:

| Research/model generation | Serialized `physical_ir_version` |
|---|---:|
| Physical IR v0 (frozen) | 1 |
| Physical IR v1 (current) | 2 |

The version-aware loader dispatches only from the explicit integer version. It neither guesses nor
upgrades. A v0 UNKNOWN block has already lost its raw type, so full-fidelity v1 requires
re-normalization from retained raw evidence.

### Block envelope and provenance

The v1 block kinds are TEXT, TITLE, HEADER, PAGE_NUMBER, TABLE, FIGURE, IMAGE, and UNKNOWN. No
legal, map, organization, or retrieval semantics are introduced. FIGURE means the parser reported
a figure-like physical object; IMAGE means it reported a raster/picture without that stronger
physical label.

Every block has parser-independent provenance fields `parser`, `parser_version`,
`parser_backend`, `source_raw_artifact`, and `source_raw_index`, plus required
`source_raw_type`. The latter retains the original case-sensitive parser label such as `Table` or
`table`; it is provenance data, not a parser-specific domain field.

### Tables

TABLE identity does not depend on structure availability. When a native Marker `html` or MinerU
`table_body` contains a defensible HTML table, a standard-library parser converts observed `tr`,
`td`, `th`, `rowspan`, and `colspan` into zero-based logical cells. Cell header state comes only
from `th` (`true`) or `td` (`false`), never from a first-row heuristic. Grid bounds and overlaps
are validated. No cell geometry or structure is inferred from line positions. Missing or malformed
structure leaves `table_structure` null without downgrading the block.

### Visual assets

Visual evidence has controlled states `relative_file`, `embedded_raw`, and `unavailable`.
Machine-absolute and parent-traversing paths are rejected. Retained bytes are fingerprinted with
SHA-256 and byte size; media type is recorded only from a recognized byte signature. Marker base64
is decoded in memory and not materialized. MinerU paths are resolved within the raw directory.
In-memory normalization performs no filesystem mutation, and normalized output cannot be written
inside raw or over explicit source/run evidence.

### Text extraction evidence

The controlled methods are `native_text`, `ocr`, and `unknown`; confidence is nullable and bounded
to [0, 1]. A confidence is never synthesized.

For the retained Marker 2.0.0 baseline, the trusted run evidence at each parser-run `run.json`
records `disable_ocr: true`. Therefore a non-empty Marker block is labeled `native_text`; an empty
block has no extraction record. This states that OCR was disabled, not that the parser text is
accurate.

MinerU 3.4.5 pipeline content lists contain text, and `middle.json` contains span/layout `score`
values, but the retained artifacts contain no field distinguishing OCR from native PDF text. All
non-empty MinerU block text is therefore `unknown`, and confidence is null. Scanned-page benchmark
labels, annotations, and research reports are deliberately excluded from normalizer inputs.

### Backward-compatible implementation

The v1 normalizers first invoke their frozen v0 counterpart for the established provenance,
geometry, ordering, and raw-input validation, then enrich the one-to-one block sequence from raw
parser evidence. This reuses proven invariants without changing v0 output bytes. Separate v1
models and serializers enforce immutable strict Pydantic records, contiguous pages and reading
order, unique IDs, conditional table/visual fields, deterministic UTF-8/LF JSON, and terminal
newline.

## Consequences

- All 107 observed tables retain TABLE identity; recoverable structure is evidence-dependent.
- Parser-native visual identity and byte provenance survive without committing derived images.
- Extraction provenance represents uncertainty explicitly; no OCR confidence exists in this
  corpus evidence.
- The TT04/2023 disagreement remains Marker FIGURE versus MinerU TABLE. Evaluation may score that
  disagreement but normalization cannot rewrite it.
- Existing Issue #004–#006 artifacts and evaluations continue to use v0/wire version 1 unchanged.
- UNKNOWN remains valid for unsupported raw labels. LIST, FOOTER, CAPTION, semantic MAP meaning,
  Structural IR, and RAG are non-goals for Issue #007.
