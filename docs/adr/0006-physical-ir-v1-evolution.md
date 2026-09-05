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

`disposition` describes the current v1 observation, independently of the old v0 kind:
CONTENT is document body content, DISCARDED is intentionally excluded boilerplate, and UNKNOWN
means the normalizer cannot determine disposition. In particular, newly recognized MinerU
`table` and `image` observations are CONTENT; they do not retain the UNKNOWN disposition that v0
used when it lacked those kinds. Marker Table, Figure, and Picture observations remain CONTENT,
while headers and page numbers remain DISCARDED where the parser mapping establishes boilerplate.

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

An explicit `<tr>` establishes one observed logical row. `rowspan` may occupy later explicit rows,
but it may not extend beyond the final explicit `<tr>` count; such markup is rejected rather than
clipped or allowed to fabricate rows. There is no analogous arbitrary rule limiting `colspan`
beyond the normal non-overlap and grid invariants. Cell text preserves `<br>` as `\n`; horizontal
and incidental source whitespace is collapsed within each `<br>`-delimited line, each line is
trimmed, and consecutive `<br>` elements preserve deterministic empty lines. Vietnamese Unicode
text is not transliterated or ASCII-escaped. `table_structure: null` deliberately does not
distinguish absent markup from rejected markup in this schema version.

### Visual assets

Visual evidence has controlled states `relative_file`, `embedded_raw`, and `unavailable`.
Machine-absolute and parent-traversing paths are rejected. Retained bytes are fingerprinted with
SHA-256 and byte size; media type is recorded only from a recognized byte signature. Marker base64
is decoded in memory and not materialized. MinerU paths are resolved within the raw directory.
In-memory normalization performs no filesystem mutation, and normalized output cannot be written
inside raw or over explicit source/run evidence.

`relative_file` is a verified retained-file claim and therefore requires all of `relative_path`,
`sha256`, and positive `byte_size`; `media_type` may be null when the byte signature is not one of
the recognized image signatures. A parser reference whose file is not retained is UNAVAILABLE,
not a path-only RELATIVE_FILE. EMBEDDED_RAW requires media type, SHA-256, and byte size and forbids
a relative path. UNAVAILABLE forbids all path and byte metadata.

### Text extraction evidence

The controlled methods are `native_text`, `ocr`, and `unknown`; confidence is nullable and bounded
to [0, 1]. A confidence is never synthesized.

For Marker, `disable_ocr: true` alone does not prove native text. A non-empty block is labeled
`native_text` only when the validated run configuration disables OCR **and** the retained
page-level `source_meta.json` positively records `text_extraction_method: pdftext` for that block's
page. This page/provider evidence legitimately applies to all non-empty block types on that page,
including structured types. Missing, different, or unavailable page/provider evidence yields
`unknown`; an empty block has no extraction record. NATIVE_TEXT means the characters were
positively established as coming through the native PDF text-layer/extraction path. It does not
claim that the text is accurate, born-digital, or supported by benchmark annotations.

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

Page dimensions, bounding-box coordinates, and extraction confidence must be finite. The v1
serializer also uses JSON's strict non-finite-number prohibition (`allow_nan=false`) as a defense
in depth, so it can never emit `NaN`, `Infinity`, or `-Infinity`. Positional block IDs are stable
only for identical normalized input and ordering; they are not persistent identities across
parser versions, parsers, or document versions.

The retained-corpus evidence remains named `physical_ir_v1_validation.v1.json` because `v1`
identifies this Physical IR validation protocol. Its internal `validation_schema_version` is 2
after the correction added kind/disposition, extraction-source, rowspan, unknown-evidence, and
verified-asset matrices; consumers must use the internal version for artifact-shape dispatch.

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
