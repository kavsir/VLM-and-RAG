# 4. Physical Document Intermediate Representation (v0)

Date: 2026-09-04

## Status

Accepted

## Context

Document parsers are volatile external runtimes. MinerU 3.4.5 produces three distinct, overlapping JSON
representations (`middle.json`, `content_list.json`, and `content_list_v2.json`) with different block
namings and structures. Downstream retrieval and semantic processing must not couple to any parser's
idiosyncratic output schema, nor should downstream code ingest raw parser JSON directly.

Furthermore, physical layout evidence ("what text, titles, bboxes, and pages physically exist") must be
distinguished from semantic understanding ("what does this text mean, what entities exist, how should it
be chunked").

## Decision

We introduce **Physical Document IR v0** as a strict, parser-independent representation of layout evidence:

1. **Physical IR Purpose**: Captures layout blocks, reading order, bounding boxes, and page dimensions
   without performing semantic interpretation, entity extraction, or chunking.
2. **Parser Decoupling**: MinerU schema is not our schema. Raw parser JSON is normalized via an explicit
   adapter (`MinerUPhysicalNormalizer`) and does not leak downstream.
3. **v0 Supported Concepts**:
   - `PhysicalDocument`: Top-level document container holding metadata, page count, and ordered pages.
   - `PhysicalPage`: 0-based page index, page dimensions (`width`, `height` in points when available),
     and ordered blocks.
   - `PhysicalBlock`: Individual layout block with unique ID, page index, reading order, kind,
     disposition, text, bounding box, optional heading level, and parser provenance.
   - `BlockKind`: `TEXT`, `TITLE`, `HEADER`, `PAGE_NUMBER`, `UNKNOWN`.
   - `BlockDisposition`: `CONTENT` (participates in standard retrieval) vs `DISCARDED` (headers, page
     numbers preserved for layout traceability but flagged as non-content).
4. **Intentionally Unsupported / Unvalidated**:
   - Multimodal blocks (`TABLE`, `IMAGE`, `FORMULA`, `CHART`, `CAPTION`) are intentionally omitted from
     validated enums in v0 because the Hanoi golden document (Decision 2512/QĐ-UBND) contains 0 multimodal
     elements. They will be validated in subsequent issues when multimodal fixtures are available.
   - Semantic fields (chunks, embeddings, entities, relations, summaries) are strictly excluded.
5. **Coordinate Convention**:
   - Origin `(0, 0)` is at the top-left corner of the document page.
   - Axis orientation: `x` increases rightward, `y` increases downward.
   - Invariant: `x0 <= x1` and `y0 <= y1`, with `x0 >= 0` and `y0 >= 0`.
   - Coordinate space: `coordinate_system = "normalized_1000"`, representing MinerU's native `0..1000`
     coordinate grid, while `width` and `height` on `PhysicalPage` preserve PDF points (72 DPI).
6. **Reading-Order Convention**:
   - Page-local 0-based integer sequence (`0, 1, 2, ...`).
   - Monotonically contiguous across blocks on each page.
7. **Deterministic Block IDs**:
   - Format: `{document_id}_{version_id}_p{page_idx:04d}_b{reading_order:04d}`.
   - Stable and repeatable across multiple normalization runs on identical raw input.
   - Independent of mutable text content.
8. **Provenance Strategy**:
   - Every block contains `BlockProvenance` recording parser name, version, backend, raw source
     artifact relative path, and 0-based index in the raw content list.
9. **Raw Format Selection**:
   - `source_content_list.json` is selected as the primary block source because it preserves linear
     reading order, block text, bounding boxes, and `text_level` (differentiating titles from paragraphs).
   - `source_middle.json` is used as a secondary source to extract physical page dimensions (`page_size`
     in points) and parser runtime metadata (`_backend`, `_version_name`).
   - `source_content_list_v2.json` was rejected as primary input because it omits `text_level` and
     nests text in complex wrapper objects without block-level page indexes.
10. **Schema Versioning**:
    - `physical_ir_version` (integer `1`) versions the Physical IR data schema independently of the
      registry's document `version_id` (`v1`).

## Consequences

* The core domain models remain completely decoupled from MinerU, Marker, or any future parser.
* Full raw physical evidence is preserved, including headers and page numbers.
* Normalization is 100% deterministic, offline, and verifiable via automated test fixtures.
* Downstream RAG and VLM modules can build upon a stable, typed foundation without re-parsing raw PDFs.
