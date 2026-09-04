# Physical Document IR v0: Empirical Gap Analysis & v1 Evolution Roadmap

## Executive Summary

Issue #006 tested `Physical Document IR v0` across a diverse 6-document golden corpus spanning 249 pages of Vietnamese legal statutes, planning resolutions, dense engineering norm tables, spatial layout diagrams, map symbology sheets, and scanned government decrees.

While `Physical IR v0` successfully proved **parser independence** by normalizing both MinerU 3.4.5 and Marker 2.0.0 into a unified schema with deterministic serialization, this comprehensive benchmark revealed several critical representational gaps that motivate the future design of `Physical Document IR v1`.

---

## 1. Concrete Gaps Identified from Benchmark Evidence

### Gap 1: Absence of `BlockKind.TABLE` (Tabular Data Flattening)
- **Empirical Evidence**: In `Thông tư 04/2026/TT-BXD Phụ lục II` (19 pages of maintenance norms) and the annexes of `hanoi-master-plan-100y`, dense multi-column tables dominate the document.
- **v0 Behavior**: Because `BlockKind` only defines `TEXT`, `TITLE`, `HEADER`, `PAGE_NUMBER`, and `UNKNOWN`, both the MinerU and Marker normalizers are forced to map tabular blocks into `BlockKind.TEXT`.
- **Impact**: Downstream consumers (e.g. VLM extractors, chunkers, RAG embedding pipelines) cannot determine whether a block contains narrative legal prose or a multi-column numeric matrix without re-parsing raw text strings. Cell coordinates, column delimitations, and row span relationships are completely erased.

### Gap 2: Absence of `BlockKind.FIGURE` / `IMAGE` (Multimodal Graphics Loss)
- **Empirical Evidence**: In `Thông tư 04/2023/TT-BKHĐT`, Page 21 contains a formal layout diagram defining national map sheet compositions, and Pages 26–31 contain raster symbology sheets illustrating spatial planning markers.
- **v0 Behavior**: `BlockKind` provides no representation for image/figure objects. Marker normalizes them into `BlockKind.UNKNOWN` with disposition `CONTENT` or `DISCARDED`, while MinerU layout analysis discards raw image embeddings.
- **Impact**: True multimodal RAG and VLM reasoning require visual bounding boxes to locate diagrams, figures, and maps for image-patch embedding or vision-model querying. Physical IR v0 currently blinds multimodal pipelines to visual assets.

### Gap 3: Hierarchical Legal Provisions Collapsed into Monolithic Text
- **Empirical Evidence**: In `Luật Quy hoạch 112/2025/QH15` (52 pages) and `VBHN 103/VBHN-VPQH`, Vietnamese legislation follows a strict hierarchy:
  $$\text{Chương} \rightarrow \text{Mục} \rightarrow \text{Điều} \rightarrow \text{Khoản} \rightarrow \text{Điểm}$$
- **v0 Behavior**: While `Chương` and `Điều` are recognized as `TITLE` blocks (with `heading_level` 1, 2, or 3), all subsequent provisions (`Khoản 1`, `Điểm a`, `Điểm b`) are merged into generic `BlockKind.TEXT`.
- **Impact**: Legal RAG systems require precise clause-level chunking and citation retrieval. Collapsing numbered provisions into standard paragraphs forces downstream NLP components to perform error-prone heuristic text regex re-splitting.

### Gap 4: Scanned / OCR Modality Ambiguity
- **Empirical Evidence**: On `Quyết định 23/2008/QĐ-UBND` (4 pages, 100% scanned raster), Marker (running with `fast-no-ocr`) produced 0 text blocks, whereas MinerU (running pipeline OCR) recognized 82 text and title blocks.
- **v0 Behavior**: Both results conform to `PhysicalDocument`, but Physical IR v0 contains no field indicating whether the extracted text originates from native PDF digital font streams or OCR inference.
- **Impact**: Downstream agents cannot assess optical recognition noise, character error rates, or distinguish authentic born-digital text from OCR hallucinations without inspecting low-level raw logs.

### Gap 5: Coarse Single-Bounding-Box Geometry
- **Empirical Evidence**: Both MinerU and Marker emit single outer bounding rectangles for multi-line paragraphs or entire table regions.
- **v0 Behavior**: `PhysicalBlock.bbox` is a single `BoundingBox` on a normalized 1000-point grid.
- **Impact**: When text blocks span complex non-rectangular geometries (e.g. text wrapping around a signature stamp or side note), a single bounding rectangle introduces significant empty space, lowering spatial IoU matching accuracy (mean IoU between 0.65 and 0.75).

---

## 2. Quantitative Benchmark Findings Summary

| Metric | MinerU 3.4.5 (Pipeline CPU) | Marker 2.0.0 (Fast-No-OCR CPU) | Primary Driver |
| :--- | :---: | :---: | :--- |
| **Born-Digital Text Recall** | ~65–75% | ~65–70% | Both parsers reliably segment narrative legal paragraphs. |
| **Scanned Document Recall** | **High (>80%)** | **0.0% (Failed)** | Marker fast-no-ocr skips OCR entirely; MinerU runs full OCR layout. |
| **Tabular Reading Concordance** | **0.5238** | 0.4286 | MinerU preserves vertical column flow slightly better than Marker. |
| **Throughput (Speed)** | ~0.15 pages/sec | **~1.5–2.5 pages/sec** | Marker is ~10–15x faster on CPU due to no-OCR heuristics. |
| **Deterministic Serializability** | 100% (Bitwise identical) | 100% (Bitwise identical) | Both normalizers achieve strict bitwise determinism. |

---

## 3. Physical IR v1 Evolution Roadmap

To address these empirical findings while maintaining architectural purity and parser independence, we propose the following evolutionary design for `Physical Document IR v1`:

1. **Expanded `BlockKind` Enum (Backward-Compatible)**:
   ```python
   class BlockKind(StrEnum):
       TEXT = "text"
       TITLE = "title"
       TABLE = "table"  # NEW in v1
       FIGURE = "figure"  # NEW in v1
       LIST_ITEM = "list_item"  # NEW in v1
       HEADER = "header"
       FOOTER = "footer"  # NEW in v1
       PAGE_NUMBER = "page_number"
       UNKNOWN = "unknown"
   ```

2. **Optional Table Structure Model**:
   - Introduce an optional `TableStructure` payload on `PhysicalBlock` when `kind == BlockKind.TABLE`:
     ```python
     class TableCell(PhysicalIRModel):
         row_index: NonNegativeInt
         col_index: NonNegativeInt
         row_span: PositiveInt = 1
         col_span: PositiveInt = 1
         bbox: BoundingBox
         text: str


     class TableStructure(PhysicalIRModel):
         num_rows: PositiveInt
         num_cols: PositiveInt
         cells: tuple[TableCell, ...]
     ```

3. **Modality & Provenance Annotations**:
   - Enrich `BlockProvenance` to explicitly declare extraction modality:
     ```python
     class ExtractionModality(StrEnum):
         NATIVE_DIGITAL = "native_digital"
         OCR = "ocr"
         HYBRID = "hybrid"
     ```

4. **Visual Asset Artifact Linkage**:
   - For `FIGURE` blocks, record relative paths to extracted image crops in the parser run directory, enabling VLM ingestion without re-rendering PDF pages.
