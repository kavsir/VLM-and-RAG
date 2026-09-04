# Physical Document IR v0: Empirical Gap Analysis & v1 Evolution Roadmap

## Executive Summary

Issue #006 evaluated `Physical Document IR v0` across a 6-document golden corpus spanning 250 pages of Vietnamese statutes, planning resolutions, dense engineering tables, spatial layout diagrams, and scanned decrees.

While `Physical IR v0` successfully proved **parser independence** by normalizing both MinerU 3.4.5 and Marker 2.0.0 into a unified schema with deterministic serialization, empirical benchmark evidence revealed several representational gaps that motivate the design of `Physical Document IR v1`.

---

## 1. Concrete Gaps Identified from Benchmark Evidence

### Gap 1: Tabular Structure Flattening
- **Empirical Evidence**: In `Thông tư 04/2026/TT-BXD Phụ lục II` (19 pages of maintenance norms) and `hanoi-master-plan-100y`, dense multi-column tables dominate the document.
- **Raw Parser Representations**: MinerU emits structured `table` layout blocks with HTML/markdown representations; Marker emits `Table` blocks.
- **Normalized v0 Behavior**: Because `BlockKind` in v0 only defines `TEXT`, `TITLE`, `HEADER`, `PAGE_NUMBER`, and `UNKNOWN`, MinerU table blocks are mapped to `BlockKind.TEXT`, while Marker tables are mapped to `BlockKind.TEXT` (or `BlockKind.UNKNOWN` when structural parsing is incomplete).
- **Information Lost**: Cell bounding boxes, row spans, column spans, and table header hierarchies are erased into unsegmented text strings.

### Gap 2: Multimodal Graphics & Diagram Loss
- **Empirical Evidence**: In `Thông tư 04/2023/TT-BKHĐT`, **PDF page 21 (page_index=20)** contains a formal layout diagram defining national map sheet compositions, and **PDF pages 26-31 (page_index=25 to 30)** contain map symbology sheets.
- **Raw Parser Representations**: Marker identifies `Picture`/`Figure` regions; MinerU layout analysis identifies `image` regions.
- **Normalized v0 Behavior**: `BlockKind` provides no representation for image/figure objects. Marker maps them to `BlockKind.UNKNOWN` or discards them; MinerU drops image crops.
- **Information Lost**: Spatial coordinates of diagrams, bounding boxes of symbology figures, and relative image asset links are omitted from the physical IR.

### Gap 3: Legal Numbered Provisions Merged into Narrative Text
- **Empirical Evidence**: In `Luật Quy hoạch 112/2025/QH15` (52 pages) and `VBHN 103/VBHN-VPQH` (48 pages), statutory provisions follow a strict hierarchy (`Chương` -> `Mục` -> `Điều` -> `Khoản` -> `Điểm`).
- **Raw Parser Representations**: Parsers recognize major headings (`Chương`, `Điều`) as headings/titles, but emit subsequent provisions (`Khoản`, `Điểm`) as standard body paragraphs.
- **Normalized v0 Behavior**: Major headings are normalized to `BlockKind.TITLE` (with `heading_level`), but `Khoản` and `Điểm` remain generic `BlockKind.TEXT`.
- **Information Lost**: Sub-article structural hierarchy and provision numbering are lost to downstream consumers without NLP regex re-splitting.

### Gap 4: Document Modality (Native Vector vs Scanned Raster)
- **Empirical Evidence**: On `Quyết định 23/2008/QĐ-UBND` (4 pages, 100% scanned raster), Marker fast-no-ocr produced 0 text blocks, whereas MinerU pipeline recognized 82 OCR text and title blocks.
- **Normalized v0 Behavior**: `PhysicalBlock` contains no field indicating whether extracted text originates from native PDF digital font streams or OCR inference.
- **Information Lost**: Downstream consumers cannot determine optical recognition confidence or distinguish authentic native digital text from potential OCR noise.

---

## 2. Quantitative Summary Across Evaluated Runs

| Dimension | Marker 2.0.0 (fast-no-ocr) | MinerU 3.4.5 (pipeline CPU) | Observation |
| :--- | :---: | :---: | :--- |
| **Throughput (CPU)** | **0.50 - 1.17 pages/sec** | 0.05 - 0.15 pages/sec | Marker is 5x-15x faster on CPU for born-digital documents. |
| **Scanned Page Modality** | 0 blocks (skips OCR) | **82 OCR blocks recovered** | Marker requires OCR policy for scanned documents; MinerU handles scans automatically. |
| **Tabular Order Accuracy** | 0.7143 | **0.7619** | MinerU preserves vertical column reading order slightly better on dense norms. |
| **Deterministic Output** | 100% (Bitwise identical) | 100% (Bitwise identical) | Both normalizers achieve bitwise-reproducible PhysicalDocument JSON. |

---

## 3. Physical IR v1 Evolution Proposals

1. **Expanded `BlockKind` Enum**:
   Add `TABLE`, `FIGURE`, `LIST_ITEM`, and `FOOTER` to `BlockKind`.
2. **Optional `TableStructure` Payload**:
   Attach structured cell bounding boxes and row/column indices to `TABLE` blocks.
3. **Modality & Provenance Declarations**:
   Record `extraction_modality: Literal["native_digital", "ocr", "hybrid"]` in `BlockProvenance`.
4. **Visual Crop File Linkage**:
   Record relative paths to extracted diagram image files in parser run output directories.
