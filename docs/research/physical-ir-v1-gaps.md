# Physical Document IR v0: Evidence-Based Gaps

This report records observed representational gaps only; it does not implement Physical IR v1.

## TABLE representation

- **Actual page evidence**: `tt-04-2026-bxd-pl2-dinh-muc`, PDF page 2 (page_index=1) contains a visually audited norm table.

- **Raw `marker` representation and v0 mapping**: 55 parser-native table objects map to 55 as `unknown`.
- **Raw `mineru` representation and v0 mapping**: 52 parser-native table objects map to 52 as `unknown`.

- **Specific information loss**: The counts follow parser-native table objects through `source_raw_index`. Separately emitted nested text may become TEXT, but that does not preserve the table object, cells, spans, or header structure.

- **Machine-derived frequency**: 107 table objects were traced across the retained runs.

- **Candidate future requirement**: Represent TABLE objects and structured cells.

## FIGURE / MAP representation

- **Actual page evidence**: TT04/2023 contains a verified layout diagram on PDF page 22 (page_index=21) and later visually verified symbology sheets.

- **Raw `marker` representation and v0 mapping**: 1 `Figure` map to 1 `unknown`.
- **Raw `mineru` representation and v0 mapping**: 1 `table` map to 1 `unknown`.

- **Specific information loss**: Physical IR v0 has no FIGURE/MAP kind or retained visual-asset link, so graphical identity and asset provenance are lost.

- **Machine-derived frequency**: the page-specific provenance trace covers 2 parser-native diagram candidates across the retained parser runs; it is not a corpus-wide figure-frequency estimate.

- **Candidate future requirement**: Add typed visual objects and relative asset references.

## OCR / modality distinction

- **Actual page evidence**: `qd-23-2008-ubnd-hanoi-vien-quy-hoach` is raster-only on PDF page 1 (page_index=0), PDF page 2 (page_index=1), PDF page 3 (page_index=2), PDF page 4 (page_index=3).

- **Raw `marker` representation and v0 mapping**: 3 `PageHeader`, 1 `Picture`, 13 `SectionHeader`, 57 `Text`; normalized as 3 `header`, 57 `text`, 13 `title`, 1 `unknown`.
- **Raw `mineru` representation and v0 mapping**: 2 `header`, 1 `image`, 3 `page_number`, 76 `text`; normalized as 2 `header`, 3 `page_number`, 70 `text`, 6 `title`, 1 `unknown`.

- **Specific information loss**: Physical IR v0 does not distinguish native digital text from OCR-derived text or attach OCR confidence. No OCR recall, CER, or WER conclusion is possible without transcription reference data.

- **Machine-derived frequency**: marker produced 74 structural blocks, 0 with non-empty text; mineru produced 82 structural blocks, 81 with non-empty text.

- **Candidate future requirement**: Carry extraction modality, OCR confidence, and provenance without treating OCR output as transcription reference data.

## Clean-clone boundary

The quantitative statements above are rendered from the committed benchmark manifest. Recomputing them from parser-native evidence requires restoration of the external paths and hashes listed there.
