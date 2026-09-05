# Physical IR v1 retained-corpus validation

> Generated from `data/benchmarks/physical_ir_v1_validation.v1.json`. Do not edit measured values by hand.

Physical IR research generation **v1** uses wire/schema version **2**. Historical Physical IR v0 remains wire version 1.
This file remains `.v1.json` because that is the Physical IR v1 validation protocol name; `validation_schema_version: 2` records the corrected evidence artifact shape.

## Validation result

- Retained parser/document pairs: **10**
- Frozen v0 SHA/size checks preserved: **True**
- Byte-identical v1 A/B normalizations: **True**
- Pages / blocks: **400 / 4969**

## Per-pair representation

| Document | Parser | Pages | Blocks | Tables | Structured | Cells | Figures | Images | Unknown | v1 SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hanoi-master-plan-100y | marker | 80 | 1205 | 0 | 0 | 0 | 0 | 1 | 1 | `137baf6376ca1967ac9034c868203d0cf2d4251333dfa420dceaca7ea8df84b8` |
| hanoi-master-plan-100y | mineru | 80 | 1186 | 0 | 0 | 0 | 0 | 0 | 0 | `738a1bd80f481b9436d178844983697c4e6c9b0a6eb6e5aa147f2d1d5ff480de` |
| luat-112-2025-qh15 | marker | 52 | 812 | 3 | 3 | 438 | 0 | 0 | 20 | `7beaa58a0bcbcffa3d995bb6f0b12217bae95297f11dc3b01c1677d8504d392f` |
| vbhn-103-2026-quy-hoach-tong-the | marker | 48 | 489 | 1 | 1 | 70 | 0 | 1 | 47 | `cb1488e94be68a6e6cd2a0926ad8272facb339474808702a00e0a38d05ee3842` |
| tt-04-2026-bxd-pl2-dinh-muc | marker | 19 | 154 | 27 | 24 | 1591 | 0 | 1 | 5 | `1deaf7b67b5e86992a6bbae072ac1802528acc7f27245e963e09c2c59fe690ed` |
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 169 | 27 | 25 | 933 | 0 | 0 | 0 | `5c27e12e74f961958127342dff2a086f968127a1135d97d0a5a65e761e8b2be8` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 413 | 24 | 11 | 1106 | 1 | 0 | 16 | `45137e123dfad108b946587ca2bb0455ec831655475e5d7874a97117ba404cbe` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 385 | 25 | 18 | 2127 | 0 | 0 | 0 | `0e8b3f2ec69f22d77858f9aeec895110e99f5b8b8a3222ab1b2a1f8aea850670` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 74 | 0 | 0 | 0 | 0 | 1 | 0 | `5a3afaad4af491c748616d8efd37f883de905efa81d76f3873c5515e0b0cd703` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 82 | 0 | 0 | 0 | 0 | 1 | 0 | `9e32f090643e027b34dd5f83c586a4cba23cd7d28dd35bdb86de075fd5b73a0f` |

## Table structure coverage by parser

| Parser | Native tables | Structured tables | Cells |
|---|---:|---:|---:|
| marker | 55 | 39 | 3205 |
| mineru | 52 | 43 | 3060 |

## Kind by disposition by parser

Disposition describes the current v1 physical observation: `content` is body content, `discarded` is intentional boilerplate exclusion, and `unknown` means the disposition cannot be determined. It is not copied v0 kind uncertainty.

| Parser | Kind | Content | Discarded | Unknown |
|---|---|---:|---:|---:|
| marker | figure | 1 | 0 | 0 |
| marker | header | 0 | 380 | 0 |
| marker | image | 4 | 0 | 0 |
| marker | table | 55 | 0 | 0 |
| marker | text | 2282 | 0 | 0 |
| marker | title | 336 | 0 | 0 |
| marker | unknown | 89 | 0 | 0 |
| mineru | header | 0 | 50 | 0 |
| mineru | image | 1 | 0 | 0 |
| mineru | page_number | 0 | 130 | 0 |
| mineru | table | 52 | 0 | 0 |
| mineru | text | 1381 | 0 | 0 |
| mineru | title | 208 | 0 | 0 |

## Representation fidelity gained from raw evidence

- Tables: **107**; recoverable logical structure: **82**; cells: **6265**.
- Table spans: rowspan **438**, colspan **173**, parser `<th>` cells **179**.
- Explicit-row rowspan audit: **0** violations across **39** table markups containing `rowspan`. A cell may not extend beyond the observed `<tr>` count.
- Visuals: FIGURE **1**, IMAGE **5**; asset hashes **6/6**.
- Verified visual assets satisfying the storage contract: **6/6**. A `relative_file` requires a safe path, SHA-256, and positive byte size; media type may be null. Missing referenced bytes are represented as `unavailable`.
- Kind transitions from frozen v0: `{"unknown->figure": 1, "unknown->image": 5, "unknown->table": 107}`.
- Extraction methods: `{"native_text": 2739, "not_recorded": 493, "unknown": 1737}`; confidence coverage **0**.

These are representation-fidelity changes, not parser-accuracy improvements. On TT04/2023 page index 21, Marker raw `Figure` maps to FIGURE, while MinerU raw `table` maps to TABLE.

## Extraction evidence policy

Marker non-empty blocks are `native_text` only where both the validated run records `disable_ocr=true` and that page's retained `source_meta.json` records `text_extraction_method=pdftext`; empty blocks carry no text-extraction record. Missing or different page/provider evidence is `unknown`. MinerU 3.4.5 pipeline artifacts expose span/layout scores but no field distinguishing OCR from native PDF text, so non-empty MinerU text is `unknown` and confidence remains null. Annotation modality labels are never normalizer inputs.

### Marker source raw type by extraction method

| Source raw type | Native text | OCR | Unknown | Not recorded |
|---|---:|---:|---:|---:|
| Figure | 0 | 0 | 0 | 1 |
| Footnote | 40 | 0 | 0 | 0 |
| ListGroup | 46 | 0 | 0 | 0 |
| PageHeader | 229 | 0 | 0 | 151 |
| Picture | 0 | 0 | 0 | 4 |
| SectionHeader | 287 | 0 | 0 | 49 |
| Table | 39 | 0 | 0 | 16 |
| TableGroup | 1 | 0 | 0 | 1 |
| TableOfContents | 1 | 0 | 0 | 0 |
| Text | 2096 | 0 | 0 | 186 |

## Remaining unknown evidence

- UNKNOWN kind by parser/source raw type: `{"marker": {"Footnote": 40, "ListGroup": 46, "TableGroup": 2, "TableOfContents": 1}, "mineru": {}}`.
- UNKNOWN text extraction by parser/source raw type: `{"marker": {}, "mineru": {"header": 50, "page_number": 130, "text": 1557}}`.

## Limitations

`table_structure: null` can mean absent markup or rejected markup; v1 does not add a field distinguishing those causes. Tables whose observed HTML cannot be converted safely retain TABLE identity. UNKNOWN remains intentional for unsupported raw types. No OCR accuracy, semantic map meaning, or structural/legal interpretation is claimed.
