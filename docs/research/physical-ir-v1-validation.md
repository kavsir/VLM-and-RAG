# Physical IR v1 retained-corpus validation

> Generated from `data/benchmarks/physical_ir_v1_validation.v1.json`. Do not edit measured values by hand.

Physical IR research generation **v1** uses wire/schema version **2**. Historical Physical IR v0 remains wire version 1.

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
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 169 | 27 | 25 | 933 | 0 | 0 | 0 | `18a19cdeb7f5b75fabed529bbd3d10b43592ec845463142a64f85d2fe24bb141` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 413 | 24 | 11 | 1106 | 1 | 0 | 16 | `45137e123dfad108b946587ca2bb0455ec831655475e5d7874a97117ba404cbe` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 385 | 25 | 18 | 2127 | 0 | 0 | 0 | `336ba840df09bc50aed4164c85be296a6cad32dd4f619f246d9f5c8a8e61d364` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 74 | 0 | 0 | 0 | 0 | 1 | 0 | `5a3afaad4af491c748616d8efd37f883de905efa81d76f3873c5515e0b0cd703` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 82 | 0 | 0 | 0 | 0 | 1 | 0 | `afa8cda92b0ab496eddb3d967274e830bc3f464642cccc2308cc442929cf68a2` |

## Table structure coverage by parser

| Parser | Native tables | Structured tables | Cells |
|---|---:|---:|---:|
| marker | 55 | 39 | 3205 |
| mineru | 52 | 43 | 3060 |

## Representation fidelity gained from raw evidence

- Tables: **107**; recoverable logical structure: **82**; cells: **6265**.
- Table spans: rowspan **438**, colspan **173**, parser `<th>` cells **179**.
- Visuals: FIGURE **1**, IMAGE **5**; asset hashes **6/6**.
- Kind transitions from frozen v0: `{"unknown->figure": 1, "unknown->image": 5, "unknown->table": 107}`.
- Extraction methods: `{"native_text": 2739, "not_recorded": 493, "unknown": 1737}`; confidence coverage **0**.

These are representation-fidelity changes, not parser-accuracy improvements. On TT04/2023 page index 21, Marker raw `Figure` maps to FIGURE, while MinerU raw `table` maps to TABLE.

## Extraction evidence policy

Marker text-bearing blocks are `native_text` only because the retained run metadata records `disable_ocr=true`; empty blocks carry no text-extraction record. MinerU 3.4.5 pipeline artifacts expose span/layout scores but no field distinguishing OCR from native PDF text, so non-empty MinerU text is `unknown` and confidence remains null. Annotation modality labels are never normalizer inputs.

## Limitations

Tables whose observed HTML cannot be converted safely retain TABLE identity with null structure. UNKNOWN remains intentional for unsupported raw types. No OCR accuracy, semantic map meaning, or structural/legal interpretation is claimed.
