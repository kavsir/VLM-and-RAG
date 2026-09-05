# Vietnamese Legal and Planning Reference Corpus v1

The corpus registers 6 official documents and 250 source pages. Source PDFs are identified by immutable SHA-256 values.

| Document | Official number | Pages | Issued | Effective | Signer | Source SHA-256 |
| --- | --- | ---: | --- | --- | --- | --- |
| `hanoi-master-plan-100y` | `2512/QĐ-UBND` | 80 | 2026-05-13 | 2026-05-13 | Vũ Đại Thắng | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| `luat-112-2025-qh15` | `112/2025/QH15` | 52 | 2025-12-10 | 2026-03-01 | Trần Thanh Mẫn | `8d9208d6105c96a44547f1232af9f2127ba0b4c2661ee08acb7d04573f342e8b` |
| `vbhn-103-2026-quy-hoach-tong-the` | `103/VBHN-VPQH` | 48 | 2026-04-22 | explicit null | Bùi Văn Cường | `342b4f13dbd6eb644c9b73201d3951d85968cb316656969ace0b74a50a3ae109` |
| `tt-04-2026-bxd-pl2-dinh-muc` | `04/2026/TT-BXD/PL2` | 19 | 2026-01-30 | 2026-02-01 | Nguyễn Danh Huy | `f1725709e9df7135c729b2fcc8905638bdd6312af8656459a7475a8c504c0b25` |
| `tt-04-2023-bkhdt-so-do-ban-do` | `04/2023/TT-BKHĐT` | 47 | 2023-06-26 | 2023-08-10 | Trần Quốc Phương | `196c79b3399e78787528a513ef6efd6d659f5e083845cd43d6b435b920c51f1b` |
| `qd-23-2008-ubnd-hanoi-vien-quy-hoach` | `23/2008/QĐ-UBND` | 4 | 2008-09-29 | 2008-10-09 | Nguyễn Thế Thảo | `2c6276e8d611a827a220abe1106008851f4a049495f80277a62332a0c0d119fc` |

## Reference annotation policy

Version v3 uses `visual_pdf_reaudit` AI visual reference annotations. For digital pages, regions target visually separable text groups, headings, coherent outer tables, and visible figures at the selected audit granularity. For scanned pages, modality is separate and full-page raster boxes are not treated as OCR text truth.

Per-page re-audit evidence is committed at `data/annotations/reference_annotation_audit.v3.json` with SHA-256 `d2cff34152d26370ecd2043201034b9b05aeef678957acaf92c0316fc30ac1da`.

These are reference annotations, not human ground truth. Prior parser-output exposure existed, but parser output was not the reference source for geometry or region type during the visual re-audit.
