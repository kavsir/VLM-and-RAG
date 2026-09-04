# Vietnamese Legal, Administrative, and Planning Multimodal Golden Corpus (v1)

## Executive Summary

Issue #006 establishes the first authoritative, multimodal golden corpus specifically curated for benchmarking document parsing and layout extraction on Vietnamese legal, administrative, and spatial planning documents.

The corpus consists of **6 verified documents** (249 total pages) sourced exclusively from official government portals (`congbao.chinhphu.vn`, `vanban.chinhphu.vn`, `hanoi.gov.vn`). Every document is immutably registered in `data/manifests/` with cryptographic SHA-256 digests and exact byte sizes, ensuring 100% offline verification and provenance tracking.

---

## 1. Corpus Inventory & Provenance

| Document ID | Official Number | Title | Issuer | Pages | Byte Size | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| `hanoi-master-plan-100y` | `2512/QĐ-UBND` | Phê duyệt Quy hoạch tổng thể Thủ đô Hà Nội tầm nhìn 100 năm | UBND TP Hà Nội | 80 | 2,218,758 | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| `luat-112-2025-qh15` | `112/2025/QH15` | Luật Quy hoạch | Quốc hội | 52 | 1,523,362 | `8d9208d6105c96a44547f1232af9f2127ba0b4c2661ee08acb7d04573f342e8b` |
| `vbhn-103-2026-quy-hoach-tong-the` | `103/VBHN-VPQH` | Hợp nhất Nghị quyết về Quy hoạch tổng thể quốc gia thời kỳ 2021-2030, tầm nhìn 2050 | Văn phòng Quốc hội | 48 | 1,169,934 | `342b4f13dbd6eb644c9b73201d3951d85968cb316656969ace0b74a50a3ae109` |
| `tt-04-2026-bxd-pl2-dinh-muc` | `04/2026/TT-BXD/PL2` | Định mức dự toán bảo dưỡng công trình đường sắt cầu Thăng Long và tuyến Bắc Hồng - Văn Điển | Bộ Xây dựng | 19 | 345,504 | `f1725709e9df7135c729b2fcc8905638bdd6312af8656459a7475a8c504c0b25` |
| `tt-04-2023-bkhdt-so-do-ban-do` | `04/2023/TT-BKHĐT` | Yêu cầu nội dung và kỹ thuật cơ sở dữ liệu hồ sơ quy hoạch và sơ đồ, bản đồ quy hoạch | Bộ Kế hoạch và Đầu tư | 47 | 2,024,373 | `196c79b3399e78787528a513ef6efd6d659f5e083845cd43d6b435b920c51f1b` |
| `qd-23-2008-ubnd-hanoi-vien-quy-hoach` | `23/2008/QĐ-UBND` | Thành lập Viện Quy hoạch Xây dựng Hà Nội | UBND TP Hà Nội | 4 | 140,726 | `2c6276e8d611a827a220abe1106008851f4a049495f80277a62332a0c0d119fc` |

---

## 2. Multimodal & Legal Phenomena Coverage

The corpus systematically exercises the complete range of complex document phenomena found in Vietnamese public administration and territorial planning:

1. **Hierarchical Statutory Legal Text (`Luật Quy hoạch 112/2025/QH15`)**:
   - Strict hierarchical nesting: Phần -> Chương -> Mục -> Điều -> Khoản -> Điểm.
   - Preamble, enacting formula, authenticating national seal and signature of the National Assembly Chairman.

2. **Cross-Document Consolidated Planning Resolution (`VBHN 103/VBHN-VPQH`)**:
   - Complex legal consolidation reconciling national planning amendments.
   - Multi-part socioeconomic orientations, sector-specific directives, and multi-tier lists.

3. **Dense Multi-Column Engineering & Maintenance Norms (`Thông tư 04/2026/TT-BXD Phụ lục II`)**:
   - Multi-level table headers with merged spanning columns.
   - Numeric tabular data (labor codes, machine shift units, component ratios) exercising spatial table recovery.

4. **Spatial Diagrams & Map Symbology Graphics (`Thông tư 04/2023/TT-BKHĐT Phụ lục II`)**:
   - Contains embedded raster figures illustrating standard map frames and layout composition (Page 21).
   - Vector + raster sheets detailing official map symbology, point markers, and line styles (Pages 26-31).

5. **Proven 100% Scanned / OCR-Dependent Case (`Quyết định 23/2008/QĐ-UBND`)**:
   - Zero native digital text (`text_len = 0` across all pages).
   - Pure scanned raster pages bearing authentic historical red seal stamps and signatures, serving as an empirical stress test for OCR vs fast-no-ocr pipelines.

6. **End-to-End Domain Cohesion & Citations**:
   - All 6 corpus members cross-reference each other within the Vietnamese planning framework: the Law on Planning empowers the National Spatial Plan, which relies on the MPI Mapping Standards, which guides the Hanoi Capital Master Plan, drafted by the Hanoi Urban Planning Institute, whose infrastructure connectivity is maintained via the Thăng Long bridge railway norms.

---

## 3. Rejected Candidates Log

During corpus construction, candidate documents were rigorously screened and rejected if they violated integrity, licensing, or structural requirements:

| Candidate Rejected | Source URL | Rejection Rationale |
| :--- | :--- | :--- |
| `Thông tư 22/2026/TT-BTC` (`22-btc.pdf`) | `datafiles.chinhphu.vn` | Excessive file size (45.5 MB, 138 pages) composed entirely of scanned pages; rejected in favor of the cleaner, compact 4-page scanned reference (`qd_23_2008`) and born-digital mapping circular (`tt_04_2023`). |
| `04-bkhdt.signed.pdf` | `datafiles.chinhphu.vn` | Scanned raster version of Circular 04/2023 without text layer; rejected in favor of the official Công báo born-digital gazette edition (`45667-1-2023815-81604-2023-tt-bkhdt.pdf`) which contains both vector text and embedded layout diagrams. |
| `55-vbhn-bnnmt.pdf` | `datafiles.chinhphu.vn` | Long text-only document (103 pages) lacking visual diagrams or tables; redundant with VBHN 103. |
| Third-party Legal Portals | Commercial SEO sites | Strictly rejected per project governance rules requiring primary official government endpoints (`congbao.chinhphu.vn`, `vanban.chinhphu.vn`, `hanoi.gov.vn`). |

---

## 4. Architectural Boundary: 1-to-1 Manifest Schema Integrity

The Golden Document Registry enforces a strict 1-to-1 relationship between `DocumentManifest` and `FileArtifact`. When administrative decrees or circulars are published with separate annex PDFs (e.g. `Thông tư 04/2026/TT-BXD Phụ lục II`), they are registered as distinct, versioned manifests (`tt_04_2026_bxd_pl2_dinh_muc.v1.yaml`) with explicit parent document references in metadata, preserving atomic provenance verification without mutating the core registry schema.
