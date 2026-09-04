# Vietnamese Legal, Administrative, and Planning Multimodal Golden Corpus (v1)

## Executive Summary

Issue #006 establishes an authoritative, multimodal golden corpus curated for benchmarking document parsing and layout extraction on Vietnamese legal, administrative, and spatial planning documents.

The corpus consists of **6 verified documents** (250 total pages) sourced exclusively from official government portals (`congbao.chinhphu.vn`, `vanban.chinhphu.vn`). Every document is immutably registered in `data/manifests/` with cryptographic SHA-256 digests and exact byte sizes, ensuring 100% offline verification and provenance tracking.

---

## 1. Corpus Inventory & Provenance

| Document ID | Official Number | Title | Issuer | Issued Date | Effective Date | Signer | Pages | Byte Size | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :---: | :---: | :--- |
| `hanoi-master-plan-100y` | `2512/QĐ-UBND` | Phê duyệt Quy hoạch tổng thể Thủ đô Hà Nội tầm nhìn 100 năm | Ủy ban nhân dân thành phố Hà Nội | 2026-05-13 | 2026-05-13 | Vũ Đại Thắng | **80** | 2,218,758 | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| `luat-112-2025-qh15` | `112/2025/QH15` | Luật Quy hoạch | Quốc hội | 2025-12-10 | 2026-03-01 | Trần Thanh Mẫn | **52** | 1,523,362 | `8d9208d6105c96a44547f1232af9f2127ba0b4c2661ee08acb7d04573f342e8b` |
| `vbhn-103-2026-quy-hoach-tong-the` | `103/VBHN-VPQH` | Văn bản hợp nhất Nghị quyết về Quy hoạch tổng thể quốc gia thời kỳ 2021-2030, tầm nhìn đến năm 2050 | Văn phòng Quốc hội | 2026-04-22 | N/A* | Bùi Văn Cường | **48** | 1,169,934 | `342b4f13dbd6eb644c9b73201d3951d85968cb316656969ace0b74a50a3ae109` |
| `tt-04-2026-bxd-pl2-dinh-muc` | `04/2026/TT-BXD/PL2` | Định mức dự toán bảo dưỡng công trình đường sắt cầu Thăng Long và tuyến Bắc Hồng - Văn Điển | Bộ Xây dựng | 2026-01-30 | 2026-02-01 | Nguyễn Danh Huy | **19** | 345,504 | `f1725709e9df7135c729b2fcc8905638bdd6312af8656459a7475a8c504c0b25` |
| `tt-04-2023-bkhdt-so-do-ban-do` | `04/2023/TT-BKHĐT` | Hướng dẫn yêu cầu nội dung và kỹ thuật cơ sở dữ liệu hồ sơ quy hoạch và sơ đồ, bản đồ quy hoạch cấp quốc gia, quy hoạch vùng, quy hoạch tỉnh | Bộ Kế hoạch và Đầu tư | 2023-06-26 | 2023-08-10 | Trần Quốc Phương | **47** | 2,024,373 | `196c79b3399e78787528a513ef6efd6d659f5e083845cd43d6b435b920c51f1b` |
| `qd-23-2008-ubnd-hanoi-vien-quy-hoach` | `23/2008/QĐ-UBND` | Thành lập Viện Quy hoạch Xây dựng Hà Nội | Ủy ban nhân dân thành phố Hà Nội | 2008-09-29 | 2008-10-09 | Nguyễn Thế Thảo | **4** | 140,726 | `2c6276e8d611a827a220abe1106008851f4a049495f80277a62332a0c0d119fc` |

*\* Note: For consolidated texts (`VBHN 103/VBHN-VPQH`), official Vietnamese jurisprudence does not assign an independent effective date to the consolidated text itself; effective dates derive from the underlying amended enactments. The manifest truthfully records effective_on as absent.*

---

## 2. Multimodal & Legal Phenomena Coverage

The corpus systematically exercises the complete range of complex document phenomena found in Vietnamese public administration and territorial planning:

1. **Hierarchical Statutory Legal Text (`Luật Quy hoạch 112/2025/QH15`)**:
   - Structural hierarchy: `Chương` -> `Mục` -> `Điều` -> `Khoản` -> `Điểm` (no `Phần` division in this statute).
   - Preamble, enacting formula, authenticating national seal and signature of the National Assembly Chairman (Trần Thanh Mẫn).

2. **Consolidated Planning Resolution (`VBHN 103/VBHN-VPQH`)**:
   - Complex legal consolidation reconciling national master planning directives.
   - Multi-part socioeconomic orientations, sector-specific directives, and multi-tier lists.

3. **Dense Multi-Column Engineering & Maintenance Norms (`Thông tư 04/2026/TT-BXD Phụ lục II`)**:
   - Multi-level table headers with merged spanning columns.
   - Numeric tabular data (labor codes, machine shift units, component ratios) exercising spatial table recovery.
   - *Relationship Note*: The registered artifact is Annex II (`Phụ lục II: Định mức dự toán bảo dưỡng công trình đường sắt cầu Thăng Long...`) attached to Circular 04/2026/TT-BXD. Registry v1 schema does not encode parent-document links explicitly; this relationship is documented here.

4. **Spatial Diagrams & Map Symbology Graphics (`Thông tư 04/2023/TT-BKHĐT Phụ lục II`)**:
   - Embedded raster figures illustrating standard map frames and layout composition on **PDF page 21 (page_index=20)**.
   - Vector and raster sheets detailing official map symbology, point markers, and line styles on **PDF pages 26-31 (page_index=25 to 30)**.

5. **100% Scanned / OCR-Dependent Case (`Quyết định 23/2008/QĐ-UBND`)**:
   - Zero native digital text (`text_len = 0` across all 4 pages).
   - Scanned raster pages bearing authentic historical red seal stamps and signatures, serving as an empirical test for Page Modality Classification and OCR recovery.

6. **Domain & Thematic Cohesion**:
   - *Direct legal cross-references*: No direct intra-corpus citation edge was established among these specific document members in v1.
   - *Thematic cohesion*: The members represent interconnected aspects of the Vietnamese territorial planning ecosystem: statutory planning frameworks (Law 112), national spatial planning resolutions (VBHN 103), technical mapping and database specifications (TT 04/2023), urban capital planning (Hanoi Master Plan), infrastructure maintenance norms (TT 04/2026 PL2), and urban planning institutional mandates (QD 23/2008).

---

## 3. Reference Annotation Quality & Traceability

A total of **46 pages** were audited, comprising **112 layout reference regions**:

- **Annotation Version**: `v2`
- **Annotator**: `antigravity`
- **Annotator Type**: `ai_visual_audit`
- **Method**: `independent_visual_pdf_audit`
- **Protocol**: Reference annotations were created by inspecting rendered PDF page canvases and frozen prior to running parser evaluations (`parser_output_used_as_ground_truth: false`).
- **Scanned Pages**: On `qd_23_2008` (4 pages), pages are labeled with `page_modality: "scanned_raster"` under the Page Modality Task without synthetic OCR transcription boxes.

---

## 4. Rejected Candidates Log

| Candidate Considered | Source | Rationale for Exclusion |
| :--- | :--- | :--- |
| `Thông tư 22/2026/TT-BTC` (`22-btc.pdf`) | `datafiles.chinhphu.vn` | Large file (45.5 MB, 138 pages) composed of repetitive scanned tables; excluded in favor of compact 4-page scanned reference (`qd_23_2008`) and born-digital norm tables (`tt_04_2026`). |
| `04-bkhdt.signed.pdf` | `datafiles.chinhphu.vn` | Scanned raster version without text layer; excluded in favor of official Công báo born-digital gazette edition containing vector text and layout diagrams (`45667-1-2023815-81604-2023-tt-bkhdt.pdf`). |
| `55-vbhn-bnnmt.pdf` | `datafiles.chinhphu.vn` | Text-only document (103 pages) lacking visual diagrams or tables; redundant with VBHN 103. |
