# Physical Document IR v0 — Hanoi Baseline Normalization Report

This research note records the first normalization pass converting the real MinerU 3.4.5 parse of the
Hanoi golden document (`hanoi-master-plan-100y`, `v1`) into the parser-independent Physical Document IR v0.

## Normalization Metrics

| Field | Observed / Normalized Value |
| :--- | :--- |
| Document / Version ID | `hanoi-master-plan-100y` / `v1` |
| Authoritative Input SHA-256 | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| Parser Runtime | MinerU 3.4.5 (`pipeline` backend) |
| Pages Normalized | 80 (contiguous indexes `0` through `79`) |
| Page Dimensions | Preserved from `middle.json` (`595.0` pt × `841.0` pt, A4 at 72 DPI) |
| Total Observed Raw Blocks | 1,186 |
| Total Normalized Physical Blocks | 1,186 |
| Dropped Blocks | 0 (all raw blocks preserved) |
| Normalization Runtime | ~0.005 – 0.010 seconds (5 – 10 ms) |
| Serialized IR Byte Size | 1,173,108 bytes |
| Serialized IR SHA-256 | `40c461e625c51c728db9c3e2f6e7ab8781deb20652cea39e371716c18dfef1e2` |
| Repeat-Run SHA-256 | `40c461e625c51c728db9c3e2f6e7ab8781deb20652cea39e371716c18dfef1e2` |
| Determinism Verdict | **Exact match (100% deterministic)** |

## Block Counts & Categorization

| Raw MinerU Structure | Physical IR Kind | Physical IR Disposition | Count |
| :--- | :--- | :--- | ---: |
| `type: "text"` (no `text_level`) | `BlockKind.TEXT` | `BlockDisposition.CONTENT` | 982 |
| `type: "text"` (`text_level: 1`) | `BlockKind.TITLE` (level 1) | `BlockDisposition.CONTENT` | 1 |
| `type: "text"` (`text_level: 2`) | `BlockKind.TITLE` (level 2) | `BlockDisposition.CONTENT` | 122 |
| `type: "header"` | `BlockKind.HEADER` | `BlockDisposition.DISCARDED` | 2 |
| `type: "page_number"` | `BlockKind.PAGE_NUMBER` | `BlockDisposition.DISCARDED` | 79 |
| **Total** | | | **1,186** |

## Representation Mapping Differences

1. **Heading Differentiation**:
   In raw MinerU `content_list.json`, both paragraphs and headings are labeled `"type": "text"`, with
   headings distinguished solely by the presence of an optional `"text_level"` key. Physical IR v0
   explicitly partitions these into `BlockKind.TEXT` (where `heading_level` must be `None`) and
   `BlockKind.TITLE` (with 1-based positive integer `heading_level`).
2. **Structural Preservation**:
   Headers (2 blocks) and page numbers (79 blocks) are not deleted or dropped during normalization. They
   are preserved as `BlockKind.HEADER` and `BlockKind.PAGE_NUMBER` with `BlockDisposition.DISCARDED`,
   allowing complete traceability of physical page geometry while signaling downstream retrieval pipelines
   to ignore non-content text.
3. **Coordinate Semantics**:
   Coordinates from `content_list.json` are in MinerU's native `0..1000` normalized layout space. They
   are stored as `BoundingBox(x0, y0, x1, y1, coordinate_system="normalized_1000")`. Physical page
   dimensions (`width: 595.0`, `height: 841.0` in PDF points) are incorporated from `middle.json` at
   the `PhysicalPage` level.
4. **Deterministic Block Identifiers**:
   Blocks receive immutable IDs formatted as `{document_id}_{version_id}_p{page_index:04d}_b{reading_order:04d}`
   (e.g., `hanoi-master-plan-100y_v1_p0000_b0000`), ensuring repeatable references across runs.

## Unresolved Representation Ambiguities & Scope Limitations

* **Zero Multimodal Elements**: As established in Issue #003, the Hanoi baseline legal decree contains
  0 tables, 0 figures/images, 0 charts, and 0 formulas. Consequently, multimodal block types (`TABLE`,
  `IMAGE`, etc.) remain unvalidated in v0.
* **Granularity**: MinerU `content_list.json` operates at the block level and does not export token- or
  span-level bounding boxes in the content list. Line and span coordinates exist in `middle.json` but
  without `text_level`. Block-level granularity is sufficient and appropriate for Physical IR v0.
