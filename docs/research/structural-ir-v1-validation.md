# Structural IR v1 retained-corpus validation

> Generated from `data/benchmarks/structural_ir_v1_validation.v1.json`. Do not edit measured values by hand.

## Scope

This report evaluates deterministic Vietnamese legal/planning structural marker detection, hierarchy reconstruction, and exact physical anchoring. It does not evaluate legal meaning, entities, relations, retrieval, or VLM/LLM behavior.

## Structural schema

Structural IR wire version 1 binds to the exact deterministic Physical IR v1 (wire version 2) bytes. Nodes are strict, immutable, parser-independent records with deterministic IDs, canonical paths, recognition evidence, and exact physical anchors.

## Reference annotation policy

The v1 AI visual structural reference contains 46 pages across 6 PDFs and 188 scored nodes. The annotator had prior Physical IR exposure, but reference structure was established from rendered PDFs and contains no parser block identity.

## Legal marker rules

Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Arabic clauses require an active ARTICLE; letter points require an active CLAUSE. Embedded cross-references are not headings.

## Planning generic rules

Heading-like Roman and decimal outlines become GENERIC_SECTION nodes conservatively. A simple decimal outside legal context requires TITLE or strong heading evidence; outside-clause letter lists remain body text.

## Content anchoring coverage

All 4067 non-empty CONTENT blocks were exactly partitioned for all 10 parser/document pairs. All 342 empty CONTENT objects had one structural owner.

## Per-document/per-parser results

| Document | Parser | Pages | Nodes | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| hanoi-master-plan-100y | marker | 80 | 62 | 262110 | `cc0dce4d8e73794ef95b13c261fff7cbc03a771916694930166c1db3252a1ba3` |
| hanoi-master-plan-100y | mineru | 80 | 65 | 258217 | `2c60c01e85a83550d2266571badbcc322e38a1bc24c24f11496d135bd75f490a` |
| luat-112-2025-qh15 | marker | 52 | 676 | 696142 | `69d292e70d3daf1b955ce2169ac4fef7adfb4ac81e3e64840c63bb541612c746` |
| vbhn-103-2026-quy-hoach-tong-the | marker | 48 | 105 | 166884 | `df29eddd4059bbdc8c2fee58bcbe557312155f0ddeee58d6a52292c231b2bde9` |
| tt-04-2026-bxd-pl2-dinh-muc | marker | 19 | 19 | 274204 | `fb2487b26834d8e84638857147c3e70ac49881d7cec625b9150864efa986fd7f` |
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 20 | 55895 | `05aec77d1f17a1b425240b382d01cd97214a06ba973170cc5a20c1ba92957f7d` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 96 | 264853 | `ae3c6f8f05d6e2f42ce4200fe85ebe81addc0c10e18f7df88df5d212102e30a7` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 99 | 133035 | `b5a2d4f73ad78774e0ce368a1495cf04c7b99960b3b5ccb6ed3c3412c0f0505e` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 1 | 15785 | `1be44c94130821198f3e224375d4a30937316cbb5f714907d0673dec407f95ee` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 4 | 18763 | `255b2c1fcdb7e626106722dc51c03c444eae55781a398c47f1cb308b20882400` |

## Per-kind metrics

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| part | 0 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| chapter | 3 | 3 | 0 | 0.500 | 1.000 | 0.667 |
| section | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| subsection | 0 | 0 | 0 | 0.000 | 0.000 | 0.000 |
| article | 24 | 0 | 9 | 1.000 | 0.727 | 0.842 |
| clause | 71 | 0 | 10 | 1.000 | 0.877 | 0.934 |
| point | 78 | 0 | 32 | 1.000 | 0.709 | 0.830 |
| appendix | 4 | 1 | 0 | 0.800 | 1.000 | 0.889 |
| generic_section | 14 | 9 | 2 | 0.609 | 0.875 | 0.718 |

## Hierarchy metrics

Parent edges: precision 0.850, recall 0.680, F1 0.756 (153 TP / 27 FP / 72 FN).

Canonical-path exact rate: 0.862 (169/196 matched nodes).

Whitespace/case-normalized exact title rate: 0.737 (28/38). No semantic title similarity is used.

## Cross-parser agreement

Agreement is consistency, not reference accuracy.

| Document | Legal Jaccard | Generic Jaccard | All Jaccard |
|---|---:|---:|---:|
| hanoi-master-plan-100y | 0.957 | 0.842 | 0.923 |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | 1.000 | 0.000 | 0.000 |
| tt-04-2023-bkhdt-so-do-ban-do | 0.922 | 0.957 | 0.930 |
| tt-04-2026-bxd-pl2-dinh-muc | 0.667 | 0.500 | 0.609 |

## False-positive analysis

False positives are audited-page predictions that fail exact parser-neutral marker occurrence matching. The main risks are table-of-contents entries and short numbered body lines represented as headings by a parser.

## False-negative analysis

False negatives arise when physical text is absent (especially scans), marker text is fragmented/corrupted, or conservative heading evidence intentionally rejects ambiguous numbering.

## Ambiguous cases

Rejected candidate counts: `{"letter_item_outside_legal_clause": 173, "marker_like_text_rejected": 10, "simple_decimal_without_heading_evidence": 103}`.

## Limitations

The reference is partial-page AI visual audit, not human ground truth. Metrics are repeated per available parser representation. Exact ordinal occurrence matching intentionally has no fuzzy recovery. Title continuation remains conservative.

## Evidence for #009 Selective VLM

Observed unresolved categories include scanned pages without usable Physical IR text, visually clear structure hidden in tables, OCR-corrupted/split markers, and ambiguous generic numbering. These are candidates for later selective escalation; no VLM behavior is implemented here.
