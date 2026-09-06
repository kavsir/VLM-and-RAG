# Structural IR v1 retained-corpus validation

> Generated from `data/benchmarks/structural_ir_v1_validation.v1.json`. Do not edit measured values by hand.

## Scope

This report evaluates deterministic Vietnamese legal/planning structural marker detection, hierarchy reconstruction, and exact physical anchoring. It does not evaluate legal meaning, entities, relations, retrieval, or VLM/LLM behavior.

## Structural schema

Structural IR wire version 1 binds to the exact deterministic Physical IR v1 (wire version 2) bytes. Nodes are strict, immutable, parser-independent records with deterministic IDs, canonical paths, recognition evidence, and exact physical anchors.

## Reference annotation policy

Reference v2 is an AI visual PDF structural re-audit of 46 pages across 6 PDFs and 186 unique scored nodes. Prior Physical IR and Structural extractor exposure is disclosed; neither Physical IR nor extractor output was used as reference truth.

The machine-readable re-audit log is `data/structural_annotations/reference_structural_audit.v2.json` (SHA-256 `9f24eec9151d483c68eeb18980d551191dfa03abeea0f36e4e06d08336101948`), with 30 corrected point-ordinal records and 47 changed or removed v1 node records.

## Evaluation methodology

The combined result is a parser-representation-weighted aggregate over 246 scored parser/reference instances. Documents represented by both Marker and MinerU contribute twice; this is not unique-corpus accuracy. Duplicate page/kind/ordinal groups match only through uniquely resolvable exact parent paths, and unresolved groups remain unmatched.

## Legal marker rules

Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Roman conversion is strict and kind-aware; Vietnamese POINT letters remain letters. Arabic clauses require an active ARTICLE and letter points require an active CLAUSE. TOC entries and line-start prose citations are suppressed before canonical-key reservation while their text remains BODY.

## Planning generic rules

Heading-like Roman and decimal outlines become GENERIC_SECTION nodes conservatively. A simple decimal outside legal context requires TITLE or strong heading evidence; outside-clause letter lists remain body text.

## Content anchoring coverage

All 4067 non-empty CONTENT blocks were exactly partitioned for all 10 parser/document pairs. All 342 empty CONTENT objects had one structural owner.

## Per-document/per-parser results

| Document | Parser | Pages | Nodes | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| hanoi-master-plan-100y | marker | 80 | 46 | 250452 | `72aaf335805e8bc55dff3491dccbf899d5b60e1f53d99a2a817ac2f01df8ad70` |
| hanoi-master-plan-100y | mineru | 80 | 47 | 245069 | `9c31c9c3e41d2de4b4f2e5c4eb1d6817c7b1e480bdade19a617ecb4980141d42` |
| luat-112-2025-qh15 | marker | 52 | 665 | 688302 | `cdb64476d104e917e28fe6fa2fc116cbd2efc9da36a28c85aa03c0efc9241d77` |
| vbhn-103-2026-quy-hoach-tong-the | marker | 48 | 100 | 162667 | `5bec975c44ccdd1b02024b0540c04e82768b7d9ff996f9d563fa4ecb73ec300c` |
| tt-04-2026-bxd-pl2-dinh-muc | marker | 19 | 11 | 268124 | `71b6e77422c6d8830de94534e160fce15a79f73cbd4f06d488239fa7d9c98a54` |
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 11 | 48881 | `09164e0ef1edcd781298a5817c6cdf63de4b9967094511740611bfea35ee7c3f` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 98 | 264794 | `afacb72c978a20812b458314cfdf0a2dbfac795a02a6fc3bc5c21fa3d8da5a92` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 99 | 130452 | `a2bd8fb9a4d2b4eedd2080bf472bee736c33132c1454a430ff401a51468b48e1` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 1 | 15785 | `1be44c94130821198f3e224375d4a30937316cbb5f714907d0673dec407f95ee` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 4 | 18763 | `255b2c1fcdb7e626106722dc51c03c444eae55781a398c47f1cb308b20882400` |

## Per-parser aggregates

| Parser | Documents | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | Path exact | Title exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| marker | 6 | 0.886 | 0.710 | 0.788 | 0.765 | 0.613 | 0.681 | 0.864 | 0.731 |
| mineru | 4 | 0.795 | 0.517 | 0.626 | 0.462 | 0.300 | 0.364 | 0.581 | 0.727 |

## Combined parser-representation-weighted per-kind metrics

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| part | 0 | 0 | 0 | N/A | N/A | N/A |
| chapter | 2 | 0 | 1 | 1.000 | 0.667 | 0.800 |
| section | 0 | 0 | 2 | N/A | 0.000 | N/A |
| subsection | 0 | 0 | 0 | N/A | N/A | N/A |
| article | 24 | 0 | 7 | 1.000 | 0.774 | 0.873 |
| clause | 54 | 14 | 26 | 0.794 | 0.675 | 0.730 |
| point | 68 | 8 | 42 | 0.895 | 0.618 | 0.731 |
| appendix | 2 | 0 | 2 | 1.000 | 0.500 | 0.667 |
| generic_section | 13 | 3 | 3 | 0.812 | 0.812 | 0.812 |

Combined node detection: precision 0.867, recall 0.663, F1 0.751 (163 TP / 25 FP / 83 FN).

## Hierarchy metrics

Full in-scope parent edges, including document-root edges: precision 0.702, recall 0.537, F1 0.608 (132 TP / 56 FP / 114 FN). Unmatched predictions are FP edges, unmatched references are FN edges, and a wrong parent contributes one FP plus one FN.

Canonical-path exact rate: 0.810 (132/163 matched nodes).

Whitespace/case-normalized exact title rate: 0.730 (27/37). No semantic title similarity is used.

## Cross-parser agreement

Agreement is consistency, not reference accuracy.

| Document | Legal Jaccard | Generic Jaccard | All Jaccard |
|---|---:|---:|---:|
| hanoi-master-plan-100y | 0.969 | 0.750 | 0.896 |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | N/A | 0.000 | 0.000 |
| tt-04-2023-bkhdt-so-do-ban-do | 0.234 | 0.909 | 0.283 |
| tt-04-2026-bxd-pl2-dinh-muc | 1.000 | 1.000 | 1.000 |

## False-positive analysis

The machine artifact contains all 25 unmatched predicted identities with page, parser, path, excerpt, physical kind, and deterministic category. TOC and line-start prose suppression occur before path reservation.

## False-negative analysis

The machine artifact contains all 83 unmatched reference identities. Causes remain `unresolved` unless directly evidenced; the report does not infer OCR or scan causation per record.

## Ambiguous cases

Rejected candidate counts: `{"letter_item_outside_legal_clause": 186, "marker_like_text_rejected": 18, "simple_decimal_without_heading_evidence": 109}`.

Corpus TOC audit rejected 15 marker-shaped entries across 2 parser/page instances before key reservation. Line-start prose audit rejected 14 candidates. Stale/repeated context handling left 237 duplicate keys and 186 outside-clause letter items as BODY.

Canonical occurrence suffixes after regeneration: 0. Duplicate structural keys are rejected to BODY with a machine diagnostic.

## Limitations

The reference is a partial-page AI visual re-audit, not human ground truth, and prior system exposure is disclosed. Combined metrics are representation-weighted. Matching has no fuzzy recovery. APPENDIX is terminal in profile v1, and title continuation remains conservative.

## Evidence for #009 Selective VLM

Observed unresolved categories include scanned pages without usable Physical IR text, visually clear structure hidden in tables, OCR-corrupted/split markers, and ambiguous generic numbering. These are candidates for later selective escalation; no VLM behavior is implemented here.
