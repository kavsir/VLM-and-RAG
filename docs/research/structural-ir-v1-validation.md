# Structural IR v1 retained-corpus validation

> Generated from `data/benchmarks/structural_ir_v1_validation.v1.json`. Do not edit measured values by hand.

## Scope

This report evaluates deterministic Vietnamese legal/planning structural marker detection, hierarchy reconstruction, and exact physical anchoring. It does not evaluate legal meaning, entities, relations, retrieval, or VLM/LLM behavior.

## Structural schema

Structural IR wire version 1 binds to the exact deterministic Physical IR v1 (wire version 2) bytes. Nodes are strict, immutable, parser-independent records with deterministic IDs, canonical paths, recognition evidence, and exact physical anchors.

## Reference annotation policy

Reference v3/schema 3 is an AI visual PDF structural re-audit of 46 pages across 6 PDFs and 187 scored reference instances over 173 canonical paths. Prior Physical IR and Structural extractor exposure is disclosed; neither Physical IR nor extractor output was used as reference truth.

The machine-readable re-audit log is `data/structural_annotations/reference_structural_audit.v3.json` (SHA-256 `8d8b97f8cd2564895293462ed43ad302334930f692bd89d47203e49f18e01e24`), with 30 retained point-ordinal corrections and 1 genuine node restored after v2 incorrectly removed it to accommodate extractor limitations.

## Evaluation methodology

The combined result is a parser-representation-weighted aggregate over 248 scored parser/reference instances. Documents represented by both Marker and MinerU contribute twice; this is not unique-corpus accuracy. Duplicate page/kind/ordinal groups match only through uniquely resolvable exact parent paths, and unresolved groups remain unmatched.

## Legal marker rules

Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Roman conversion and raw-to-key validation are strict and kind-aware at extractor and schema boundaries; Vietnamese POINT letters remain letters. TOC leader events and line-start prose citations are suppressed before canonical-key reservation while their text remains BODY, independently of parser TITLE classification.

## Planning generic rules

Heading-like Roman and decimal outlines become GENERIC_SECTION nodes conservatively. Within an evidenced non-legal appendix outline, decimal and letter children may become generic sections and close incompatible stale ARTICLE/CLAUSE state. Outside compatible generic or legal context, letter lists remain body text.

## Content anchoring coverage

All 4067 non-empty CONTENT blocks were exactly partitioned for all 10 parser/document pairs. All 342 empty CONTENT objects had one structural owner.

## Per-document/per-parser results

| Document | Parser | Pages | Nodes | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| hanoi-master-plan-100y | marker | 80 | 46 | 250452 | `72aaf335805e8bc55dff3491dccbf899d5b60e1f53d99a2a817ac2f01df8ad70` |
| hanoi-master-plan-100y | mineru | 80 | 47 | 245069 | `9c31c9c3e41d2de4b4f2e5c4eb1d6817c7b1e480bdade19a617ecb4980141d42` |
| luat-112-2025-qh15 | marker | 52 | 664 | 687760 | `76d42b58e1aa487fcba19218889e8b8f98210cea178cfdf76ff24b78679df59f` |
| vbhn-103-2026-quy-hoach-tong-the | marker | 48 | 100 | 162667 | `5bec975c44ccdd1b02024b0540c04e82768b7d9ff996f9d563fa4ecb73ec300c` |
| tt-04-2026-bxd-pl2-dinh-muc | marker | 19 | 10 | 267367 | `fbfb13e3dc1cf60f6faaa55dfd04bc5278486cd5a52f615f8098ff11a76cf654` |
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 10 | 48165 | `e0bfce12273846b0505c04673eb343b24602ad0042db17a3994ce48a25283512` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 196 | 367425 | `1884d0b2a5179b55db6eebbe899f65bf9cc806059c1539c6214282f58b3b2112` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 203 | 239313 | `26a5af2c701abb85c6f3b86e8dd48af50b33f19124165dd51f8e5a033d6de9b6` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 1 | 15785 | `1be44c94130821198f3e224375d4a30937316cbb5f714907d0673dec407f95ee` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 18 | 31178 | `607dc5f31beead9c2ca754b86768c8604e3b58730810af2718a649c0285c5ce8` |

## Per-parser aggregates

| Parser | Documents | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | Path exact | Title exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| marker | 6 | 0.858 | 0.743 | 0.797 | 0.815 | 0.706 | 0.756 | 0.950 | 0.731 |
| mineru | 4 | 0.576 | 0.623 | 0.598 | 0.576 | 0.623 | 0.598 | 1.000 | 0.727 |

## Combined parser-representation-weighted per-kind metrics

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| part | 0 | 0 | 0 | N/A | N/A | N/A |
| chapter | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| section | 1 | 0 | 1 | 1.000 | 0.500 | 0.667 |
| subsection | 0 | 0 | 0 | N/A | N/A | N/A |
| article | 24 | 0 | 9 | 1.000 | 0.727 | 0.842 |
| clause | 64 | 4 | 16 | 0.941 | 0.800 | 0.865 |
| point | 68 | 8 | 42 | 0.895 | 0.618 | 0.731 |
| appendix | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| generic_section | 13 | 39 | 3 | 0.250 | 0.812 | 0.382 |

Combined node detection: precision 0.776, recall 0.714, F1 0.744 (177 TP / 51 FP / 71 FN).

## Hierarchy metrics

Full in-scope parent edges, including document-root edges: precision 0.746, recall 0.685, F1 0.714 (170 TP / 58 FP / 78 FN). Unmatched predictions are FP edges, unmatched references are FN edges, and a wrong parent contributes one FP plus one FN.

Canonical-path exact rate: 0.960 (170/177 matched nodes).

Whitespace/case-normalized exact title rate: 0.730 (27/37). No semantic title similarity is used.

## Cross-parser agreement

Agreement is consistency, not reference accuracy.

| Document | Legal Jaccard | Generic Jaccard | All Jaccard |
|---|---:|---:|---:|
| hanoi-master-plan-100y | 0.969 | 0.750 | 0.896 |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | N/A | 0.000 | 0.000 |
| tt-04-2023-bkhdt-so-do-ban-do | 0.922 | 0.961 | 0.946 |
| tt-04-2026-bxd-pl2-dinh-muc | 0.667 | 1.000 | 0.800 |

TT04/2023 legal agreement is 0.922: 71 shared, 1 Marker-only, and 5 MinerU-only paths. Generic agreement is 0.961: 122 shared, 1 Marker-only, and 4 MinerU-only paths. The prior 0.234 legal collapse came from stale Article 14 / Clause 13 state after rejected Appendix headings, not suffix removal; recognizing the visually present appendix boundary restores compatible hierarchy without targeting a score.

## False-positive analysis

The machine artifact contains all 51 unmatched predicted identities with page, parser, path, excerpt, physical kind, and deterministic unmatched reason. This is status evidence, not a claimed root-cause classification. TOC and line-start prose suppression occur before path reservation.

## False-negative analysis

The machine artifact contains all 71 unmatched reference identities. Causes remain `unresolved` unless directly evidenced; the report does not infer OCR or scan causation per record.

## Ambiguous cases

Rejected candidate counts: `{"letter_item_outside_legal_clause": 41, "marker_like_text_rejected": 12, "simple_decimal_without_heading_evidence": 50}`.

Corpus TOC audit rejected 15 marker-shaped leader events across 2 parser/page instances before key reservation; no page is suppressed wholesale. Line-start prose audit rejected 4 candidates. Duplicate-key rejections changed from 237 before the stack correction to 43 after it; unclassified remainder stays `unresolved`. Outside-clause letter items left as BODY total 41.

Canonical occurrence suffixes after regeneration: 0. Duplicate structural keys are rejected to BODY with a machine diagnostic.

## Limitations

Reference v2 was invalidated because it removed a visually genuine repeated QD23 Article 2 to fit extractor capability. Reference v3 restores that PDF-based instance with a parser-independent reference instance ID; if unsupported it remains a false negative. The reference is a partial-page AI visual re-audit, not human ground truth, and prior system exposure is disclosed. Combined metrics are representation-weighted. Matching has no fuzzy recovery. APPENDIX is terminal in profile v1. Structural IR v1 still rejects indistinguishable duplicate canonical keys rather than inventing occurrence suffixes.

## Evidence for #009 Selective VLM

Observed unresolved categories include scanned pages without usable Physical IR text, visually clear structure hidden in tables, OCR-corrupted/split markers, and ambiguous generic numbering. These are candidates for later selective escalation; no VLM behavior is implemented here.
