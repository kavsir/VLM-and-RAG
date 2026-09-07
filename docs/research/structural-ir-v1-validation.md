# Structural IR v1 retained-corpus validation

> Generated from `data/benchmarks/structural_ir_v1_validation.v1.json`. Do not edit measured values by hand.

## Scope

This report evaluates deterministic Vietnamese legal/planning structural marker detection, hierarchy reconstruction, and exact physical anchoring. It does not evaluate legal meaning, entities, relations, retrieval, or VLM/LLM behavior.

## Structural schema

Structural IR wire version 1 binds to the exact deterministic Physical IR v1 (wire version 2) bytes. Nodes are strict, immutable, parser-independent records with deterministic IDs, canonical paths, recognition evidence, and exact physical anchors.

## Reference annotation policy

Reference v4/schema 4 is an AI visual PDF structural re-audit of 46 pages across 6 PDFs and 187 scored structural instances over 173 canonical paths. A stable reference_instance_id identifies a visual structural instance, reference_record_id identifies each page-local annotation record, and parent_reference_instance_id binds the exact visual parent. Each parent marker must begin on or before its child marker; 188 unique non-root instance assignments satisfy this ordering invariant. Prior Physical IR and Structural extractor exposure is disclosed; neither Physical IR nor extractor output was used as reference truth.

The machine-readable re-audit log is `data/structural_annotations/reference_structural_audit.v4.json` (SHA-256 `4a3c6d1faecf9d2c0b31206a6eb6f0eac1e0ffff79dcfeb87e7cbdc61a99c68b`), with 30 retained point-ordinal corrections and 1 genuine node restored after v2 incorrectly removed it to accommodate extractor limitations.

## Evaluation methodology

The combined result is a parser-representation-weighted aggregate over 248 scored parser/reference instances. Documents represented by both Marker and MinerU contribute twice; this is not unique-corpus accuracy. Duplicate page/kind/ordinal groups match only through uniquely resolvable exact parent paths, and unresolved groups remain unmatched.

## Legal marker rules

Line-start PHẦN, CHƯƠNG, MỤC, TIỂU MỤC, ĐIỀU, and PHỤ LỤC markers take precedence. Roman conversion and raw-to-key validation are strict and kind-aware at extractor and schema boundaries; unnumbered nodes use the literal `unnumbered` final path segment and Vietnamese POINT letters remain letters. Explicit TOCs suppress only locally contiguous entries ending in dotted page leaders; a real formal body heading terminates the region. Multi-page continuation suppression remains event-scoped.

## Planning generic rules

Heading-like Roman and decimal outlines become GENERIC_SECTION nodes conservatively. Generic nodes select an exact numeric prefix, then a lower generic level, then the deepest compatible formal container. An active planning outline takes precedence over stale legal CLAUSE/POINT state; only a sequential legal clause marker re-enters legal context. Corpus-evidenced inline boundaries recover flattened formal headings and the Law 112 second-clause list with exact, gap-free source offsets.

## Content anchoring coverage

All 4067 non-empty CONTENT blocks were exactly partitioned for all 10 parser/document pairs. All 342 empty CONTENT objects had one structural owner.

## Per-document/per-parser results

| Document | Parser | Pages | Nodes | Bytes | SHA-256 |
|---|---:|---:|---:|---:|---|
| hanoi-master-plan-100y | marker | 80 | 67 | 269845 | `b52749f6635fe257a4910eb083c600f460b5692c6a8b50e09e1b9a71b411443e` |
| hanoi-master-plan-100y | mineru | 80 | 64 | 259833 | `dfe4ad4be1dd20d75eb6339548c9a0c3490251b20416251e5312d1b09ea33060` |
| luat-112-2025-qh15 | marker | 52 | 692 | 712274 | `27fa3570d6ffe12bd3960723f4fde174964496c105a4fad3d2839f77b819ab39` |
| vbhn-103-2026-quy-hoach-tong-the | marker | 48 | 100 | 162667 | `5bec975c44ccdd1b02024b0540c04e82768b7d9ff996f9d563fa4ecb73ec300c` |
| tt-04-2026-bxd-pl2-dinh-muc | marker | 19 | 10 | 267367 | `fbfb13e3dc1cf60f6faaa55dfd04bc5278486cd5a52f615f8098ff11a76cf654` |
| tt-04-2026-bxd-pl2-dinh-muc | mineru | 19 | 10 | 48165 | `e0bfce12273846b0505c04673eb343b24602ad0042db17a3994ce48a25283512` |
| tt-04-2023-bkhdt-so-do-ban-do | marker | 47 | 207 | 377262 | `e075f67f0904ec421db5ffb051323a29bd56cb90faef8cbf2136a111fd50632b` |
| tt-04-2023-bkhdt-so-do-ban-do | mineru | 47 | 210 | 245414 | `df6ef39f3548de0a526b354c357469ef187626db4c98021c2b8b601fad3bd923` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | marker | 4 | 1 | 15785 | `1be44c94130821198f3e224375d4a30937316cbb5f714907d0673dec407f95ee` |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | mineru | 4 | 18 | 31178 | `607dc5f31beead9c2ca754b86768c8604e3b58730810af2718a649c0285c5ce8` |

## Per-parser aggregates

| Parser | Documents | Node P | Node R | Node F1 | Edge P | Edge R | Edge F1 | Path exact | Title exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| marker | 6 | 0.900 | 0.818 | 0.857 | 0.847 | 0.770 | 0.807 | 0.941 | 0.704 |
| mineru | 4 | 0.530 | 0.574 | 0.551 | 0.530 | 0.574 | 0.551 | 1.000 | 0.727 |

## Combined parser-representation-weighted per-kind metrics

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| part | 0 | 0 | 0 | N/A | N/A | N/A |
| chapter | 3 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| section | 2 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| subsection | 0 | 0 | 0 | N/A | N/A | N/A |
| article | 24 | 0 | 9 | 1.000 | 0.727 | 0.842 |
| clause | 67 | 0 | 13 | 1.000 | 0.838 | 0.912 |
| point | 74 | 0 | 36 | 1.000 | 0.673 | 0.804 |
| appendix | 4 | 0 | 0 | 1.000 | 1.000 | 1.000 |
| generic_section | 14 | 48 | 2 | 0.226 | 0.875 | 0.359 |

Combined node detection: precision 0.797, recall 0.758, F1 0.777 (188 TP / 48 FP / 60 FN).

## Hierarchy metrics

Full in-scope parent edges, including document-root edges: precision 0.758, recall 0.722, F1 0.740 (179 TP / 57 FP / 69 FN). Unmatched predictions are FP edges, unmatched references are FN edges, and a wrong parent contributes one FP plus one FN.

Canonical-path exact rate: 0.952 (179/188 matched nodes).

Whitespace/case-normalized exact title rate: 0.711 (27/38); 0 indistinguishable duplicate pairings are excluded. Reference IDs never choose a metric-bearing title pairing, and no semantic title similarity is used.

## Cross-parser agreement

Agreement is consistency, not reference accuracy.

| Document | Legal Jaccard | Generic Jaccard | All Jaccard |
|---|---:|---:|---:|
| hanoi-master-plan-100y | 0.400 | 0.000 | 0.066 |
| qd-23-2008-ubnd-hanoi-vien-quy-hoach | N/A | 0.000 | 0.000 |
| tt-04-2023-bkhdt-so-do-ban-do | 0.935 | 0.985 | 0.967 |
| tt-04-2026-bxd-pl2-dinh-muc | 0.667 | 1.000 | 0.800 |

TT04/2023 legal agreement is 0.935: 72 shared, 1 Marker-only, and 4 MinerU-only paths. Generic agreement is 0.985: 132 shared, 1 Marker-only, and 1 MinerU-only paths. The prior 0.234 legal collapse came from stale Article 14 / Clause 13 state after rejected Appendix headings, not suffix removal; recognizing the visually present appendix boundary restores compatible hierarchy without targeting a score.

## False-positive analysis

The machine artifact contains all 48 unmatched predicted identities with page, parser, path, excerpt, physical kind, and deterministic unmatched reason. This is status evidence, not a claimed root-cause classification. TOC and line-start prose suppression occur before path reservation.

## False-negative analysis

The machine artifact contains all 60 unmatched reference identities. Causes remain `unresolved` unless directly evidenced; the report does not infer OCR or scan causation per record.

## Ambiguous cases

Rejected candidate counts: `{"letter_item_outside_legal_clause": 12, "marker_like_text_rejected": 12, "simple_decimal_without_heading_evidence": 40}`.

Corpus TOC audit rejected 15 marker-shaped entry events across 2 parser/page instances before key reservation; no page is suppressed wholesale. Line-start prose audit rejected 2 candidates. The committed historical audit at `data/benchmarks/structural_duplicate_audit.38d1042.v1.json` (SHA-256 `a11be0646c41804497f0b6ca31dd666b23745cab4da10e3a35bdb97df2326298`) covers all 43 diagnostics at 38d1042: 35 genuine structures are recovered and 8 quoted/nested-legislation limitations remain BODY. Current duplicate rejections total 8, with zero representable genuine structures unresolved. Outside-clause letter items left as BODY total 12.

Canonical occurrence suffixes after regeneration: 0. Duplicate structural keys are rejected to BODY with a machine diagnostic.

## Limitations

Reference v4 changes no visual truth from v3. It separates structural-instance identity from page-record identity and explicitly binds QD23 Clause 1-4 descendants to the first Article 2 instance; the second Article 2 remains a distinct scored instance and false negative when unsupported. Parent-instance validation proves only existence, kind/path agreement, acyclicity, and that a parent marker does not begin after its child. Same-page duplicated canonical parents remain visually ambiguous when no other existing non-title structural evidence distinguishes them; no geometry or reading-order inference is fabricated. Parent-edge scoring remains canonical-path based because predictions have no reference identity. The reference is a partial-page AI visual re-audit, not human ground truth. Combined metrics are representation-weighted. Matching has no fuzzy recovery. APPENDIX is terminal in profile v1. Eight quoted/nested-legislation duplicates remain a conservative v1 limitation; no occurrence suffixes are invented.

## Evidence for #009 Selective VLM

Observed unresolved categories include scanned pages without usable Physical IR text, visually clear structure hidden in tables, OCR-corrupted/split markers, and ambiguous generic numbering. These are candidates for later selective escalation; no VLM behavior is implemented here.
