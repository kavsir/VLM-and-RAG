# Semantic IR v1 + Selective VLM validation

This evaluates ten parser representations over the exact fixed 46-page #008 audit set. Reference v2/schema 2 was rebuilt from a visual review of rendered source-PDF pages. It is AI-authored, discloses prior Physical IR, Structural IR, and Semantic extractor exposure, and is not human, blind, or independent ground truth. Runtime extractor output is never loaded as reference truth.

## Reference provenance

Semantic reference v2/schema 2 covers six documents and 46 fixed pages. Each occurrence has its own audited statement/mention identity based on the source PDF, page, excerpt occurrence, and character span—never parser block IDs or runtime Semantic IDs. Candidate suggestions were visible during audit, so metrics remain diagnostic rather than an independent estimate of real-world accuracy. Counts by kind: `{"document_identifier": 90, "legal_reference": 47, "organization_or_authority": 130, "quantity": 29, "temporal_expression": 89}`.

## Text-only baseline

- TEXT_ONLY mention P/R/F1: 0.9272030651340997 / 0.7768860353130016 / 0.8454148471615721.
- Exact evidence-span rate: 0.024793388429752067.
- Normalized-value exact rate: 1.0.
- Legal-reference component exact rate: 1.0.
- Statement exact-evidence coverage: 0.04585152838427948.

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| legal_reference | 23 | 0 | 34 | 1.0 | 0.40350877192982454 | 0.575 |
| document_identifier | 126 | 4 | 8 | 0.9692307692307692 | 0.9402985074626866 | 0.9545454545454547 |
| temporal_expression | 144 | 13 | 13 | 0.9171974522292994 | 0.9171974522292994 | 0.9171974522292994 |
| quantity | 50 | 1 | 8 | 0.9803921568627451 | 0.8620689655172413 | 0.9174311926605505 |
| organization_or_authority | 141 | 20 | 76 | 0.8757763975155279 | 0.6497695852534562 | 0.746031746031746 |

## Selector and A/B ablation

- Selector audited-region/task-family P/R/F1: 0.044642857142857144 / 0.1724137931034483 / 0.07092198581560284.
- Selected requests: 112; eligible requests: 1899; requests/representation: 11.2; requests/source document across retained representations: 18.666666666666668.
- Matching is deterministic and one-to-one, requiring the same page and task family plus IoU >= 0.25 or audited-reference containment >= 0.50.
- Eligible selection reasons: `{"empty_or_text_deficient_visual_region": 6, "ocr_corruption_signal": 311, "table_missing_structure": 25, "unknown_text_extraction": 1557}`; selected reasons: `{"empty_or_text_deficient_visual_region": 6, "ocr_corruption_signal": 76, "table_missing_structure": 23, "unknown_text_extraction": 7}`; non-selection reasons: `{"document_budget_exhausted": 1406, "page_budget_exhausted": 151, "unsupported_visual_source": 230}`; budget exhaustion: `{"document_budget_exhausted": 1406, "page_budget_exhausted": 151}`.
- TEXT_ONLY F1: 0.8454148471615721.
- SELECTIVE_VLM F1: 0.8454148471615721; executed requests: 0; F1 delta: 0.0.
- ALL_ELIGIBLE_VLM: N/A.
- Zero-request equality is an engineering baseline, not evidence that a model improves quality.

## Parser representations and consistency

- Marker P/R/F1: 0.9166666666666666 / 0.7714285714285715 / 0.8377997179125529.
- MinerU P/R/F1: 0.9444444444444444 / 0.7857142857142857 / 0.8577981651376145.
- Combined results are parser-representation-weighted: six source documents produce ten representations.

Cross-parser mention/path/value Jaccard consistency:

- `hanoi-master-plan-100y`: 0.3786764705882353 (intersection 103, union 272).
- `qd-23-2008-ubnd-hanoi-vien-quy-hoach`: 0.0 (intersection 0, union 11).
- `tt-04-2023-bkhdt-so-do-ban-do`: 0.7608695652173914 (intersection 35, union 46).
- `tt-04-2026-bxd-pl2-dinh-muc`: 0.625 (intersection 10, union 16).

## Failure and efficiency trace

- Semantic false positives: 38; semantic false negatives: 139. Machine records retain document, parser, page, kind, raw evidence, normalized value, and `unresolved` cause.
- VLM selector misses: 24; other VLM failures: zero because no corpus requests were executed.
- Transcription exact match/edit distance, model ID, endpoint, temperature, image bytes/dimensions, latency, failed requests, and tokens: N/A.

## Reproducibility

- Semantic determinism: 10/10 byte-identical.
- Structural and Physical freeze checks: passed.

## Real VLM experiment

**REAL VLM QUALITY EXPERIMENT NOT COMPLETED.** No model ID, endpoint, latency, or model-derived corpus outputs are claimed. Replay/contract tests prove offline engineering behavior only. A real model execution is required before claiming improvement.

## Limitations

Statements are exact direct-content spans, not paraphrases. Mentions use a controlled deterministic Vietnamese taxonomy. Detection matches occurrences one-to-one by page, kind, and raw text; normalized and legal-component correctness are measured only after raw occurrence matching. Failures remain `unresolved` unless evidence establishes a cause. Parser aggregates are representation-weighted: six source documents produce ten parser representations. Selector labels are audited PDF regions with task families, but remain AI-authored diagnostic evidence. No RAG, KG, cross-document citation resolution, or entity resolution is implemented.
