# Semantic IR v1 + Selective VLM validation

This evaluates ten parser representations over the exact fixed 46-page #008 audit set. The reference is AI-assisted and discloses prior Physical IR, Structural IR, and Semantic extractor exposure; it is not human, blind, independent ground truth.

## Reference provenance

Semantic reference v1/schema 1 covers six documents and 46 fixed pages. Identity uses document, PDF page, exact evidence excerpt, and character span—never parser block IDs or runtime Semantic IDs. Candidate coverage is the union of retained parser representations and is therefore exposure-biased; metrics are diagnostic, not an independent estimate of real-world accuracy. Counts by kind: `{"document_identifier": 82, "legal_reference": 21, "organization_or_authority": 53, "quantity": 20, "temporal_expression": 79}`.

## Text-only baseline

- TEXT_ONLY mention P/R/F1: 0.7375478927203065 / 0.9436274509803921 / 0.8279569892473119.
- Exact evidence-span rate: 0.9093137254901961.
- Normalized-value exact rate: 0.9436274509803921.
- Legal-reference component exact rate: 1.0.
- Statement exact-evidence coverage: 0.650917176209005.

| Kind | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| legal_reference | 22 | 1 | 0 | 0.9565217391304348 | 1.0 | 0.9777777777777777 |
| document_identifier | 122 | 8 | 2 | 0.9384615384615385 | 0.9838709677419355 | 0.9606299212598426 |
| temporal_expression | 131 | 26 | 6 | 0.8343949044585988 | 0.9562043795620438 | 0.8911564625850341 |
| quantity | 34 | 17 | 6 | 0.6666666666666666 | 0.85 | 0.7472527472527473 |
| organization_or_authority | 76 | 85 | 9 | 0.4720496894409938 | 0.8941176470588236 | 0.6178861788617888 |

## Selector and A/B ablation

- Selector coarse page-level P/R/F1: 0.08823529411764706 / 0.3333333333333333 / 0.13953488372093023.
- Selected requests: 112; eligible requests: 1899; requests/representation: 11.2.
- TEXT_ONLY F1: 0.8279569892473119.
- SELECTIVE_VLM F1: 0.8279569892473119; executed requests: 0; F1 delta: 0.0.
- ALL_ELIGIBLE_VLM: N/A.
- Zero-request equality is an engineering baseline, not evidence that a model improves quality.

## Parser representations and consistency

- Marker P/R/F1: 0.7438271604938271 / 0.9450980392156862 / 0.8324697754749567.
- MinerU P/R/F1: 0.7272727272727273 / 0.9411764705882353 / 0.8205128205128205.
- Combined results are parser-representation-weighted: six source documents produce ten representations.

Cross-parser mention/path/value Jaccard consistency:

- `hanoi-master-plan-100y`: 0.3786764705882353 (intersection 103, union 272).
- `qd-23-2008-ubnd-hanoi-vien-quy-hoach`: 0.0 (intersection 0, union 11).
- `tt-04-2023-bkhdt-so-do-ban-do`: 0.7608695652173914 (intersection 35, union 46).
- `tt-04-2026-bxd-pl2-dinh-muc`: 0.625 (intersection 10, union 16).

## Failure and efficiency trace

- Semantic false positives: 137; semantic false negatives: 23. Machine records retain document, parser, page, kind, raw evidence, normalized value, and `unresolved` cause.
- VLM selector misses: 12; other VLM failures: zero because no corpus requests were executed.
- Transcription exact match/edit distance, model ID, endpoint, temperature, image bytes/dimensions, latency, failed requests, and tokens: N/A.

## Reproducibility

- Semantic determinism: 10/10 byte-identical.
- Structural and Physical freeze checks: passed.

## Real VLM experiment

**REAL VLM QUALITY EXPERIMENT NOT COMPLETED.** No model ID, endpoint, latency, or model-derived corpus outputs are claimed. Replay/contract tests prove offline engineering behavior only. A real model execution is required before claiming improvement.

## Limitations

Statements are exact direct-content spans, not paraphrases. Mentions use a controlled deterministic Vietnamese taxonomy. Matching uses page, kind, raw text, and normalized value; failures remain `unresolved` unless evidence establishes a cause. Parser aggregates are representation-weighted: six source documents produce ten parser representations. Selector labels are coarse page-level visual-audit labels, so selector metrics are diagnostic rather than region-level quality estimates. No RAG, KG, cross-document citation resolution, or entity resolution is implemented.
