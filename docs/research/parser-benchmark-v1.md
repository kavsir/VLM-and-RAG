# Parser Benchmark: Vietnamese Legal and Planning Corpus v1

## Scope and provenance

This benchmark contains 6 authoritative documents spanning 250 pages. It evaluates 110 reference regions on 46 selected pages.

The annotations are v3 AI visual reference annotations created with `visual_pdf_reaudit`. Prior parser output exposure was `true`; parser outputs were not used as the reference source for region geometry or type.

## Parser coverage

- **marker 2.0.0**: 6/6 documents and 250/250 pages (`fast-no-ocr`).
- **mineru 3.4.5**: 4/6 documents and 150/250 pages (`pipeline`).

## Results

| Document ID | Parser | Pages | Runtime (s) | Pages/s | Audited pages | Spatial P | Spatial R | Spatial F1 | Mean IoU | Pairwise order |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hanoi-master-plan-100y` | marker | 80 | 159.4492 | 0.5017 | 10 | 0.0276 | 0.2500 | 0.0497 | 0.7862 | 1.0000 |
| `hanoi-master-plan-100y` | mineru | 80 | 1607.9059 | 0.0498 | 10 | 0.0195 | 0.1875 | 0.0353 | 0.7136 | 0.3333 |
| `luat-112-2025-qh15` | marker | 52 | 49.9471 | 1.0411 | 10 | 0.1507 | 0.6471 | 0.2444 | 0.7604 | 0.7446 |
| `vbhn-103-2026-quy-hoach-tong-the` | marker | 48 | 43.3946 | 1.1061 | 8 | 0.1881 | 0.7037 | 0.2969 | 0.7849 | 0.7544 |
| `tt-04-2026-bxd-pl2-dinh-muc` | marker | 19 | 22.0152 | 0.8630 | 6 | 0.2258 | 0.6364 | 0.3333 | 0.7871 | 0.7143 |
| `tt-04-2026-bxd-pl2-dinh-muc` | mineru | 19 | 125.2493 | 0.1517 | 6 | 0.2059 | 0.6364 | 0.3111 | 0.7311 | 0.7619 |
| `tt-04-2023-bkhdt-so-do-ban-do` | marker | 47 | 40.2257 | 1.1684 | 8 | 0.2500 | 0.7273 | 0.3721 | 0.8501 | 0.8000 |
| `tt-04-2023-bkhdt-so-do-ban-do` | mineru | 47 | 418.1107 | 0.1124 | 8 | 0.2500 | 0.7273 | 0.3721 | 0.7413 | 0.3833 |
| `qd-23-2008-ubnd-hanoi-vien-quy-hoach` | marker | 4 | 16.4995 | 0.2424 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |
| `qd-23-2008-ubnd-hanoi-vien-quy-hoach` | mineru | 4 | 82.4946 | 0.0485 | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A |

## Metric definitions

- `spatial_precision`: `TP_spatial / (TP_spatial + FP_spatial)`.
- `spatial_recall`: `TP_spatial / (TP_spatial + FN_spatial)`.
- `spatial_f1`: `2 * spatial_precision * spatial_recall / (spatial_precision + spatial_recall)`.
- `mean_iou`: `sum(IoU of spatial matches) / spatial_matches`.
- `classification_accuracy_on_matched`: `correct_classified_spatial_matches / spatial_matches`.
- `pairwise_order_accuracy`: `concordant comparable pairs / total comparable pairs`. Null when fewer than two matched regions; prediction ties are non-concordant.

## Scanned-document evidence

- **marker** produced 74 structural Physical IR blocks, 0 with non-empty text, and 70 typed TEXT/TITLE.
- **mineru** produced 82 structural Physical IR blocks, 81 with non-empty text, and 76 typed TEXT/TITLE.
No OCR recall, CER, or WER is reported because no text transcription reference exists.

## Visual page evidence

The layout diagram is on PDF page 22 (page_index=21). PDF page 26 (page_index=25) is prose. The verified symbology sheets are: PDF page 27 (page_index=26), PDF page 28 (page_index=27), PDF page 29 (page_index=28), PDF page 30 (page_index=29), PDF page 31 (page_index=30), PDF page 32 (page_index=31).

## Normalization determinism

All 10 retained parser/document pairs produced byte-identical A/B Physical IR JSON. This statement is generated from `data/benchmarks/normalization_determinism.v1.json`.

## Reproducibility boundary

A clean clone can validate committed machine artifacts, regenerate these Markdown reports offline, and run unit tests. Raw parser outputs, source PDFs, run manifests, and Physical IR files remain intentionally untracked; restoring the paths and hashes listed in the benchmark manifest is required to recollect Stage A evidence.
