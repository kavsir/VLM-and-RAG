# Marker fast-no-ocr Hanoi baseline

This report records one real Marker 2.0.0 parse of the verified Hanoi golden document and its
normalization into the existing Physical Document IR v0. Counts and timings are parser
observations, not accuracy measurements.

## Environment

| Field | Observed value |
| --- | --- |
| Document / version | `hanoi-master-plan-100y` / `v1` |
| Source bytes | 2,218,758 |
| Source SHA-256 | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| Marker distribution | `marker-pdf==2.0.0` |
| Parser Python | 3.12.10 |
| Host | Windows; Torch 2.14.0+cpu; CUDA unavailable |
| Experiment | `fast-no-ocr` |
| Effective configuration | `mode=fast`, `disable_ocr=true`, `output_format=json` |
| Process wall time | 159.449 seconds |
| Wall time per page | 1.993 seconds/page |
| Marker-reported conversion time | 122.409 seconds |

The source size and checksum were verified against the Issue #002 manifest before either the
version probe or parser process ran. The parser environment was created with:

```powershell
uv venv .venv-marker --python 3.12
uv pip install --python .venv-marker\Scripts\python.exe "marker-pdf==2.0.0"
```

The exact installation resolved without an additional compatibility constraint. All 83 external
packages, including Marker, Surya, and Torch, stayed inside `.venv-marker`.

The independent version probe returned `2.0.0`:

```powershell
.venv-marker\Scripts\python.exe -c "import importlib.metadata; print(importlib.metadata.version('marker-pdf'))"
```

That command and the environment row above describe the retained historical run. The hardened
adapter now requires `marker_single` and Python to resolve from the same environment directory,
runs its package/device probe and parse with the same CUDA/HIP/ROCm-hiding environment, and fails
sets Marker's `TORCH_DEVICE=cpu`, and fails if Torch still observes an accelerator. Successful
future run manifests record the observed
device, the `cpu-only` policy, and the effective isolation variables.

## Command

The adapter executed this argument sequence (shown as a readable PowerShell command):

```powershell
.venv-marker\Scripts\marker_single.exe `
  data\golden\hanoi_master_plan_100y\v1\source.pdf `
  --mode fast `
  --disable_ocr `
  --output_format json `
  --output_dir data\golden\hanoi_master_plan_100y\v1\parser_runs\marker\2.0.0\fast-no-ocr\raw `
  --JSONRenderer_keep_pageheader_in_output `
  --JSONRenderer_keep_pagefooter_in_output
```

The installed `marker_single --help` confirmed `--mode [balanced|fast]`, `--disable_ocr`,
`--output_format [markdown|json|html|chunks]`, `--output_dir`, and both JSON renderer retention
flags. `--use_llm` was not supplied.

## Raw output structure

The output directory is
`data/golden/hanoi_master_plan_100y/v1/parser_runs/marker/2.0.0/fast-no-ocr/` and is ignored by Git.

The primary `raw/source/source.json` is an object with exactly `children` and `block_type` at its
root. `block_type` is `Document`; `children` is an 80-item array of `Page` objects. Page IDs span
`/page/0/Page/337` through `/page/79/Page/219`. Every page has fields `id`, `block_type`, `html`,
`polygon`, `bbox`, `children`, `section_hierarchy`, and `images`.

Every page bbox is `[0.0, 0.0, 596.0, 842.0]`, with a matching four-corner polygon. Direct page
children have the same structural fields and are already in reading order. HTML is the visible
representation for leaf blocks. The one `ListGroup` instead contains `<content-ref>` placeholders
and 12 nested `ListItem` children. The one `Picture` contains an inline base64 JPEG in its `images`
mapping; no separate image file was emitted.

`raw/source/source_meta.json` is a distinct metadata sidecar with `table_of_contents`,
`page_stats`, and `debug_data_path`. It contains 80 page-stat entries and 119 table-of-contents
entries. Every page-stat entry reports `pdftext`, and aggregate LLM request count is zero. The
primary document tree does not contain a `metadata` field. Discovery therefore excludes
`_meta.json` by name and validates the document-tree shape; ambiguity fails.

## Raw counts

The canonical population is every block directly owned by a `Page`. This is the nearest
non-overlapping page-level population observed in Marker 2.0.0.

| Raw direct-page block type | Count |
| --- | ---: |
| `Text` | 1,004 |
| `SectionHeader` | 119 |
| `PageHeader` | 80 |
| `PageFooter` | 0 |
| `ListGroup` | 1 |
| `Picture` | 1 |
| `Table` | 0 |
| `Figure` | 0 |
| `Equation` | 0 |
| `Caption` | 0 |
| **Canonical total** | **1,205** |

The tree has 1,297 nodes below the `Document`: 80 pages, 1,205 canonical blocks, and 12 nested
`ListItem` nodes. Maximum depth below `Document` is three levels (`Page` → `ListGroup` →
`ListItem`). All 1,205 canonical blocks have bbox data. A nonempty `section_hierarchy` appears on
1,202 canonical blocks, but it is context inherited from prior headings and is not used to invent a
heading level.

## Retained artifacts

| Relative run path | Bytes | SHA-256 |
| --- | ---: | --- |
| `raw/source/source.json` | 1,596,521 | `be138bdaf3bc630ab0a4a8c77b21e0a49bd7073039ef8073bb2d234cc3562bda` |
| `raw/source/source_meta.json` | 109,937 | `08d016a59075bfb9f84353c1a15a875510fd2c8528244b8e65605af5f22ebad8` |
| `run.json` | 1,856 | `28751876baef128b37cab078c2e434c3e28a8bce6441220c4abde5d7baf1fe04` |
| `stderr.log` | 228 | `9db49c39ac710c59596fb81b174c231d2019518c7f76daff81bd1ee3114f21e0` |
| `stdout.log` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

`artifact_manifest.json` records these hashes and intentionally cannot hash itself.

## Normalization mapping

| Marker canonical type | Physical IR kind | Disposition |
| --- | --- | --- |
| `Text` | `TEXT` | `CONTENT` |
| `SectionHeader` | `TITLE` | `CONTENT` |
| `PageHeader` | `HEADER` | `DISCARDED` |
| `PageFooter` | `UNKNOWN` | `DISCARDED` |
| Any other direct-page type | `UNKNOWN` | `CONTENT` |

HTML is converted to deterministic plain visible text with the Python standard library. For the
canonical `ListGroup`, nested references are expanded only to obtain the parent's visible text;
nested children do not become additional PhysicalBlocks. A `SectionHeader` gets a heading level
only from its own rendered `<h1>` through `<h6>` tag. Numeric footer text is never inferred to be a
page number.

Marker native bbox coordinates are projected to `normalized_1000` relative to the full native page
bounds, with nonzero origins supported and results rounded to six decimal places. Out-of-page or
reversed geometry fails rather than being clipped. A native coordinate no more than `1e-6` beyond
an edge is treated solely as floating-point noise and snapped to that exact native edge before
projection; an excursion beyond that tolerance fails.

## Physical IR counts

| Physical IR metric | Count |
| --- | ---: |
| Pages | 80 |
| PhysicalBlocks | 1,205 |
| TEXT | 1,004 |
| TITLE | 119 |
| HEADER | 80 |
| PAGE_NUMBER | 0 |
| UNKNOWN | 2 |
| CONTENT disposition | 1,125 |
| DISCARDED disposition | 80 |
| UNKNOWN disposition | 0 |
| Bbox coverage | 1,205 / 1,205 (100%) |
| Heading-level coverage | 104 / 119 (87.4%) |
| Dropped canonical blocks | 0 |

The two UNKNOWN blocks are the one `ListGroup` and one `Picture`. Both remain CONTENT and preserve
raw provenance. These are representational gaps in v0, not silently dropped evidence.

## Determinism

The same retained raw JSON was normalized twice through the canonical Physical IR serializer:

| Measurement | Run A | Run B |
| --- | --- | --- |
| Bytes | 1,162,526 | 1,162,526 |
| SHA-256 | `d58d65296160dabf7fdb81774c123e15165477495e40aabe879c08e28eb546aa` | `d58d65296160dabf7fdb81774c123e15165477495e40aabe879c08e28eb546aa` |

This proves deterministic normalization for the identical retained input and ordering. It does not
claim that rerunning Marker produces byte-identical raw output.

## MinerU comparison

| Metric | MinerU 3.4.5 pipeline | Marker 2.0.0 fast-no-ocr |
| --- | ---: | ---: |
| Process runtime | 1,607.906 s | 159.449 s |
| Seconds/page | 20.099 | 1.993 |
| Pages | 80 | 80 |
| Physical blocks | 1,186 | 1,205 |
| TEXT | 982 | 1,004 |
| TITLE | 123 | 119 |
| HEADER | 2 | 80 |
| PAGE_NUMBER | 79 | 0 |
| UNKNOWN | 0 | 2 |
| Bbox coverage | 1,186 / 1,186 (100%) | 1,205 / 1,205 (100%) |
| Heading-level coverage | 123 / 123 (100%) | 104 / 119 (87.4%) |

Differences reflect parser representation and this conservative v0 mapping. They are not evidence
that one parser is more accurate.

## Known limitations

* One born-digital Vietnamese planning document cannot characterize OCR, tables, equations, or a
  broad document population.
* The no-OCR baseline deliberately does not evaluate scanned/corrupted-page recovery.
* Physical IR v0 cannot name list or picture kinds; they remain traceable UNKNOWN content.
* Header classification differs substantially from MinerU and has not been quality-scored.
* Empty visible text occurs in some raw Text/SectionHeader records and the Picture; normalization
  preserves those canonical records rather than silently dropping them.
* Parser-output repeatability and OCR/balanced/VLM comparisons remain Issue #006 work.
