# MinerU Hanoi baseline

This report characterizes one real parse of the verified Hanoi golden document. It is
observation input for Issue #004, not a normalized document model or a claim that the
output is byte-deterministic. The accompanying
[`mineru-hanoi-baseline.json`](mineru-hanoi-baseline.json) contains the same core
measurements in machine-readable form.

## Run

| Field | Observed value |
| --- | --- |
| Document / version | `hanoi-master-plan-100y` / `v1` |
| Input SHA-256 | `ce87f7f636ca1c0518d237bcca9f92e184e478d0321a0ad580122c15500d6028` |
| Input size | 2,218,758 bytes |
| Source PDF pages | 80 |
| MinerU | 3.4.5 |
| Backend | `pipeline` |
| Parser environment | Python 3.12.10 on Windows |
| Device selected by MinerU | CPU |
| Started (UTC) | 2026-09-04T09:19:28.026523+00:00 |
| Completed (UTC) | 2026-09-04T09:46:15.931838+00:00 |
| Duration | 1,607.906 seconds |
| Exit code | 0 |

The source checksum was verified against the Issue #002 manifest before the adapter
started MinerU. The generated `run.json`, logs, and raw output are retained locally at
`data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/` and ignored
by Git.

## Structure observed

| Metric | Count |
| --- | ---: |
| MinerU pages | 80 |
| Paragraphs | 982 |
| Titles/headings | 123 |
| Tables | 0 |
| Images | 0 |
| Charts (explicit type) | 0 |
| Formulas | 0 |
| Discarded blocks | 81 |
| Headers | 2 |
| Footers | 0 |
| Page numbers | 79 |

Page indexes are complete and contiguous from 0 through 79. All 80 `middle.json`
pages contain `page_size`. All 1,186 exported `content_list.json` and
`content_list_v2.json` items contain a bbox. Heading levels are level 1 (one item) and
level 2 (122 items).

The zero table, image, chart, and formula counts mean that MinerU emitted no blocks of
those types in this run. They are not a conclusion about what a human would identify
visually in the source.

## Representation differences

- `source_middle.json` reports `_backend: pipeline`, `_version_name: 3.4.5`, and 80
  `pdf_info` entries. Its retained `para_blocks` comprise 982 `text` and 123 `title`
  blocks; `discarded_blocks` comprise 2 headers and 79 page numbers.
- `source_content_list.json` is a flat reading-order list of 1,186 items. Its 1,105
  `text` items combine the 982 paragraphs and 123 headings; `text_level` distinguishes
  the headings. The remaining items are 2 headers and 79 page numbers.
- `source_content_list_v2.json` is a list of 80 per-page lists, not a flat list of
  objects with `page_idx`. Its 1,186 blocks use the types `paragraph` (982), `title`
  (123), `page_header` (2), and `page_number` (79). Page association is positional in
  the outer list.
- `source.md` contains 234,522 characters across 2,151 lines, including 123 Markdown
  heading lines.

These differences are preserved rather than reconciled into a new schema.

## Captured artifacts

| Relative raw path | Bytes | SHA-256 |
| --- | ---: | --- |
| `source/auto/source.md` | 308,780 | `57ad2127afd3c2e43bd88c32de690a6d005536625d2b66b33c3a55ef65dcc4c9` |
| `source/auto/source_content_list.json` | 518,799 | `23d24aafd5e13a5babf6344aaa6a0a28ba944da49ca19ca708e7075d4fefc4ec` |
| `source/auto/source_content_list_v2.json` | 770,845 | `d3a19945ddf5ff3a9ab28a389af88d6e5bac1c06c25f9b23d0516051d5d31149` |
| `source/auto/source_layout.pdf` | 9,996,435 | `fae13d4e4b8c4244689ee48a7339379291c3d13fa2085ba1ec432447001be03e` |
| `source/auto/source_middle.json` | 7,256,419 | `f8bb08cc846b6502a940cd674d10e2bf773dcb388b977acc360a7e8656932cff` |
| `source/auto/source_model.json` | 1,300,246 | `e8a214568c7c7bcdc77a39bdff6fb3040c65e8255a2116028fc66bad5ea1ddf0` |
| `source/auto/source_origin.pdf` | 2,075,074 | `42d26544aeae963fbcd08ec71311c6a173de37966864edff4b69b9344f4eb1de` |
| `source/auto/source_span.pdf` | 9,956,572 | `8a2396e77a16c2239bf069c1ab96cbef83fb8a93c1a55c0bbebb7fd2baf01b2d` |

The generated `source_origin.pdf` still has 80 pages, but its byte size and SHA-256
differ from the registered input PDF. No image files were emitted.

## Runtime observations

- MinerU 3.4.5's published `pipeline` dependencies did not install `six`, although
  this runtime imported it. The isolated parser environment therefore required the
  explicit compatibility installation `six==1.17.0`.
- The default Hugging Face model retrieval reached the Windows cache but failed when
  it attempted to create a symlink without the required Windows privilege. Retrying
  with MinerU's documented `MINERU_MODEL_SOURCE=modelscope` setting completed the
  model acquisition and parse.
- Model acquisition is a parser-environment setup concern. The adapter itself makes
  no network request, and CI uses subprocess mocks without installing MinerU.

The raw structures need broader comparison across documents before any stable
physical document IR can be designed.
