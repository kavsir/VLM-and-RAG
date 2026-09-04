# 5. Add Marker Through an External Runtime Boundary

Date: 2026-09-04

## Status

Accepted

## Context

Physical Document IR v0 was first populated from MinerU. A second independent parser is needed to
test whether that IR is genuinely parser-independent without changing its schema for
parser-specific concepts. Marker carries Torch, Surya, model, and service-client dependencies that
do not belong in the main package or its offline CI environment.

The registered Hanoi PDF is born-digital and has a usable text layer. This experiment concerns the
parser boundary, physical structure, reading order, and coordinate mapping; it is not an OCR or VLM
quality benchmark.

## Decision

Marker is installed only as `marker-pdf==2.0.0` in a separate Python 3.12 `.venv-marker`
environment. The main package invokes the public `marker_single` executable with `shell=False` and
never imports `marker.*` or `surya.*`. The adapter verifies the registered source bytes before the
process, probes the installed distribution with `importlib.metadata`, uses a timeout, captures
stdout/stderr, and hashes retained run evidence.

The resolved `marker_single` and Python executables must have the same environment directory. The
version/device probe and parse subprocess receive the same controlled environment, which hides
CUDA, HIP, and ROCm devices and configures Marker's supported `TORCH_DEVICE=cpu` setting. Torch is
imported only by the external probe process to observe the effective device; an accelerator that
remains available is an error under the recorded `cpu-only`
policy. The successful run manifest records the observed device, policy, and isolation variables
rather than fabricating device provenance.

The experiment is named `fast-no-ocr` and explicitly executes `mode=fast`,
`disable_ocr=true`, and `output_format=json`. Header and footer HTML retention is also explicit so
these detected physical regions remain observable. No LLM service or network-backed semantic layer
is configured. OCR, balanced, and VLM-assisted variants are deferred to Issue #006.

The observed Marker 2.0.0 JSON boundary is:

* a primary object with `block_type: Document` and `children` containing 80 `Page` objects;
* page objects with native `polygon`/`bbox` geometry and ordered `children`;
* direct page children as the canonical, non-overlapping PhysicalBlock population; and
* a separate `_meta.json` sidecar containing table-of-contents and page-statistics metadata.

Nested children remain in the immutable raw JSON but do not become duplicate PhysicalBlocks. When
a canonical group contains `<content-ref>` placeholders, its nested children are used only to
render that canonical block's visible text. Duplicate or unresolved references, cycles, and
excessive nesting fail with adapter-domain errors.

Marker native page bounds are projected into Physical IR v0's `normalized_1000` coordinates. The
projection accounts for nonzero page origins, rejects invalid/out-of-page geometry, and rounds to
six decimal places for deterministic serialization. Native coordinates outside a page by at most
`1e-6` are treated as floating-point noise and snapped to the exact native boundary before
projection; larger excursions fail rather than being broadly clipped. `PhysicalPage.width` and `height` retain the
parser-reported native canvas dimensions; they are not universally described as 72-DPI points.

File normalization accepts an explicit authoritative source-artifact path and rejects that exact
path as an output before parsing or writing. It also rejects the raw directory, every descendant,
and run-evidence files as destinations, preserving source and raw evidence on failure.

Physical IR v0 remains frozen. `Text`, `SectionHeader`, and `PageHeader` map to existing kinds.
`PageFooter` and richer unsupported physical types map conservatively to `UNKNOWN`; unsupported
ordinary content remains `CONTENT`, while headers and footers are `DISCARDED`. Numeric text is not
inferred to be a page number. Every block retains an exact raw artifact path and deterministic
global raw index.

## Consequences

* Marker, Surya, Torch, and models remain absent from `pyproject.toml`, `uv.lock`, and CI.
* Both MinerU and Marker populate the same Physical IR v0 without a core schema change.
* The raw tree remains available for richer future modeling while v0 avoids hierarchical text
  duplication.
* Unsupported `ListGroup` and `Picture` evidence remains visible as `UNKNOWN`, giving Issue #006
  concrete input for evaluating a later IR revision.
* Parser environment provisioning and model-cache licensing remain operational concerns. Marker
  code is Apache-2.0, while its published model-weight terms require separate review before wider
  commercial deployment.
