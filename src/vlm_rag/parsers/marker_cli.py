"""Command-line entry point for the optional external Marker adapter."""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from vlm_rag.parsers.marker import MarkerAdapter, MarkerError
from vlm_rag.registry import ArtifactError, ManifestValidationError, load_manifest

DEFAULT_MANIFEST = Path("data/manifests/hanoi_master_plan_100y.v1.yaml")
DEFAULT_INPUT = Path("data/golden/hanoi_master_plan_100y/v1/source.pdf")
DEFAULT_OUTPUT_ROOT = Path("data/golden/hanoi_master_plan_100y/v1/parser_runs/marker")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe or run external Marker 2.0.0")
    parser.add_argument("--executable", default="marker_single")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--timeout", type=float, default=14_400.0, metavar="SECONDS")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="report the installed marker-pdf package version")
    run_parser = subparsers.add_parser("run", help="verify and parse the Hanoi golden PDF")
    run_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional Marker adapter CLI."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    adapter = MarkerAdapter(
        executable=arguments.executable,
        python_executable=arguments.python_executable,
        timeout_seconds=arguments.timeout,
    )
    try:
        if arguments.command == "probe":
            print(json.dumps(asdict(adapter.probe()), default=str, indent=2))
            return 0
        manifest = load_manifest(arguments.manifest)
        run = adapter.run(
            manifest=manifest,
            input_pdf=arguments.input,
            output_root=arguments.output_root,
        )
    except (ArtifactError, ManifestValidationError, MarkerError, OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
