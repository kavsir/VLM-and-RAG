"""Command-line entry point for the optional external MinerU adapter."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vlm_rag.parsers.mineru import (
    MINERU_BACKEND,
    MINERU_NAME,
    MinerUAdapter,
    MinerUError,
)
from vlm_rag.registry import ArtifactError, ManifestValidationError, load_manifest

DEFAULT_MANIFEST = Path("data/manifests/hanoi_master_plan_100y.v1.yaml")
DEFAULT_INPUT = Path("data/golden/hanoi_master_plan_100y/v1/source.pdf")
DEFAULT_OUTPUT_ROOT = Path("data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe or run the external MinerU parser")
    parser.add_argument(
        "--executable",
        default=MINERU_NAME,
        help="MinerU executable name or explicit path",
    )
    parser.add_argument("--timeout", type=float, default=14_400.0, metavar="SECONDS")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("probe", help="report the actual external MinerU version")

    run_parser = subparsers.add_parser("run", help="verify and parse the Hanoi golden PDF")
    run_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    run_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run_parser.add_argument("--backend", choices=[MINERU_BACKEND], default=MINERU_BACKEND)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional MinerU adapter CLI."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        adapter = MinerUAdapter(
            executable=arguments.executable,
            backend=getattr(arguments, "backend", MINERU_BACKEND),
            timeout_seconds=arguments.timeout,
        )
        if arguments.command == "probe":
            probe = adapter.probe()
            print(
                json.dumps(
                    {
                        "parser": probe.parser,
                        "version": probe.version,
                        "executable_path": str(probe.executable_path),
                    },
                    indent=2,
                )
            )
            return 0

        manifest = load_manifest(arguments.manifest)
        run = adapter.run(
            manifest=manifest,
            input_pdf=arguments.input,
            output_root=arguments.output_root,
        )
    except (ArtifactError, ManifestValidationError, MinerUError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(json.dumps(run.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
