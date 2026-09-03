"""Command-line interface for the golden document registry."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from vlm_rag.registry.artifacts import ArtifactError, fetch_artifact
from vlm_rag.registry.manifest import ManifestValidationError, load_manifest

DEFAULT_MANIFEST = Path("data/manifests/hanoi_master_plan_100y.v1.yaml")
DEFAULT_DESTINATION = Path("data/golden/hanoi_master_plan_100y/v1/source.pdf")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and fetch golden document artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a manifest offline")
    validate_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)

    fetch_parser = subparsers.add_parser("fetch", help="fetch and verify a registered artifact")
    fetch_parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT_MANIFEST)
    fetch_parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    fetch_parser.add_argument("--timeout", type=float, default=60.0, metavar="SECONDS")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the registry CLI and return a process status code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        manifest = load_manifest(arguments.manifest)
        if arguments.command == "validate":
            print(
                f"valid manifest: document={manifest.document.id} "
                f"version={manifest.version.id} sha256={manifest.artifact.sha256}"
            )
            print(f"landing page: {manifest.source.landing_page_url}")
            print(f"asset: {manifest.source.asset_url}")
            return 0

        verified = fetch_artifact(
            manifest.source,
            manifest.artifact,
            arguments.output,
            timeout_seconds=arguments.timeout,
        )
    except (ArtifactError, ManifestValidationError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    print(
        f"verified artifact: path={verified.path} bytes={verified.byte_size} "
        f"sha256={verified.sha256}"
    )
    print(f"resolved asset: {verified.final_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
