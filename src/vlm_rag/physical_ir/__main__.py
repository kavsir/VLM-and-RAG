"""Command-line interface for Physical Document IR v0."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from vlm_rag.normalizers.mineru import MinerUPhysicalNormalizer, NormalizationError
from vlm_rag.physical_ir.serialization import (
    PhysicalIRSerializationError,
    physical_document_from_json,
)
from vlm_rag.registry import ManifestValidationError, load_manifest

DEFAULT_MANIFEST = Path("data/manifests/hanoi_master_plan_100y.v1.yaml")
DEFAULT_RAW_DIR = Path(
    "data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/raw"
)
DEFAULT_OUTPUT = Path(
    "data/golden/hanoi_master_plan_100y/v1/parser_runs/mineru/3.4.5/pipeline/physical_ir_v0.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physical Document IR v0 normalization and validation"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_parser = subparsers.add_parser(
        "normalize", help="normalize raw parser output into Physical Document IR"
    )
    normalize_parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="directory containing raw MinerU outputs",
    )
    normalize_parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="path to document manifest YAML",
    )
    normalize_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination path for serialized PhysicalDocument JSON",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="validate an existing PhysicalDocument JSON file"
    )
    validate_parser.add_argument(
        "file",
        type=Path,
        help="path to physical_ir_v0.json to validate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Physical Document IR CLI."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    if arguments.command == "validate":
        try:
            content = arguments.file.read_text(encoding="utf-8")
            doc = physical_document_from_json(content)
        except (OSError, PhysicalIRSerializationError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")
        total_blocks = sum(len(p.blocks) for p in doc.pages)
        print(
            f"valid PhysicalDocument: document={doc.document_id} version={doc.version_id} "
            f"pages={doc.page_count} blocks={total_blocks}"
        )
        return 0

    if arguments.command == "normalize":
        try:
            manifest = load_manifest(arguments.manifest)
            normalizer = MinerUPhysicalNormalizer()
            doc = normalizer.normalize_to_file(
                arguments.raw_dir,
                arguments.output,
                manifest=manifest,
            )
        except (ManifestValidationError, NormalizationError, OSError, ValueError) as exc:
            parser.exit(1, f"error: {exc}\n")

        total_blocks = sum(len(p.blocks) for p in doc.pages)
        print(
            f"normalized PhysicalDocument: document={doc.document_id} version={doc.version_id} "
            f"pages={doc.page_count} blocks={total_blocks}"
        )
        print(f"serialized to: {arguments.output}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
