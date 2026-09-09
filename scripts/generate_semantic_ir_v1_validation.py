"""Create Semantic IR v1 references/evidence or reproduce the report offline."""

import argparse
from pathlib import Path

from vlm_rag.evaluation.semantic import (
    load_semantic_reference_annotations,
    write_semantic_ir_v1_validation,
    write_semantic_reference_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-candidates",
        action="store_true",
        help="export non-authoritative parser/extractor candidates for an explicit visual audit",
    )
    parser.add_argument(
        "--validate-reference",
        action="store_true",
        help="validate committed authoritative reference v2 without generating or changing truth",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="verify committed outputs and reproduce the report without raw parser artifacts",
    )
    args = parser.parse_args()
    if args.export_candidates:
        write_semantic_reference_candidates(Path.cwd())
        return
    if args.validate_reference:
        load_semantic_reference_annotations(Path.cwd())
        return
    write_semantic_ir_v1_validation(Path.cwd(), collect=not args.render_only)


if __name__ == "__main__":
    main()
