"""Create Semantic IR v1 references/evidence or reproduce the report offline."""

import argparse
from pathlib import Path

from vlm_rag.evaluation.semantic import (
    write_semantic_ir_v1_validation,
    write_semantic_reference_annotations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create-references",
        action="store_true",
        help="create disclosed reference v1 from the frozen #008 visual-audit excerpts",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="verify committed outputs and reproduce the report without raw parser artifacts",
    )
    args = parser.parse_args()
    if args.create_references:
        write_semantic_reference_annotations(Path.cwd())
        return
    write_semantic_ir_v1_validation(Path.cwd(), collect=not args.render_only)


if __name__ == "__main__":
    main()
