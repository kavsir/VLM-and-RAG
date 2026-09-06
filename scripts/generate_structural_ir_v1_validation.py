"""Collect Structural IR v1 evidence or reproduce its report offline."""

import argparse
from pathlib import Path

from vlm_rag.evaluation.structural import write_structural_ir_v1_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="verify committed outputs/metrics and render without retained raw parser artifacts",
    )
    args = parser.parse_args()
    write_structural_ir_v1_validation(Path.cwd(), collect=not args.render_only)


if __name__ == "__main__":
    main()
