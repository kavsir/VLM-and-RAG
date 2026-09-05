"""Collect or render Physical IR v1 retained-corpus validation evidence."""

import argparse
from pathlib import Path

from vlm_rag.physical_ir.validation_v1 import write_physical_ir_v1_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render the report from committed machine evidence without retained raw artifacts",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    write_physical_ir_v1_validation(root, collect=not arguments.render_only)


if __name__ == "__main__":
    main()
