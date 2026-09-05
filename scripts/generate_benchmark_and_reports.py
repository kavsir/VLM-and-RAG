"""CLI entry point for benchmark evidence collection and offline report rendering."""

import argparse
from pathlib import Path

from vlm_rag.evaluation.reporting import (
    generate_all_benchmarks_and_reports,
    render_reports_from_committed_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="render reports from committed machine evidence without external parser artifacts",
    )
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    if arguments.render_only:
        render_reports_from_committed_artifacts(root)
    else:
        generate_all_benchmarks_and_reports(root)


if __name__ == "__main__":
    main()
