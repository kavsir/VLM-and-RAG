"""CLI entry point to execute evaluations and generate benchmark artifacts and reports."""

from pathlib import Path

from vlm_rag.evaluation.reporting import generate_all_benchmarks_and_reports


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    generate_all_benchmarks_and_reports(root)


if __name__ == "__main__":
    main()
