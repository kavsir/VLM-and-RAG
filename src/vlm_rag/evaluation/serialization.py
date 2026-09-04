"""Deterministic JSON serialization and loading for annotations and evaluation reports."""

import json
from pathlib import Path
from typing import Any

from vlm_rag.evaluation.models import DocumentAnnotation


def load_annotation_file(path: Path) -> DocumentAnnotation:
    """Read and validate a ground-truth layout annotation file."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return DocumentAnnotation.model_validate(raw)


def dump_evaluation_report(report_data: dict[str, Any], path: Path | None = None) -> str:
    """Serialize evaluation results into deterministic UTF-8 LF JSON."""
    text = json.dumps(report_data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return text
