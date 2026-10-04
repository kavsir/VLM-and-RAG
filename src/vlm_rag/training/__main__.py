"""Validate local training manifests without importing Torch or downloading a model."""

import argparse
import json
from pathlib import Path

from vlm_rag.training.data import load_samples, validate_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path("data/semantic_annotations"))
    args = parser.parse_args()
    result = validate_splits(
        load_samples(args.train, args.images),
        load_samples(args.validation, args.images),
        args.reference_root,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
