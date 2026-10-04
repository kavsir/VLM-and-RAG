"""Local end-to-end VLM/RAG commands; network requires --live and a request budget."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pydantic import ValidationError

from vlm_rag.physical_ir.serialization_v1 import physical_document_v1_from_json
from vlm_rag.rag.answer import answer_question
from vlm_rag.rag.index import create_index, search
from vlm_rag.runtime.chat import ChatClient, ModelCallError, ModelSettings
from vlm_rag.runtime.pdf import PDFiumRenderer
from vlm_rag.runtime.vlm import enrich_document, write_new_json
from vlm_rag.semantic_ir import build_semantic_document, semantic_document_from_json
from vlm_rag.semantic_ir.serialization import semantic_document_to_json
from vlm_rag.structural_ir import VietnameseStructuralExtractor
from vlm_rag.structural_ir.serialization import structural_document_from_json
from vlm_rag.vlm.evidence import VerifiedAssetResolver, VerifiedPDFResolver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="create a new index from explicit Semantic IR files")
    index.add_argument("inputs", type=Path, nargs="+")
    index.add_argument("--output", type=Path, required=True)
    ask = commands.add_parser("ask", help="retrieve evidence or generate a cited answer")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, required=True)
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--document-id")
    ask.add_argument("--version-id")
    ask.add_argument("--live", action="store_true")
    ask.add_argument("--output", type=Path)
    prepare = commands.add_parser(
        "prepare", help="Physical IR v1 to Semantic IR, optionally with VLM"
    )
    prepare.add_argument("--physical", type=Path, required=True)
    prepare.add_argument("--structural", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    sources = prepare.add_mutually_exclusive_group()
    sources.add_argument("--pdf", type=Path)
    sources.add_argument("--asset-root", type=Path)
    prepare.add_argument("--live", action="store_true")
    prepare.add_argument("--max-requests", type=int, default=2)
    return parser


def run(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "index":
            documents = [
                semantic_document_from_json(path.read_text(encoding="utf-8"))
                for path in args.inputs
            ]
            result: object = create_index(args.output, documents)
        elif args.command == "ask":
            if args.output is not None and args.output.exists():
                raise ValueError("output already exists")
            hits = search(
                args.index,
                args.question,
                limit=args.top_k,
                document_id=args.document_id,
                version_id=args.version_id,
            )
            client = ChatClient(ModelSettings()) if args.live else None
            try:
                answer = answer_question(args.question, hits, client=client)
            finally:
                if client is not None:
                    client.close()
            result = answer.model_dump(mode="json")
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_new_json(args.output, result)
        else:
            if args.output.exists() or "raw" in args.output.resolve().parts:
                raise ValueError("use a new output directory outside raw evidence")
            physical = physical_document_v1_from_json(args.physical.read_text(encoding="utf-8"))
            structural = (
                structural_document_from_json(args.structural.read_text(encoding="utf-8"))
                if args.structural
                else VietnameseStructuralExtractor().extract(physical)
            )
            if args.live:
                if args.pdf is None and args.asset_root is None:
                    raise ValueError("live preparation requires --pdf or --asset-root")
                resolver = (
                    VerifiedPDFResolver(args.pdf, PDFiumRenderer())
                    if args.pdf
                    else VerifiedAssetResolver(args.asset_root)
                )
                client = ChatClient(ModelSettings())
                try:
                    path = enrich_document(
                        physical,
                        structural,
                        resolver=resolver,
                        client=client,
                        output=args.output,
                        max_requests=args.max_requests,
                        live=True,
                    )
                finally:
                    client.close()
            else:
                document = build_semantic_document(physical, structural)
                args.output.mkdir(parents=True, exist_ok=False)
                path = args.output / "semantic.json"
                with path.open("xb") as stream:
                    stream.write(semantic_document_to_json(document).encode("utf-8"))
            result = {"semantic_path": str(path), "live": args.live}
        # JSON escapes preserve Unicode even under Windows redirected legacy encodings.
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0
    except ValidationError:
        print(
            "Invalid configuration or model payload; check schema and MODEL_* settings.",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, sqlite3.Error) as exc:
        # ModelCallError text is sanitized by the transport; never print provider payloads.
        message = str(exc) if isinstance(exc, ModelCallError) else type(exc).__name__
        print(
            f"Command failed: {message}. Check inputs and retained run evidence.", file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(run())
