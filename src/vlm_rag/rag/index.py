"""Immutable SQLite FTS5 indexes over explicit Semantic IR representations."""

import hashlib
import json
import re
import sqlite3
import unicodedata
from contextlib import closing
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from vlm_rag.semantic_ir.models import SemanticDocument, TextSemanticAnchor, VisualSemanticAnchor
from vlm_rag.semantic_ir.serialization import semantic_document_to_json

APPLICATION_ID = 1447841106


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    document_id: str
    version_id: str
    source_artifact_sha256: str
    semantic_sha256: str
    statement_id: str
    structural_path: str
    page_indexes: tuple[int, ...]
    char_start: int
    char_end: int
    text: str
    provenance: str
    source_visual_observation_id: str | None
    anchors: tuple[TextSemanticAnchor | VisualSemanticAnchor, ...]


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    chunk: Chunk
    score: float


def chunks_for_document(document: SemanticDocument, *, size: int = 1200) -> list[Chunk]:
    if not 100 <= size <= 4000:
        raise ValueError("chunk size must be between 100 and 4000 characters")
    digest = hashlib.sha256(semantic_document_to_json(document).encode("utf-8")).hexdigest()
    result = []
    for statement in document.statements:
        for start in range(0, len(statement.text), size):
            end = min(start + size, len(statement.text))
            text = statement.text[start:end]
            if not text.strip():
                continue
            identity = json.dumps([digest, statement.id, start, end]).encode()
            result.append(
                Chunk(
                    id=hashlib.sha256(identity).hexdigest(),
                    document_id=document.document_id,
                    version_id=document.version_id,
                    source_artifact_sha256=document.source_artifact_sha256,
                    semantic_sha256=digest,
                    statement_id=statement.id,
                    structural_path=statement.structural_canonical_path,
                    page_indexes=tuple(
                        sorted({anchor.page_index for anchor in statement.evidence_anchors})
                    ),
                    char_start=start,
                    char_end=end,
                    text=text,
                    provenance=statement.provenance.value,
                    source_visual_observation_id=statement.source_visual_observation_id,
                    anchors=statement.evidence_anchors,
                )
            )
    return result


def create_index(path: Path, documents: list[SemanticDocument]) -> dict[str, int]:
    if not documents:
        raise ValueError("at least one SemanticDocument is required")
    identities = [(doc.document_id, doc.version_id) for doc in documents]
    if len(set(identities)) != len(identities):
        raise ValueError("choose exactly one representation per document/version")
    chunks = [chunk for doc in documents for chunk in chunks_for_document(doc)]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never replaces a source artifact or an existing index.
    with path.open("xb"):
        pass
    try:
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
            connection.execute("PRAGMA user_version=1")
            connection.execute(
                "CREATE TABLE chunks (id TEXT PRIMARY KEY, document_id TEXT, "
                "version_id TEXT, payload TEXT)"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE search USING fts5(text, path, "
                "tokenize='unicode61 remove_diacritics 2')"
            )
            for chunk in sorted(chunks, key=lambda item: item.id):
                cursor = connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                    (chunk.id, chunk.document_id, chunk.version_id, chunk.model_dump_json()),
                )
                connection.execute(
                    "INSERT INTO search(rowid, text, path) VALUES (?, ?, ?)",
                    (
                        cursor.lastrowid,
                        unicodedata.normalize("NFC", chunk.text),
                        chunk.structural_path,
                    ),
                )
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return {"documents": len(documents), "chunks": len(chunks)}


def search(
    path: Path,
    question: str,
    *,
    limit: int = 5,
    document_id: str | None = None,
    version_id: str | None = None,
) -> list[SearchHit]:
    if not question.strip() or len(question) > 2000 or not 1 <= limit <= 20:
        raise ValueError("query requires 1..2000 characters and limit 1..20")
    tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", unicodedata.normalize("NFC", question))))[
        :64
    ]
    if not tokens:
        return []
    expression = " OR ".join('"' + token + '"' for token in tokens)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if (
            connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID
            or connection.execute("PRAGMA user_version").fetchone()[0] != 1
        ):
            raise ValueError("unsupported RAG index")
        rows = connection.execute(
            "SELECT chunks.payload, bm25(search, 1.0, 0.2) AS score FROM search "
            "JOIN chunks ON chunks.rowid=search.rowid WHERE search MATCH ? "
            "AND (? IS NULL OR chunks.document_id=?) AND (? IS NULL OR chunks.version_id=?) "
            "ORDER BY score, chunks.id LIMIT ?",
            (expression, document_id, document_id, version_id, version_id, limit),
        ).fetchall()
    return [
        SearchHit(chunk=Chunk.model_validate_json(payload), score=-float(score))
        for payload, score in rows
    ]
