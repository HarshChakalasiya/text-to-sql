"""Sync markdown doc chunks to Postgres pgvector and load dense index rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.llm.provider import LLMProvider
from app.retrieval.chunker import Chunk, load_chunks


def content_hash(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def embedding_model_name(settings: Settings, provider: LLMProvider) -> str:
    if provider.name == "openai":
        return settings.openai_embedding_model
    return settings.ollama_embedding_model


def pgvector_enabled(engine: Engine) -> bool:
    settings = get_settings()
    if settings.vector_store != "pgvector":
        return False
    return engine.dialect.name == "postgresql"


@dataclass
class IndexMeta:
    embedding_model: str
    dimension: int


def _table_exists(session: Session, table_name: str) -> bool:
    row = session.execute(
        text(
            """
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = :name
            """
        ),
        {"name": table_name},
    ).first()
    return row is not None


def _existing_embedding_dimension(session: Session) -> int | None:
    if not _table_exists(session, "document_embeddings"):
        return None
    row = session.execute(
        text(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = 'public'
              AND c.relname = 'document_embeddings'
              AND a.attname = 'embedding'
              AND NOT a.attisdropped
            """
        )
    ).first()
    if not row or row[0] is None:
        return None
    dim = int(row[0])
    return dim if dim > 0 else None


def _read_meta(session: Session) -> IndexMeta | None:
    if not _table_exists(session, "document_embedding_index"):
        return None
    try:
        row = session.execute(
            text(
                "SELECT embedding_model, dimension FROM document_embedding_index "
                "WHERE id = 1"
            )
        ).first()
    except ProgrammingError:
        session.rollback()
        return None
    if not row:
        return None
    return IndexMeta(embedding_model=str(row[0]), dimension=int(row[1]))


def ensure_pgvector_extension(engine: Engine) -> None:
    if engine.dialect.name != "postgresql":
        return
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))


def _ensure_embeddings_table(session: Session, *, dimension: int, embedding_model: str) -> None:
    meta = _read_meta(session)
    existing_dim = _existing_embedding_dimension(session)
    model_changed = meta is not None and meta.embedding_model != embedding_model
    dim_changed = meta is not None and meta.dimension != dimension
    legacy_dim_mismatch = meta is None and existing_dim is not None and existing_dim != dimension
    if model_changed or dim_changed or legacy_dim_mismatch:
        session.execute(text("DROP TABLE IF EXISTS document_embeddings"))
        session.execute(text("DROP TABLE IF EXISTS document_embedding_index"))
        session.commit()

    session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    session.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS document_embeddings (
                chunk_id VARCHAR(128) PRIMARY KEY,
                title VARCHAR(200) NOT NULL,
                source VARCHAR(200) NOT NULL,
                text TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL,
                embedding_model VARCHAR(128) NOT NULL,
                embedding vector({dimension}) NOT NULL
            )
            """
        )
    )
    session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS document_embedding_index (
                id INT PRIMARY KEY CHECK (id = 1),
                embedding_model VARCHAR(128) NOT NULL,
                dimension INT NOT NULL
            )
            """
        )
    )
    session.execute(
        text(
            """
            INSERT INTO document_embedding_index (id, embedding_model, dimension)
            VALUES (1, :model, :dim)
            ON CONFLICT (id) DO UPDATE SET
                embedding_model = EXCLUDED.embedding_model,
                dimension = EXCLUDED.dimension
            """
        ),
        {"model": embedding_model, "dim": dimension},
    )
    session.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_document_embeddings_hnsw
            ON document_embeddings
            USING hnsw (embedding vector_cosine_ops)
            """
        )
    )
    session.commit()


def sync_document_embeddings(
    session: Session,
    provider: LLMProvider,
    *,
    settings: Settings | None = None,
) -> int:
    """Upsert embeddings for changed chunks. Returns count of newly embedded chunks."""
    settings = settings or get_settings()
    chunks = load_chunks()
    model = embedding_model_name(settings, provider)

    if not chunks:
        if _table_exists(session, "document_embeddings"):
            session.execute(text("DELETE FROM document_embeddings"))
            session.commit()
        return 0

    sample = provider.embed([chunks[0].text])[0]
    dimension = len(sample)
    _ensure_embeddings_table(session, dimension=dimension, embedding_model=model)

    existing_rows = session.execute(
        text(
            "SELECT chunk_id, content_hash, embedding_model FROM document_embeddings"
        )
    ).all()
    existing = {str(r[0]): (str(r[1]), str(r[2])) for r in existing_rows}

    current_ids = {c.id for c in chunks}
    for chunk_id in set(existing.keys()) - current_ids:
        session.execute(
            text("DELETE FROM document_embeddings WHERE chunk_id = :id"),
            {"id": chunk_id},
        )

    stale: list[Chunk] = []
    for chunk in chunks:
        prev = existing.get(chunk.id)
        h = content_hash(chunk.text)
        if prev and prev[0] == h and prev[1] == model:
            continue
        stale.append(chunk)

    if not stale:
        session.commit()
        return 0

    vectors = provider.embed([c.text for c in stale])
    if len(vectors) != len(stale):
        raise RuntimeError("Embedding API returned unexpected batch size")

    for chunk, vec in zip(stale, vectors, strict=True):
        if len(vec) != dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch for {chunk.id}: expected {dimension}, got {len(vec)}"
            )
        vec_literal = "[" + ",".join(str(float(x)) for x in vec) + "]"
        session.execute(
            text(
                """
                INSERT INTO document_embeddings (
                    chunk_id, title, source, text, content_hash, embedding_model, embedding
                )
                VALUES (
                    :chunk_id, :title, :source, :text, :content_hash, :embedding_model,
                    CAST(:embedding AS vector)
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source = EXCLUDED.source,
                    text = EXCLUDED.text,
                    content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding = EXCLUDED.embedding
                """
            ),
            {
                "chunk_id": chunk.id,
                "title": chunk.title,
                "source": chunk.source,
                "text": chunk.text,
                "content_hash": content_hash(chunk.text),
                "embedding_model": model,
                "embedding": vec_literal,
            },
        )

    session.commit()
    return len(stale)


def load_embeddings_from_db(session: Session) -> tuple[list[Chunk], np.ndarray]:
    rows = session.execute(
        text(
            """
            SELECT chunk_id, title, source, text, embedding::text AS embedding
            FROM document_embeddings
            ORDER BY chunk_id
            """
        )
    ).all()
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)

    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for row in rows:
        chunks.append(
            Chunk(
                id=str(row[0]),
                title=str(row[1]),
                source=str(row[2]),
                text=str(row[3]),
            )
        )
        raw = str(row[4]).strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1]
            if not inner.strip():
                vec: list[float] = []
            else:
                vec = [float(x) for x in inner.split(",")]
        else:
            raise ValueError(f"Unexpected embedding format for chunk {row[0]}")
        vectors.append(vec)

    matrix = np.asarray(vectors, dtype=np.float32)
    return chunks, matrix
