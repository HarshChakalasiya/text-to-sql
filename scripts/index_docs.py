#!/usr/bin/env python3
"""Create pgvector tables and embed app/retrieval/docs into Postgres."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db.session import SessionLocal, engine  # noqa: E402
from app.llm.provider import get_llm_provider  # noqa: E402
from app.retrieval.chunker import load_chunks  # noqa: E402
from app.retrieval.embeddings import reset_vector_store  # noqa: E402
from app.retrieval.indexing import (  # noqa: E402
    ensure_pgvector_extension,
    pgvector_enabled,
    sync_document_embeddings,
)


def main() -> None:
    if not pgvector_enabled(engine):
        print(
            "pgvector storage is disabled (VECTOR_STORE=memory or non-Postgres DATABASE_URL)."
        )
        sys.exit(1)

    ensure_pgvector_extension(engine)
    chunks = load_chunks()
    print(f"Found {len(chunks)} document chunks under app/retrieval/docs/")

    provider = get_llm_provider()
    db = SessionLocal()
    try:
        updated = sync_document_embeddings(db, provider)
        print(f"Embedded {updated} chunk(s) with {provider.name} / {provider.model}.")
    finally:
        db.close()

    reset_vector_store()
    print("Done. In-memory vector cache cleared; next request reloads from Postgres.")


if __name__ == "__main__":
    main()
