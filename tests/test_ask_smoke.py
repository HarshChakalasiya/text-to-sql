"""End-to-end ask() smoke test with a deterministic fake LLM."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.models import Base
from app.db.seed import seed_database
from app.llm.provider import LLMProvider
from app.retrieval.embeddings import reset_vector_store
from app.services.ask import ask
from app.sql.validator import validate_select_only


class FakeLLM(LLMProvider):
    def __init__(self, name: str = "openai") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return "fake-model"

    def chat(self, messages, *, temperature=0.0, max_tokens=1024) -> str:
        joined = " ".join(m["content"] for m in messages).lower()
        if "classify the user question" in joined:
            if "what is invoice discounting" in joined:
                return json.dumps({"intent": "docs", "reason": "definition"})
            return json.dumps({"intent": "sql", "reason": "analytics"})
        if "generate a single read-only select" in joined or "text-to-sql" in joined:
            return json.dumps(
                {
                    "sql": (
                        "SELECT invoice_number, amount, status FROM invoices "
                        "WHERE status = 'overdue' AND amount > 100000 "
                        "ORDER BY amount DESC LIMIT 20"
                    ),
                    "explanation": "List large overdue invoices",
                }
            )
        return "Invoice discounting advances cash to sellers against approved invoices."

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text_item in texts:
            rng = np.random.default_rng(abs(hash(text_item)) % (2**32))
            vec = rng.normal(size=32).astype(np.float32)
            vec /= max(float(np.linalg.norm(vec)), 1e-12)
            vectors.append(vec.tolist())
        return vectors


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path}/scf.db")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    seed_database(session, seed=1)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_ask_sql_path(db_session, monkeypatch):
    reset_vector_store()
    # SQLite has no statement_timeout; patch executor timeout set
    monkeypatch.setattr(
        "app.sql.executor.get_settings",
        lambda: type("S", (), {"sql_query_timeout_seconds": 10})(),
    )

    original_execute = db_session.execute

    def safe_execute(statement, *args, **kwargs):
        sql = str(statement)
        if "statement_timeout" in sql:
            return original_execute(text("SELECT 1"))
        return original_execute(statement, *args, **kwargs)

    monkeypatch.setattr(db_session, "execute", safe_execute)

    result = ask(
        db_session,
        "Show overdue invoices above 100000",
        provider=FakeLLM("openai"),
    )
    assert result["provider"] == "openai"
    assert result["sql"]
    validate_select_only(result["sql"])
    assert result["rows"] is not None
    assert result["answer"]


def test_ask_docs_path(db_session):
    reset_vector_store()
    result = ask(
        db_session,
        "What is invoice discounting in this system?",
        provider=FakeLLM("ollama"),
    )
    assert result["provider"] == "ollama"
    assert result["intent"] == "docs"
    assert result["sources"]
    assert "discount" in result["answer"].lower() or "invoice" in result["answer"].lower()