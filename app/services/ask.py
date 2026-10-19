"""Orchestrate intent routing, hybrid search, and Text-to-SQL."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy.orm import Session

from app.llm.prompts import (
    ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    build_rag_user_prompt,
)
from app.llm.provider import LLMProvider, get_llm_provider
from app.retrieval.hybrid import format_sources, hybrid_search
from app.sql.executor import execute_select
from app.sql.generator import generate_sql
from app.sql.validator import SQLValidationError


def _extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def classify_intent(question: str, provider: LLMProvider) -> str:
    q = question.lower()
    doc_hints = (
        "what is",
        "explain",
        "define",
        "meaning of",
        "how does",
        "glossary",
        "difference between",
    )
    sql_hints = (
        "how many",
        "total",
        "sum",
        "average",
        "top",
        "list",
        "show",
        "count",
        "overdue",
        "amount",
        "by buyer",
        "by seller",
    )
    has_doc = any(_has_phrase(q, h) for h in doc_hints)
    has_sql = any(_has_phrase(q, h) for h in sql_hints)
    # Strong heuristic wins for clear definition-only questions
    if has_doc and not has_sql:
        return "docs"
    if has_sql and not has_doc:
        return "sql"

    try:
        raw = provider.chat(
            [
                {"role": "system", "content": INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        data = _extract_json(raw)
        intent = str(data.get("intent", "")).lower()
        if intent in {"sql", "docs", "both"}:
            return intent
    except Exception:
        pass

    if has_doc and has_sql:
        return "both"
    if has_doc:
        return "docs"
    if has_sql:
        return "sql"
    return "both"


def _serialize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for k, v in row.items():
            if hasattr(v, "isoformat"):
                item[k] = v.isoformat()
            elif hasattr(v, "__float__") and type(v).__name__ == "Decimal":
                item[k] = float(v)
            else:
                item[k] = v
        out.append(item)
    return out


def ask(
    db: Session,
    question: str,
    *,
    provider: LLMProvider | None = None,
) -> dict[str, Any]:
    llm = provider or get_llm_provider()
    intent = classify_intent(question, llm)

    sources: list[dict[str, str]] = []
    sql: str | None = None
    explanation: str | None = None
    rows: list[dict[str, Any]] | None = None
    error: str | None = None

    # Always retrieve a bit of business context for SQL quality
    try:
        hybrid_hits = hybrid_search(question, provider=llm)
        sources = format_sources(hybrid_hits)
    except Exception as exc:  # embeddings may be unavailable
        error = f"Hybrid search unavailable: {exc}"

    business_context = "\n\n".join(
        f"{s['title']}: {s['text']}" for s in sources[:3]
    )

    if intent in {"sql", "both"}:
        try:
            generated = generate_sql(question, business_context, provider=llm)
            sql = generated["sql"]
            explanation = generated.get("explanation")
            rows = _serialize_rows(execute_select(db, sql))
        except SQLValidationError as exc:
            error = f"SQL validation failed: {exc}"
        except Exception as exc:
            error = f"SQL generation/execution failed: {exc}"

    answer: str
    if intent == "docs" and not rows:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_rag_user_prompt(question, sources),
            },
        ]
        answer = llm.chat(messages, temperature=0.2, max_tokens=700)
    elif rows is not None or sources:
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_rag_user_prompt(
                    question, sources, sql=sql, rows=rows
                ),
            },
        ]
        try:
            answer = llm.chat(messages, temperature=0.2, max_tokens=700)
        except Exception as exc:
            answer = explanation or "Query completed but answer synthesis failed."
            error = (error + "; " if error else "") + f"Answer synthesis failed: {exc}"
    else:
        answer = error or "Unable to answer the question."

    return {
        "question": question,
        "intent": intent,
        "answer": answer,
        "sql": sql,
        "explanation": explanation,
        "rows": rows,
        "sources": [
            {
                "id": s["id"],
                "title": s["title"],
                "source": s["source"],
                "score": s["score"],
            }
            for s in sources
        ],
        "error": error,
        "provider": llm.name,
        "model": llm.model,
    }