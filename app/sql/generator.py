"""Generate SQL from natural language using the configured LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from app.llm.prompts import SQL_SYSTEM_PROMPT, build_sql_user_prompt
from app.llm.provider import LLMProvider, get_llm_provider
from app.sql.validator import SQLValidationError, validate_select_only


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


def _extract_sql_fallback(raw: str) -> str:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    match = re.search(
        r"((?:WITH|SELECT)\b[\s\S]+)",
        raw,
        re.IGNORECASE,
    )
    if not match:
        return ""
    sql = match.group(1).strip()
    # Trim trailing prose after the first statement
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    return sql.strip()


def generate_sql(
    question: str,
    business_context: str = "",
    provider: LLMProvider | None = None,
) -> dict[str, str]:
    llm = provider or get_llm_provider()
    messages = [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_sql_user_prompt(question, business_context),
        },
    ]
    raw = llm.chat(messages, temperature=0.0, max_tokens=800)
    data = _extract_json(raw)
    sql = str(data.get("sql", "")).strip()
    explanation = str(data.get("explanation", "")).strip()
    if not sql:
        sql = _extract_sql_fallback(raw)
    if not sql:
        raise ValueError(f"LLM did not return SQL: {raw[:300]}")
    sql = validate_select_only(sql)
    return {"sql": sql, "explanation": explanation, "raw": raw}