"""Execute validated read-only SQL against PostgreSQL."""

from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.orm import Session

from app.config import get_settings
from app.sql.validator import validate_select_only


def execute_select(db: Session, sql: str) -> list[dict[str, Any]]:
    safe_sql = validate_select_only(sql)
    timeout_ms = get_settings().sql_query_timeout_seconds * 1000

    db.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
    result: Result = db.execute(text(safe_sql))
    rows = result.mappings().all()
    return [dict(row) for row in rows]