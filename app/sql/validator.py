"""Validate that generated SQL is a single read-only SELECT."""

from __future__ import annotations

import re

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|"
    r"COPY|CALL|EXECUTE|MERGE|REPLACE|ATTACH|DETACH|VACUUM|ANALYZE|"
    r"COMMENT|SECURITY|OWNER|INTO\s+OUTFILE|LOAD\s+DATA)\b",
    re.IGNORECASE,
)


class SQLValidationError(ValueError):
    pass


def validate_select_only(sql: str) -> str:
    if not sql or not sql.strip():
        raise SQLValidationError("SQL is empty")

    cleaned = sql.strip().rstrip(";")
    if ";" in cleaned:
        raise SQLValidationError("Multiple SQL statements are not allowed")

    # Strip simple line comments for checks
    no_comments = re.sub(r"--.*?$", "", cleaned, flags=re.MULTILINE)
    no_comments = re.sub(r"/\*.*?\*/", "", no_comments, flags=re.DOTALL).strip()

    if not re.match(r"^(WITH|SELECT)\b", no_comments, re.IGNORECASE):
        raise SQLValidationError("Only SELECT (or WITH ... SELECT) queries are allowed")

    if FORBIDDEN.search(no_comments):
        raise SQLValidationError("Query contains forbidden keywords")

    # Block SELECT ... INTO
    if re.search(r"\bINTO\b", no_comments, re.IGNORECASE) and not re.search(
        r"\bINSERT\b", no_comments, re.IGNORECASE
    ):
        # SELECT INTO is dangerous; also catch CTAS-like patterns
        if re.search(r"\bSELECT\b.+\bINTO\b", no_comments, re.IGNORECASE | re.DOTALL):
            raise SQLValidationError("SELECT INTO is not allowed")

    return cleaned