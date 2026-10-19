"""Prompt templates for Text-to-SQL and hybrid RAG answers."""

from app.db.models import SCHEMA_DESCRIPTION

SQL_SYSTEM_PROMPT = """You are an expert PostgreSQL Text-to-SQL assistant for a Supply Chain Finance (SCF) database.
Generate a single read-only SELECT query that answers the user's question.

Rules:
- Output ONLY valid JSON with keys: "sql" (string) and "explanation" (short string).
- Use only the tables and columns described in the schema.
- Prefer explicit JOINs; do not invent tables or columns.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT, or multiple statements.
- Use CURRENT_DATE for date comparisons when needed.
- Limit result sets with LIMIT 100 unless the user asks for aggregates only.
- Amounts are Numeric; compare as numbers without currency symbols.
"""


def build_sql_user_prompt(question: str, business_context: str = "") -> str:
    context_block = ""
    if business_context.strip():
        context_block = f"\nBusiness context (may help interpret terms):\n{business_context}\n"
    return (
        f"Schema:\n{SCHEMA_DESCRIPTION}\n"
        f"{context_block}\n"
        f"Question: {question}\n"
        'Respond as JSON: {"sql": "...", "explanation": "..."}'
    )


ANSWER_SYSTEM_PROMPT = """You answer questions about Supply Chain Finance using retrieved documents
and/or SQL query results. Be concise and accurate. Cite source titles when using documents.
If SQL rows are provided, ground numerical claims in those rows. If you cannot answer, say so."""


def build_rag_user_prompt(
    question: str,
    sources: list[dict[str, str]],
    sql: str | None = None,
    rows: list[dict] | None = None,
) -> str:
    parts = [f"Question: {question}\n"]
    if sources:
        parts.append("Retrieved documents:")
        for i, src in enumerate(sources, 1):
            parts.append(f"[{i}] Title: {src.get('title', 'doc')}\n{src.get('text', '')}\n")
    if sql:
        parts.append(f"SQL executed:\n{sql}\n")
    if rows is not None:
        preview = rows[:25]
        parts.append(f"SQL result rows ({len(rows)} total, showing {len(preview)}):\n{preview}\n")
    parts.append("Write a clear answer for a business analyst.")
    return "\n".join(parts)


INTENT_SYSTEM_PROMPT = """Classify the user question for an SCF analytics assistant.
Return ONLY JSON: {"intent": "sql"|"docs"|"both", "reason": "..."}.
- sql: needs numbers/aggregates/filters from the database
- docs: asks for definitions, process explanation, or business concepts
- both: needs domain meaning plus database facts
"""