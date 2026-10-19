from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import SCHEMA_DESCRIPTION
from app.db.session import get_db
from app.llm.provider import get_llm_provider
from app.services.ask import ask

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class AskResponse(BaseModel):
    question: str
    intent: str
    answer: str
    sql: str | None = None
    explanation: str | None = None
    rows: list[dict[str, Any]] | None = None
    sources: list[dict[str, str]] = []
    error: str | None = None
    provider: str
    model: str


@router.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    settings = get_settings()
    db_ok = False
    db_error = None
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as exc:
        db_error = str(exc)

    llm_info: dict[str, Any] = {"provider": settings.llm_provider}
    try:
        llm = get_llm_provider()
        llm_info["model"] = llm.model
        llm_info["ready"] = True
    except Exception as exc:
        llm_info["ready"] = False
        llm_info["error"] = str(exc)

    return {
        "status": "ok" if db_ok else "degraded",
        "database": {"ok": db_ok, "error": db_error},
        "llm": llm_info,
    }


@router.get("/schema")
def schema() -> dict[str, str]:
    return {"schema": SCHEMA_DESCRIPTION}


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, db: Session = Depends(get_db)) -> AskResponse:
    try:
        result = ask(db, body.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AskResponse(**result)