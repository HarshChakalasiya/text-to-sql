from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(
    title="SCF Text-to-SQL + Hybrid Search",
    description=(
        "Natural language querying over a Supply Chain Finance database "
        "with hybrid retrieval (BM25 + dense RAG)."
    ),
    version="1.0.0",
)
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )