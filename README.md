# SCF Text-to-SQL + Hybrid Search

Natural-language analytics over a synthetic **Supply Chain Finance (SCF)** database, with:

- **Text-to-SQL** — schema-aware NL → PostgreSQL `SELECT` generation
- **Hybrid search** — BM25 + dense embeddings fused with Reciprocal Rank Fusion (RRF)
- **Three LLM backends** — switch with `LLM_PROVIDER=openai`, `ollama`, or `claude`

## Quick start

```bash
# 1) Start Postgres (host port 5433 → container 5432)
docker compose up -d

# 2) Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set LLM_PROVIDER and provider credentials/models

# 3) Schema + seed data
python scripts/init_db.py

# 4) Index business docs into Postgres (pgvector; requires LLM embedding API)
python scripts/index_docs.py

# 5) Run API + UI
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | SQLAlchemy URL for Postgres (default host port **5433**) |
| `LLM_PROVIDER` | `openai`, `ollama`, or `claude` |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` / `OPENAI_EMBEDDING_MODEL` | Used when `LLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_EMBEDDING_MODEL` | Used when `LLM_PROVIDER=ollama`. Embeddings are also used when `LLM_PROVIDER=claude` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` | Used when `LLM_PROVIDER=claude` |
| `VECTOR_STORE` | `pgvector` (default) persists doc embeddings in Postgres; `memory` skips DB storage |

### OpenAI

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Ollama

```bash
ollama pull llama3.2:1b
ollama pull nomic-embed-text
```

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### Claude

Anthropic does not provide embeddings. Chat goes to Claude; dense search still uses `OLLAMA_EMBEDDING_MODEL`, so Ollama must be running with that model pulled.

```bash
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-haiku-4-5
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

Restart the app after changing `.env` (provider is cached at first use).

## API

- `GET /` — simple query UI
- `GET /health` — DB + LLM readiness
- `GET /schema` — SCF schema description
- `POST /ask` — `{ "question": "..." }` → answer, optional SQL/rows, hybrid sources

## Safety

Generated SQL is validated to a single read-only `SELECT` / `WITH ... SELECT`. DDL/DML and multi-statements are rejected. Statement timeout is applied per query.

## Project layout

See `app/` for API, LLM adapters, SQL pipeline, and hybrid retrieval; `app/retrieval/docs/` for SCF business context used by RAG.

### Document embeddings (pgvector)

Postgres runs the [pgvector](https://github.com/pgvector/pgvector) image (`docker-compose.yml`). Markdown chunks under `app/retrieval/docs/` are embedded via your configured embedding model and stored in `document_embeddings`. Re-run `python scripts/index_docs.py` after editing docs or changing `OPENAI_EMBEDDING_MODEL` / `OLLAMA_EMBEDDING_MODEL` (the index is rebuilt when the model or vector dimension changes). At runtime, `get_vector_store()` syncs only chunks whose content hash changed, then loads vectors from Postgres into an in-memory matrix for cosine search (hybrid RRF unchanged).

If you previously used plain `postgres:16-alpine`, recreate the volume so the extension is available: `docker compose down -v && docker compose up -d`.