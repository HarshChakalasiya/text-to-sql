from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://scf:scf@localhost:5433/scf"
    llm_provider: Literal["openai", "ollama", "claude"] = "openai"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_embedding_model: str = "nomic-embed-text"

    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-haiku-4-5"

    sql_query_timeout_seconds: int = 10
    hybrid_top_k: int = 5
    # pgvector persistence for document embeddings; use "memory" for in-process-only index
    vector_store: Literal["pgvector", "memory"] = "pgvector"


@lru_cache
def get_settings() -> Settings:
    return Settings()