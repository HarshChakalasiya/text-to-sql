"""LLM provider abstraction: OpenAI, Ollama, or Claude via LLM_PROVIDER."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import httpx
from openai import OpenAI

from app.config import Settings, get_settings


class LLMProvider(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )

    @property
    def name(self) -> str:
        return "openai"

    @property
    def model(self) -> str:
        return self._settings.openai_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=texts,
        )
        # API may return out of order; sort by index
        data = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in data]


class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.ollama_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._settings.ollama_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._settings.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(f"{self._base}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        return (data.get("message") or {}).get("content") or ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120.0) as client:
            for text in texts:
                response = client.post(
                    f"{self._base}/api/embeddings",
                    json={
                        "model": self._settings.ollama_embedding_model,
                        "prompt": text,
                    },
                )
                response.raise_for_status()
                vectors.append(list(response.json()["embedding"]))
        return vectors


def _claude_accepts_temperature(model: str) -> bool:
    # Claude 5.x models reject temperature, top_p, and top_k.
    return re.search(r"-(?:haiku|sonnet|opus|fable)-5", model) is None


class ClaudeProvider(LLMProvider):
    """Chat via the Anthropic Messages API.

    Anthropic has no embeddings endpoint. Dense retrieval uses the configured
    Ollama embedding model (OLLAMA_EMBEDDING_MODEL).
    """

    def __init__(self, settings: Settings) -> None:
        if not settings.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=claude"
            )
        self._settings = settings
        self._base = settings.anthropic_base_url.rstrip("/")
        self._embedder = OllamaProvider(settings)

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model(self) -> str:
        return self._settings.anthropic_model

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                chat_messages.append(
                    {"role": message["role"], "content": message["content"]}
                )
        if not chat_messages:
            chat_messages = [{"role": "user", "content": ""}]

        payload: dict[str, Any] = {
            "model": self._settings.anthropic_model,
            "max_tokens": max_tokens,
            "messages": chat_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if _claude_accepts_temperature(self._settings.anthropic_model):
            payload["temperature"] = temperature

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self._base}/v1/messages",
                headers={
                    "x-api-key": self._settings.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        blocks = data.get("content") or []
        return "".join(
            block.get("text", "")
            for block in blocks
            if block.get("type") == "text"
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)


def build_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings)
    if settings.llm_provider == "openai":
        return OpenAIProvider(settings)
    if settings.llm_provider == "claude":
        return ClaudeProvider(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    return build_llm_provider()


def reset_llm_provider_cache() -> None:
    get_llm_provider.cache_clear()