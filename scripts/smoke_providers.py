#!/usr/bin/env python3
"""Smoke-test provider switching and optional live /ask calls."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.llm.provider import (  # noqa: E402
    ClaudeProvider,
    OllamaProvider,
    OpenAIProvider,
    build_llm_provider,
)


def test_switch() -> None:
    openai = build_llm_provider(
        Settings(llm_provider="openai", openai_api_key="sk-test")
    )
    ollama = build_llm_provider(Settings(llm_provider="ollama"))
    claude = build_llm_provider(
        Settings(llm_provider="claude", anthropic_api_key="sk-ant-test")
    )
    assert isinstance(openai, OpenAIProvider)
    assert isinstance(ollama, OllamaProvider)
    assert isinstance(claude, ClaudeProvider)
    print("OK provider switch: openai ->", openai.name, openai.model)
    print("OK provider switch: ollama ->", ollama.name, ollama.model)
    print("OK provider switch: claude ->", claude.name, claude.model)


def test_openai_chat_mocked() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="sk-test")
    provider = OpenAIProvider(settings)
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    with patch.object(provider._client.chat.completions, "create", return_value=fake_response):
        assert provider.chat([{"role": "user", "content": "hi"}]) == "hello"
    print("OK openai chat (mocked)")


def test_ollama_live_if_available() -> None:
    import httpx

    settings = Settings(llm_provider="ollama")
    try:
        tags = httpx.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags", timeout=5).json()
    except Exception as exc:
        print("SKIP ollama live:", exc)
        return
    names = {m.get("name", "") for m in tags.get("models", [])}
    # Prefer configured model, else any chat-ish model that is not only embeddings
    model = settings.ollama_model
    if model not in names and f"{model}:latest" not in names:
        candidates = [n for n in names if "embed" not in n]
        if not candidates:
            print("SKIP ollama live: no chat model pulled yet; have", names)
            return
        model = candidates[0]
        settings = Settings(llm_provider="ollama", ollama_model=model)

    provider = OllamaProvider(settings)
    text = provider.chat(
        [{"role": "user", "content": "Reply with exactly: pong"}],
        max_tokens=16,
    )
    print("OK ollama live chat via", provider.model, "->", repr(text[:120]))

    emb = provider.embed(["supply chain finance invoice"])
    print("OK ollama live embed dim", len(emb[0]) if emb else 0)


def main() -> None:
    test_switch()
    test_openai_chat_mocked()
    test_ollama_live_if_available()
    print("Smoke providers complete.")


if __name__ == "__main__":
    main()