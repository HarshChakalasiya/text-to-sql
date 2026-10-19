import pytest

from app.config import Settings
from app.llm.provider import (
    ClaudeProvider,
    OllamaProvider,
    OpenAIProvider,
    build_llm_provider,
)


def test_build_openai_provider():
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.name == "openai"
    assert provider.model == "gpt-4o-mini"


def test_build_ollama_provider():
    settings = Settings(
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.2:1b",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.name == "ollama"
    assert provider.model == "llama3.2:1b"


def test_openai_requires_api_key():
    settings = Settings(llm_provider="openai", openai_api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm_provider(settings)


def test_build_claude_provider():
    settings = Settings(
        llm_provider="claude",
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-haiku-4-5",
    )
    provider = build_llm_provider(settings)
    assert isinstance(provider, ClaudeProvider)
    assert provider.name == "claude"
    assert provider.model == "claude-haiku-4-5"


def test_claude_requires_api_key():
    settings = Settings(llm_provider="claude", anthropic_api_key="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        build_llm_provider(settings)


def test_claude_chat_sends_system_separately(monkeypatch):
    settings = Settings(
        llm_provider="claude",
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-haiku-4-5",
    )
    provider = ClaudeProvider(settings)
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"content": [{"type": "text", "text": "SELECT 1"}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.llm.provider.httpx.Client", FakeClient)
    assert provider.chat(
        [
            {"role": "system", "content": "Return SQL"},
            {"role": "user", "content": "count invoices"},
        ],
        temperature=0.0,
        max_tokens=80,
    ) == "SELECT 1"
    assert captured["json"]["system"] == "Return SQL"
    assert captured["json"]["messages"] == [
        {"role": "user", "content": "count invoices"}
    ]
    assert captured["json"]["temperature"] == 0.0
    assert captured["headers"]["x-api-key"] == "sk-ant-test"


def test_claude_5_omits_temperature(monkeypatch):
    settings = Settings(
        llm_provider="claude",
        anthropic_api_key="sk-ant-test",
        anthropic_model="claude-haiku-5-5",
    )
    provider = ClaudeProvider(settings)
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"content": [{"type": "text", "text": "ok"}]}

    class FakeClient:
        def __init__(self, timeout: float) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url, headers, json):
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr("app.llm.provider.httpx.Client", FakeClient)
    provider.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert "temperature" not in captured["json"]