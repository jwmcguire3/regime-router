import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.llm.ollama_client import OllamaModelClient
from router.llm.openai_client import OpenAIModelClient
from router.runtime import create_model_client


def test_create_model_client_openai_uses_openai_client_and_base_url(monkeypatch):
    monkeypatch.setenv("MY_OPENAI_KEY", "test-secret")

    client = create_model_client(
        provider="openai",
        ollama_base_url="http://localhost:11434",
        openai_base_url="https://example-openai/v1",
        openai_api_key_env="MY_OPENAI_KEY",
    )

    assert isinstance(client, OpenAIModelClient)
    assert client.base_url == "https://example-openai/v1"
    assert client.api_key == "test-secret"


def test_create_model_client_ollama_uses_ollama_client():
    client = create_model_client(
        provider="ollama",
        ollama_base_url="http://localhost:3344",
        openai_base_url="https://unused.example/v1",
        openai_api_key_env="UNUSED_KEY",
    )

    assert isinstance(client, OllamaModelClient)
    assert client.base_url == "http://localhost:3344"


def test_create_model_client_openai_missing_env_var_fails_with_clear_message(monkeypatch):
    monkeypatch.delenv("MISSING_OPENAI_ENV", raising=False)

    with pytest.raises(RuntimeError, match="MISSING_OPENAI_ENV"):
        create_model_client(
            provider="openai",
            ollama_base_url="http://localhost:11434",
            openai_base_url="https://api.openai.com/v1",
            openai_api_key_env="MISSING_OPENAI_ENV",
        )
