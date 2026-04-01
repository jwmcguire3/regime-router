import json
import sys
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from router.llm.openai_client import OpenAIModelClient


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_generate_constructs_correct_request_payload() -> None:
    client = OpenAIModelClient(api_key="test-key", base_url="https://example.com/v1", timeout=42)

    with patch("router.llm.openai_client.urlopen", return_value=_FakeResponse({"choices": [{"message": {"content": "ok"}}]})) as mock_urlopen:
        client.generate(
            model="gpt-test",
            system="You are helpful",
            prompt="Say hi",
            temperature=0.7,
            num_predict=321,
        )

    req = mock_urlopen.call_args.args[0]
    assert req.full_url == "https://example.com/v1/chat/completions"
    assert req.get_method() == "POST"
    assert req.headers["Authorization"] == "Bearer test-key"
    assert req.headers["Content-type"] == "application/json"

    payload = json.loads(req.data.decode("utf-8"))
    assert payload == {
        "model": "gpt-test",
        "temperature": 0.7,
        "max_completion_tokens": 321,
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Say hi"},
        ],
    }
    assert mock_urlopen.call_args.kwargs["timeout"] == 42


def test_generate_maps_response_to_ollama_style() -> None:
    client = OpenAIModelClient(api_key="test-key")

    with patch(
        "router.llm.openai_client.urlopen",
        return_value=_FakeResponse({"choices": [{"message": {"content": "Hello from model"}}]}),
    ):
        result = client.generate(model="gpt-test", system="s", prompt="p")

    assert result == {"response": "Hello from model"}


def test_generate_wraps_http_error() -> None:
    client = OpenAIModelClient(api_key="test-key")
    error = HTTPError(
        url="https://api.example.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=BytesIO(b'{"error":"invalid key"}'),
    )

    with patch("router.llm.openai_client.urlopen", side_effect=error):
        with pytest.raises(RuntimeError, match=r"HTTP error 401: .*invalid key"):
            client.generate(model="gpt-test", system="s", prompt="p")


def test_list_models_calls_correct_endpoint() -> None:
    client = OpenAIModelClient(api_key="test-key", base_url="https://example.com/v1", timeout=15)

    with patch(
        "router.llm.openai_client.urlopen",
        return_value=_FakeResponse({"data": [{"id": "gpt-4o-mini"}]}),
    ) as mock_urlopen:
        result = client.list_models()

    req = mock_urlopen.call_args.args[0]
    assert req.full_url == "https://example.com/v1/models"
    assert req.get_method() == "GET"
    assert req.headers["Authorization"] == "Bearer test-key"
    assert mock_urlopen.call_args.kwargs["timeout"] == 15
    assert result == {"data": [{"id": "gpt-4o-mini"}]}
