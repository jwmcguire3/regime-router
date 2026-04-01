import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.cli import main


class RecordingOpenAIClient:
    init_calls = []

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: int = 120) -> None:
        RecordingOpenAIClient.init_calls.append({"api_key": api_key, "base_url": base_url, "timeout": timeout})

    @classmethod
    def reset(cls) -> None:
        cls.init_calls.clear()

    def list_models(self):
        return {"provider": "openai", "models": ["gpt-5.4-mini"]}


class RecordingOllamaClient:
    init_calls = []

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300) -> None:
        RecordingOllamaClient.init_calls.append({"base_url": base_url, "timeout": timeout})

    @classmethod
    def reset(cls) -> None:
        cls.init_calls.clear()

    def list_models(self):
        return {"provider": "ollama", "models": ["dolphin29:latest"]}


def test_models_with_openai_provider_uses_openai_client(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("router.runtime.OpenAIModelClient", RecordingOpenAIClient)
    monkeypatch.setattr("router.runtime.OllamaModelClient", RecordingOllamaClient)
    monkeypatch.setenv("TEST_OPENAI_KEY", "integration-test-key")
    RecordingOpenAIClient.reset()
    RecordingOllamaClient.reset()

    settings_file = tmp_path / "settings.json"
    main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--provider",
            "openai",
            "--openai-api-key-env",
            "TEST_OPENAI_KEY",
            "--openai-base-url",
            "https://api.openai.com/v1",
        ]
    )
    capsys.readouterr()

    rc = main(["--settings-file", str(settings_file), "models"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["provider"] == "openai"
    assert RecordingOpenAIClient.init_calls[-1]["api_key"] == "integration-test-key"
    assert RecordingOllamaClient.init_calls == []


def test_models_with_ollama_provider_uses_ollama_client(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("router.runtime.OpenAIModelClient", RecordingOpenAIClient)
    monkeypatch.setattr("router.runtime.OllamaModelClient", RecordingOllamaClient)
    RecordingOpenAIClient.reset()
    RecordingOllamaClient.reset()

    settings_file = tmp_path / "settings.json"
    main(["--settings-file", str(settings_file), "settings", "set", "--provider", "ollama"])
    capsys.readouterr()

    rc = main(["--settings-file", str(settings_file), "models"])
    payload = json.loads(capsys.readouterr().out)

    assert rc == 0
    assert payload["provider"] == "ollama"
    assert RecordingOllamaClient.init_calls[-1]["base_url"] == "http://localhost:11434"
    assert RecordingOpenAIClient.init_calls == []


def test_models_command_routes_to_active_provider(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("router.runtime.OpenAIModelClient", RecordingOpenAIClient)
    monkeypatch.setattr("router.runtime.OllamaModelClient", RecordingOllamaClient)
    monkeypatch.setenv("TEST_OPENAI_KEY", "integration-test-key")
    RecordingOpenAIClient.reset()
    RecordingOllamaClient.reset()

    settings_file = tmp_path / "settings.json"

    main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--provider",
            "openai",
            "--openai-api-key-env",
            "TEST_OPENAI_KEY",
        ]
    )
    capsys.readouterr()
    rc_openai = main(["--settings-file", str(settings_file), "models"])
    payload_openai = json.loads(capsys.readouterr().out)

    main(["--settings-file", str(settings_file), "settings", "set", "--provider", "ollama"])
    capsys.readouterr()
    rc_ollama = main(["--settings-file", str(settings_file), "models"])
    payload_ollama = json.loads(capsys.readouterr().out)

    assert rc_openai == 0
    assert rc_ollama == 0
    assert payload_openai["provider"] == "openai"
    assert payload_ollama["provider"] == "ollama"


def test_openai_provider_without_key_fails_with_clear_error(tmp_path, capsys):
    settings_file = tmp_path / "settings.json"

    main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--provider",
            "openai",
            "--openai-api-key-env",
            "MISSING_OPENAI_KEY",
        ]
    )
    capsys.readouterr()

    rc = main(["--settings-file", str(settings_file), "models"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "OpenAI provider requires environment variable 'MISSING_OPENAI_KEY'" in captured.err
