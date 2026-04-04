import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.settings import CliSettings, UserSettings, default_model_for_provider


def test_default_model_for_provider_includes_deepseek():
    assert default_model_for_provider("ollama") == "dolphin29:latest"
    assert default_model_for_provider("openai") == "gpt-5.4-mini"
    assert default_model_for_provider("deepseek") == "deepseek-reasoner"


def test_fresh_user_and_cli_settings_default_to_deepseek():
    user = UserSettings()
    cli = CliSettings()

    assert user.provider == "deepseek"
    assert user.model == "deepseek-reasoner"
    assert user.openai_base_url == "https://api.deepseek.com"
    assert user.openai_api_key_env == "DEEPSEEK_API_KEY"
    assert user.task_analyzer_model == "deepseek-reasoner"

    assert cli.user.provider == "deepseek"
    assert cli.user.model == "deepseek-reasoner"


def test_user_settings_from_dict_accepts_deepseek_provider_defaults():
    parsed = UserSettings.from_dict({"provider": "deepseek"})

    assert parsed.provider == "deepseek"
    assert parsed.model == "deepseek-reasoner"
    assert parsed.task_analyzer_model == "deepseek-reasoner"


def test_old_flat_settings_json_still_loads():
    parsed = CliSettings.from_dict(
        {
            "provider": "openai",
            "model": "gpt-5.4-mini",
            "openai_base_url": "https://api.openai.com/v1",
            "openai_api_key_env": "OPENAI_API_KEY",
            "use_task_analyzer": False,
            "task_analyzer_model": "gpt-5.4-mini",
            "model_profile": "balanced",
        }
    )

    assert parsed.user.provider == "openai"
    assert parsed.user.model == "gpt-5.4-mini"
    assert parsed.user.openai_base_url == "https://api.openai.com/v1"
    assert parsed.user.openai_api_key_env == "OPENAI_API_KEY"
    assert parsed.user.use_task_analyzer is False
    assert parsed.user.task_analyzer_model == "gpt-5.4-mini"
    assert parsed.model_controls.model_profile == "balanced"
