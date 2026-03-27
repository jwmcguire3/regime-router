import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.cli import main


def _pwsh() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


@pytest.mark.skipif(_pwsh() is None, reason="PowerShell is not available in this environment")
def test_wrapper_settings_show_set_reset(tmp_path):
    pwsh = _pwsh()
    assert pwsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "router.ps1"
    settings_file = tmp_path / "wrapper-settings.json"

    set_result = subprocess.run(
        [
            pwsh,
            "-NoProfile",
            "-File",
            str(script),
            "settings-set",
            "-SettingsFile",
            str(settings_file),
            "-Model",
            "qwen3",
            "-DebugRouting",
        ],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    set_payload = json.loads(set_result.stdout)
    assert set_payload["settings"]["model"] == "qwen3"
    assert set_payload["settings"]["debug_routing"] is True

    show_result = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(script), "settings-show", "-SettingsFile", str(settings_file)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    show_payload = json.loads(show_result.stdout)
    assert show_payload["settings"]["model"] == "qwen3"

    reset_result = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(script), "settings-reset", "-SettingsFile", str(settings_file)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    reset_payload = json.loads(reset_result.stdout)
    assert reset_payload["settings"]["model"] == "llama3"


@pytest.mark.skipif(_pwsh() is None, reason="PowerShell is not available in this environment")
def test_wrapper_settings_set_does_not_override_existing_defaults_without_flags(tmp_path):
    pwsh = _pwsh()
    assert pwsh is not None
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "router.ps1"
    settings_file = tmp_path / "wrapper-settings.json"

    main(["--settings-file", str(settings_file), "settings", "set", "--model", "qwen3", "--max-switches", "7"])

    subprocess.run(
        [pwsh, "-NoProfile", "-File", str(script), "settings-set", "-SettingsFile", str(settings_file)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    show_result = subprocess.run(
        [pwsh, "-NoProfile", "-File", str(script), "settings-show", "-SettingsFile", str(settings_file)],
        check=True,
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    payload = json.loads(show_result.stdout)
    assert payload["settings"]["model"] == "qwen3"
    assert payload["settings"]["max_switches"] == 7
