import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.cli import main
from router.models import RegimeExecutionResult
from router.routing import Router, RegimeComposer
from router.state import Handoff


class FakeRuntime:
    init_calls = []
    execute_calls = []
    plan_calls = []

    def __init__(self, ollama_base_url: str, use_task_analyzer: bool, task_analyzer_model: str):
        self.router = Router()
        self.composer = RegimeComposer()
        self.router_state = None
        FakeRuntime.init_calls.append(
            {
                "ollama_base_url": ollama_base_url,
                "use_task_analyzer": use_task_analyzer,
                "task_analyzer_model": task_analyzer_model,
            }
        )

    @classmethod
    def reset(cls):
        cls.init_calls.clear()
        cls.execute_calls.clear()
        cls.plan_calls.clear()

    def execute(self, task, model, risk_profile, handoff_expected, bounded_orchestration, max_switches):
        FakeRuntime.execute_calls.append(
            {
                "task": task,
                "model": model,
                "risk_profile": risk_profile,
                "handoff_expected": handoff_expected,
                "bounded_orchestration": bounded_orchestration,
                "max_switches": max_switches,
            }
        )
        decision = self.router.route(task)
        regime = self.composer.compose(decision.primary_regime, risk_profile=risk_profile, handoff_expected=handoff_expected)
        result = RegimeExecutionResult(
            task=task,
            model=model,
            regime_name=regime.name,
            stage=regime.stage,
            system_prompt="sys",
            user_prompt="user",
            raw_response='{"ok": true}',
            artifact_text='{"ok": true}',
            validation={"is_valid": True},
            ollama_meta={},
        )
        handoff = Handoff(
            current_bottleneck=task,
            dominant_frame="frame",
            what_is_known=["known"],
            what_remains_uncertain=["unknown"],
            active_contradictions=[],
            assumptions_in_play=[],
            main_risk_if_continue="risk",
            recommended_next_regime=decision.runner_up_regime,
            minimum_useful_artifact="artifact",
            recommended_next_regime_full=None,
        )
        return decision, regime, result, handoff

    def plan(self, bottleneck, risk_profile, handoff_expected):
        FakeRuntime.plan_calls.append(
            {
                "bottleneck": bottleneck,
                "risk_profile": risk_profile,
                "handoff_expected": handoff_expected,
            }
        )
        decision = self.router.route(bottleneck)
        regime = self.composer.compose(decision.primary_regime, risk_profile=risk_profile, handoff_expected=handoff_expected)
        handoff = Handoff(
            current_bottleneck=bottleneck,
            dominant_frame="frame",
            what_is_known=["known"],
            what_remains_uncertain=["unknown"],
            active_contradictions=[],
            assumptions_in_play=[],
            main_risk_if_continue="risk",
            recommended_next_regime=decision.runner_up_regime,
            minimum_useful_artifact="artifact",
            recommended_next_regime_full=None,
        )
        return decision, regime, handoff


def test_settings_defaults_show(tmp_path, capsys):
    settings_file = tmp_path / "settings.json"

    rc = main(["--settings-file", str(settings_file), "settings", "show"])
    out = capsys.readouterr().out

    payload = json.loads(out)
    assert rc == 0
    assert payload["settings"]["model"] == "llama3"
    assert payload["settings"]["use_task_analyzer"] is True
    assert payload["settings"]["task_analyzer_model"] == "llama3"
    assert payload["settings"]["debug_routing"] is False
    assert payload["settings"]["bounded_orchestration"] is True
    assert payload["settings"]["max_switches"] == 2


def test_settings_set_and_reset(tmp_path, capsys):
    settings_file = tmp_path / "settings.json"

    rc_set = main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--model",
            "qwen3",
            "--no-bounded-orchestration",
            "--max-switches",
            "5",
            "--debug-routing",
        ]
    )
    set_payload = json.loads(capsys.readouterr().out)
    assert rc_set == 0
    assert set_payload["settings"]["model"] == "qwen3"
    assert set_payload["settings"]["bounded_orchestration"] is False
    assert set_payload["settings"]["max_switches"] == 5
    assert set_payload["settings"]["debug_routing"] is True

    rc_reset = main(["--settings-file", str(settings_file), "settings", "reset"])
    reset_payload = json.loads(capsys.readouterr().out)
    assert rc_reset == 0
    assert reset_payload["settings"]["model"] == "llama3"
    assert reset_payload["settings"]["bounded_orchestration"] is True
    assert reset_payload["settings"]["max_switches"] == 2
    assert reset_payload["settings"]["debug_routing"] is False


def test_run_and_plan_use_persisted_defaults(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("router.cli.CognitiveRouterRuntime", FakeRuntime)
    FakeRuntime.reset()
    settings_file = tmp_path / "settings.json"

    main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--model",
            "qwen3",
            "--task-analyzer-model",
            "llama3.2",
            "--debug-routing",
            "--bounded-orchestration",
            "--max-switches",
            "3",
        ]
    )
    capsys.readouterr()

    rc_run = main(
        ["--settings-file", str(settings_file), "--out-dir", str(tmp_path), "run", "--task", "Choose a direction"]
    )
    run_out = capsys.readouterr().out
    rc_plan = main(["--settings-file", str(settings_file), "plan", "--task", "Choose a direction"])
    plan_out = capsys.readouterr().out

    assert rc_run == 0
    assert rc_plan == 0
    assert "=== Debug ===" in plan_out
    assert "Saved run to:" in run_out
    assert FakeRuntime.init_calls[0]["use_task_analyzer"] is True
    assert FakeRuntime.init_calls[0]["task_analyzer_model"] == "llama3.2"
    assert FakeRuntime.execute_calls[0]["model"] == "qwen3"
    assert FakeRuntime.execute_calls[0]["bounded_orchestration"] is True
    assert FakeRuntime.execute_calls[0]["max_switches"] == 3


def test_explicit_run_flags_override_stored_defaults(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("router.cli.CognitiveRouterRuntime", FakeRuntime)
    FakeRuntime.reset()
    settings_file = tmp_path / "settings.json"

    main(
        [
            "--settings-file",
            str(settings_file),
            "settings",
            "set",
            "--model",
            "qwen3",
            "--bounded-orchestration",
            "--max-switches",
            "4",
            "--use-task-analyzer",
        ]
    )
    capsys.readouterr()

    rc = main(
        [
            "--settings-file",
            str(settings_file),
            "--out-dir",
            str(tmp_path),
            "run",
            "--task",
            "Choose a direction",
            "--model",
            "llama3.1",
            "--no-use-task-analyzer",
            "--no-bounded-orchestration",
            "--max-switches",
            "1",
        ]
    )
    capsys.readouterr()

    assert rc == 0
    assert FakeRuntime.init_calls[0]["use_task_analyzer"] is False
    assert FakeRuntime.execute_calls[0]["model"] == "llama3.1"
    assert FakeRuntime.execute_calls[0]["bounded_orchestration"] is False
    assert FakeRuntime.execute_calls[0]["max_switches"] == 1


def test_plan_output_sections_compact_mode(monkeypatch, capsys):
    monkeypatch.setattr("router.cli.CognitiveRouterRuntime", FakeRuntime)
    FakeRuntime.reset()

    rc = main(["--output", "compact", "plan", "--task", "Choose a direction", "--no-use-task-analyzer"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== Routing summary ===" in out
    assert "=== Regime output ===" in out
    assert "=== Handoff ===" in out
    assert "Confidence detail" not in out
