import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.runtime import CognitiveRuntime
from router.state import make_record, to_jsonable
from router.storage import SessionStore


STRUCTURAL_TASK = (
    "Find the strongest interpretation of what this actually is. "
    "When we define the effort it expands instead of narrowing. "
    "Concrete versions feel too small. "
    "The fragments are understood, but the spine is still missing."
)


class FakeOllama:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, *, model, system, prompt, stream=False, temperature=0.2, num_predict=1200):
        if not self._responses:
            raise AssertionError("No fake response left for generate().")
        return {"response": self._responses.pop(0)}


def _synthesis_ok_json() -> str:
    return json.dumps(
        {
            "regime": "Synthesis Core",
            "purpose": "Produce the strongest coherent interpretation from live signals.",
            "stage": "synthesis",
            "artifact_type": "dominant_frame",
            "completion_signal": "coherent_frame_stable",
            "failure_signal": "frame_collapses_under_pressure_points",
            "recommended_next_regime": "adversarial",
            "artifact": {
                "central_claim": "Structure must dominate concrete fragments.",
                "organizing_idea": "A spine-level frame resolves the mismatch.",
                "key_tensions": ["Definition narrows locally while expanding structurally."],
                "supporting_structure": ["The bottleneck says fragments are clear but the spine is missing."],
                "pressure_points": ["If no spine emerges, this interpretation should be rejected."],
            },
        }
    )


def test_to_jsonable_and_saved_record_include_router_state_structure(tmp_path):
    runtime = CognitiveRuntime()
    ok_json = _synthesis_ok_json()
    runtime.ollama = FakeOllama([ok_json, ok_json, ok_json, ok_json])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(
        task=STRUCTURAL_TASK,
        model="fake",
        bounded_orchestration=True,
        max_switches=1,
    )
    assert runtime.router_state is not None

    record = make_record(
        STRUCTURAL_TASK,
        set(),
        "fake",
        decision,
        regime,
        result,
        handoff,
        runtime.router_state,
        bounded_orchestration=True,
        max_switches=1,
    )
    jsonable = to_jsonable(record)
    assert isinstance(jsonable, dict)
    assert jsonable["router_state"]["current_regime"]["stage"] in jsonable["orchestration"]["execution_stages"]
    assert jsonable["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert jsonable["router_state"]["recommended_next_regime"]["stage"]
    assert jsonable["handoff"]["recommended_next_regime"] == decision.runner_up_regime.value
    assert jsonable["orchestration"]["bounded_orchestration"] is True
    assert jsonable["orchestration"]["max_switches"] == 1
    assert "switches_attempted" in jsonable["orchestration"]
    assert "switches_executed" in jsonable["orchestration"]
    assert "switch_history" in jsonable["orchestration"]
    assert "stop_reason" in jsonable["orchestration"]

    saved = store.save(record, filename="stateful.json")
    loaded = store.load(saved.name)
    assert loaded["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["router_state"]["recommended_next_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["orchestration"]["bounded_orchestration"] is True
    assert loaded["orchestration"]["max_switches"] == 1


def test_session_store_load_backfills_missing_router_state_for_legacy_runs(tmp_path):
    store = SessionStore(root=str(tmp_path))
    legacy_path = Path(tmp_path) / "legacy.json"
    legacy_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "task": "legacy",
                "risk_profile": [],
                "model": "fake",
                "routing": {},
                "regime": {},
                "result": {},
                "handoff": {},
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load("legacy.json")
    assert "router_state" in loaded
    assert loaded["router_state"] is None
    assert loaded["orchestration"]["stop_reason"] == "legacy_record"


def test_runtime_can_restore_router_state_with_full_regime_payload(tmp_path):
    runtime = CognitiveRuntime()
    ok_json = _synthesis_ok_json()
    runtime.ollama = FakeOllama([ok_json, ok_json])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    record = make_record(STRUCTURAL_TASK, set(), "fake", decision, regime, result, handoff, runtime.router_state)
    saved = store.save(record, filename="stateful_restore.json")
    loaded = store.load(saved.name)

    restored = runtime.restore_router_state(loaded["router_state"])
    assert restored is not None
    assert restored.current_regime.stage == decision.primary_regime
    assert restored.runner_up_regime is not None
    assert restored.runner_up_regime.stage == decision.runner_up_regime
    assert restored.recommended_next_regime is not None
    assert restored.recommended_next_regime.stage == decision.runner_up_regime
    assert restored.prior_regimes[0].regime.stage == decision.primary_regime


def test_saved_record_includes_switch_denied_and_switch_exhausted_metadata(tmp_path):
    runtime = CognitiveRuntime()
    ok_json = _synthesis_ok_json()
    runtime.ollama = FakeOllama([ok_json, ok_json, ok_json, ok_json])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(
        task=STRUCTURAL_TASK,
        model="fake",
        bounded_orchestration=True,
        max_switches=1,
    )
    assert runtime.router_state is not None

    record = make_record(
        STRUCTURAL_TASK,
        set(),
        "fake",
        decision,
        regime,
        result,
        handoff,
        runtime.router_state,
        bounded_orchestration=True,
        max_switches=1,
    )
    saved = store.save(record, filename="bounded_audit.json")
    loaded = store.load(saved.name)

    orchestration = loaded["orchestration"]
    assert orchestration["bounded_orchestration"] is True
    assert orchestration["switches_executed"] <= orchestration["max_switches"]
    assert orchestration["switch_history"]
    assert orchestration["switch_history"][-1]["switch_executed"] is False
    assert orchestration["stop_reason"] in {"switch_limit_reached", "switch_not_recommended", "loop_prevented_same_stage", "loop_prevented_prior_stage"}


def test_load_router_state_adapts_legacy_stage_only_regimes(tmp_path):
    store = SessionStore(root=str(tmp_path))
    legacy_path = Path(tmp_path) / "legacy_stage_only.json"
    legacy_path.write_text(
        json.dumps(
            {
                "timestamp_utc": "2026-01-01T00:00:00+00:00",
                "task": "legacy-stage-only",
                "risk_profile": [],
                "model": "fake",
                "routing": {},
                "regime": {},
                "result": {},
                "handoff": {},
                "router_state": {
                    "task_id": "task-legacy",
                    "task_summary": "legacy",
                    "current_bottleneck": "legacy",
                    "current_regime": "synthesis",
                    "runner_up_regime": {"stage": "epistemic"},
                    "regime_confidence": {"level": "low"},
                    "dominant_frame": None,
                    "knowns": [],
                    "uncertainties": [],
                    "contradictions": [],
                    "assumptions": [],
                    "risks": [],
                    "stage_goal": "legacy goal",
                    "switch_trigger": None,
                    "recommended_next_regime": "adversarial",
                    "decision_pressure": 0,
                    "evidence_quality": 0,
                    "recurrence_potential": 0,
                    "prior_regimes": [
                        {
                            "regime": "operator",
                            "reason_entered": "legacy",
                            "completion_signal_seen": False,
                            "failure_signal_seen": True,
                            "outcome_summary": "legacy step",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    runtime = CognitiveRuntime()
    restored = store.load_router_state("legacy_stage_only.json", runtime.composer.compose)
    assert restored is not None
    assert restored.current_regime.stage.value == "synthesis"
    assert restored.runner_up_regime is not None
    assert restored.runner_up_regime.stage.value == "epistemic"
    assert restored.recommended_next_regime is not None
    assert restored.recommended_next_regime.stage.value == "adversarial"
    assert restored.prior_regimes[0].regime.stage.value == "operator"
