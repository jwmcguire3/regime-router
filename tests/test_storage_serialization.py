import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
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
