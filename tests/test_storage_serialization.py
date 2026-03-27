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
    runtime.ollama = FakeOllama([ok_json, ok_json])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    assert runtime.router_state is not None

    record = make_record(STRUCTURAL_TASK, set(), "fake", decision, regime, result, handoff, runtime.router_state)
    jsonable = to_jsonable(record)
    assert isinstance(jsonable, dict)
    assert jsonable["router_state"]["current_regime"]["stage"] == decision.primary_regime.value
    assert jsonable["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert jsonable["router_state"]["recommended_next_regime"]["stage"] == decision.runner_up_regime.value
    assert jsonable["handoff"]["recommended_next_regime"] == decision.runner_up_regime.value

    saved = store.save(record, filename="stateful.json")
    loaded = store.load(saved.name)
    assert loaded["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["router_state"]["recommended_next_regime"]["stage"] == decision.runner_up_regime.value


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
