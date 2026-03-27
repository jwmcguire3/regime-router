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
    "The fragments are understood, but the spine is still missing; "
    "a single frame compresses too early."
)


class FakeOllama:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, *, model, system, prompt, stream=False, temperature=0.2, num_predict=1200):
        if not self._responses:
            raise AssertionError("No fake response left for generate().")
        return {"response": self._responses.pop(0)}


def _synthesis_ok_json() -> str:
    payload = {
        "regime": "Synthesis Core",
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
        "artifact": {
            "central_claim": "The frame must expand when defined because the concrete view shrinks the real structure.",
            "organizing_idea": "Definition changes the unit of analysis.",
            "key_tensions": ["Define vs expand."],
            "supporting_structure": ["Fragments imply linkage bottleneck."],
            "pressure_points": ["If spine mapping fails, frame weakens."],
        },
    }
    return json.dumps(payload)


def test_router_state_serializes_into_saved_record_and_json_safe(tmp_path):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([_synthesis_ok_json(), _synthesis_ok_json()])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    record = make_record(STRUCTURAL_TASK, set(), "fake", decision, regime, result, handoff, runtime.router_state)

    path = store.save(record, filename="router_state_serialization.json")
    loaded = store.load(path.name)

    assert loaded["router_state"] is not None
    assert loaded["router_state"]["current_regime"]["stage"] == decision.primary_regime.value
    assert loaded["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["router_state"]["recommended_next_regime"]["stage"] == decision.runner_up_regime.value

    # Ensure this is clean JSON after round-trip and enum/dataclass conversion.
    json.dumps(loaded)
    json.dumps(to_jsonable(record))


def test_session_store_load_preserves_backward_compat_for_old_records(tmp_path):
    store = SessionStore(root=str(tmp_path))
    old_style = {
        "timestamp_utc": "2026-03-27T00:00:00+00:00",
        "task": "legacy task",
        "risk_profile": [],
        "model": "fake",
        "routing": {},
        "regime": {},
        "result": {},
        "handoff": {},
    }
    legacy_path = Path(tmp_path) / "legacy.json"
    legacy_path.write_text(json.dumps(old_style), encoding="utf-8")

    loaded = store.load("legacy.json")
    assert "router_state" in loaded
    assert loaded["router_state"] is None
