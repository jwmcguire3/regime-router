import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.runtime import CognitiveRuntime


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


@pytest.fixture
def synthesis_ok_json() -> str:
    return json.dumps(
        {
            "regime": "Synthesis Core",
            "stage": "synthesis",
            "artifact_type": "dominant_frame",
            "artifact": {
                "central_claim": "The spine is missing despite legible fragments.",
                "organizing_idea": "Structure-level framing restores coherence.",
                "key_tensions": ["Concrete slice vs whole-system relation."],
                "supporting_structure": ["The prompt explicitly states a parts/whole mismatch."],
                "pressure_points": ["If concrete variants do not map back to a spine, this fails."],
            },
        }
    )


def test_handoff_fields_are_router_state_backed_after_plan():
    runtime = CognitiveRuntime()
    decision, _, handoff = runtime.plan(STRUCTURAL_TASK)

    assert runtime.router_state is not None
    state = runtime.router_state
    assert handoff.what_is_known == state.knowns
    assert handoff.what_remains_uncertain == state.uncertainties
    assert handoff.active_contradictions == state.contradictions
    assert handoff.assumptions_in_play == state.assumptions
    assert handoff.recommended_next_regime == state.recommended_next_regime.stage
    assert handoff.recommended_next_regime == decision.runner_up_regime


def test_handoff_recommended_next_regime_remains_stage_shaped_after_execute(synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json, synthesis_ok_json])

    decision, _, _, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == decision.runner_up_regime
    assert isinstance(handoff.recommended_next_regime, Stage)
    assert handoff.recommended_next_regime == decision.runner_up_regime
