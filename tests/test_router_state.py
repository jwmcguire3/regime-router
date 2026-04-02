import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Regime, Stage
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
            "purpose": "Produce the strongest coherent interpretation from live signals.",
            "stage": "synthesis",
            "artifact_type": "dominant_frame",
            "completion_signal": "coherent_frame_stable",
            "failure_signal": "frame_collapses_under_pressure_points",
            "recommended_next_regime": "adversarial",
            "artifact": {
                "central_claim": "The frame expands under definition because concrete cuts hide structure.",
                "organizing_idea": "Fragments need a shared spine-level structure.",
                "key_tensions": ["Concrete detail vs structural coherence."],
                "supporting_structure": ["Fragments are legible but the backbone is not."],
                "pressure_points": ["If the spine does not improve explanatory power, this frame is weaker."],
            },
        }
    )


def test_plan_populates_router_state_core_fields():
    runtime = CognitiveRuntime()
    decision, _, _ = runtime.plan(STRUCTURAL_TASK)

    assert runtime.router_state is not None
    state = runtime.router_state
    assert state.task_id.startswith("task-")
    assert state.task_summary
    assert state.current_bottleneck == STRUCTURAL_TASK
    assert isinstance(state.current_regime, Regime)
    assert state.current_regime.stage == decision.primary_regime
    assert state.runner_up_regime is not None
    assert isinstance(state.runner_up_regime, Regime)
    assert state.regime_confidence.level in {"low", "medium", "high"}
    assert state.stage_goal
    assert state.knowns
    assert state.uncertainties == []
    assert state.contradictions == []
    assert state.assumptions == [
        "Structural signals observed: expansion_when_defined, concrete_versions_feel_too_small, fragments_understood_spine_missed"
    ]
    assert state.risks
    assert isinstance(state.decision_pressure, float)
    assert isinstance(state.evidence_quality, float)
    assert isinstance(state.recurrence_potential, float)
    assert state.prior_regimes == []

