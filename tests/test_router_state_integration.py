import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeExecutionResult, RoutingDecision, Stage
from router.runtime import CognitiveRuntime
from router.state import Handoff


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


def test_plan_populates_router_state_required_fields():
    runtime = CognitiveRuntime()
    decision, regime, handoff = runtime.plan(STRUCTURAL_TASK)

    assert runtime.router_state is not None
    state = runtime.router_state
    assert state.task_id
    assert state.task_summary
    assert state.current_bottleneck == STRUCTURAL_TASK
    assert state.current_regime.name == regime.name
    assert state.runner_up_regime is not None
    assert state.regime_confidence.level in {"low", "medium", "high"}
    assert state.stage_goal
    assert isinstance(state.knowns, list)
    assert isinstance(state.uncertainties, list)
    assert isinstance(state.contradictions, list)
    assert isinstance(state.assumptions, list)
    assert isinstance(state.risks, list)
    assert isinstance(state.decision_pressure, float)
    assert isinstance(state.evidence_quality, float)
    assert isinstance(state.recurrence_potential, float)
    assert isinstance(state.prior_regimes, list)

    assert handoff.current_bottleneck == state.current_bottleneck
    assert handoff.what_is_known == state.knowns


def test_execute_populates_router_state_and_prior_regimes():
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([_synthesis_ok_json(), _synthesis_ok_json()])

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    state = runtime.router_state
    assert state.current_regime.name == regime.name
    assert len(state.prior_regimes) == 1
    assert state.prior_regimes[0].regime == decision.primary_regime
    assert isinstance(result, RegimeExecutionResult)
    assert isinstance(handoff, Handoff)


def test_handoff_projection_is_consistent_with_router_state_lists_and_recommended_next():
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([_synthesis_ok_json(), _synthesis_ok_json()])

    _, _, _, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    assert runtime.router_state is not None
    state = runtime.router_state

    assert handoff.what_is_known == state.knowns
    assert handoff.what_remains_uncertain == state.uncertainties
    assert handoff.active_contradictions == state.contradictions
    assert handoff.assumptions_in_play == state.assumptions
    assert handoff.recommended_next_regime == (
        state.recommended_next_regime.stage if state.recommended_next_regime else None
    )


def test_runtime_legacy_output_shapes_remain_compatible():
    runtime = CognitiveRuntime()
    planned = runtime.plan("Can you help?")
    assert len(planned) == 3
    assert isinstance(planned[0], RoutingDecision)

    runtime.ollama = FakeOllama([_synthesis_ok_json(), _synthesis_ok_json()])
    executed = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    assert len(executed) == 4
    assert executed[0].primary_regime in set(Stage)
