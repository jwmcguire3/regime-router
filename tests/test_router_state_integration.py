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
            "purpose": "Produce the strongest coherent interpretation from live signals.",
            "stage": "synthesis",
            "artifact_type": "dominant_frame",
            "completion_signal": "coherent_frame_stable",
            "failure_signal": "frame_collapses_under_pressure_points",
            "recommended_next_regime": "adversarial",
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
    assert handoff.recommended_next_regime_full is state.recommended_next_regime
    assert handoff.recommended_next_regime == decision.runner_up_regime


def test_handoff_recommended_next_regime_remains_stage_shaped_after_execute(synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json, synthesis_ok_json])

    decision, _, _, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == decision.runner_up_regime
    assert runtime.router_state.recommended_next_regime is runtime.router_state.runner_up_regime
    assert isinstance(handoff.recommended_next_regime, Stage)
    assert handoff.recommended_next_regime == decision.runner_up_regime
    assert handoff.recommended_next_regime_full is runtime.router_state.recommended_next_regime


def test_execute_overrides_recommended_next_regime_from_validated_output():
    runtime = CognitiveRuntime()
    execution_json = json.dumps(
        {
            "regime": "Synthesis Core",
            "purpose": "Compress to a coherent frame grounded in the task's structural signals.",
            "stage": "synthesis",
            "artifact_type": "dominant_frame",
            "completion_signal": "coherent_frame_stable",
            "failure_signal": "frame_collapses_under_pressure_points",
            "recommended_next_regime": "adversarial",
            "artifact": {
                "central_claim": "Defining the effort expands it because concrete slices miss the spine.",
                "organizing_idea": "A spine-level frame explains why fragments look coherent only in isolation.",
                "key_tensions": ["Concrete detail vs structural spine coherence."],
                "supporting_structure": ["Task signals explicitly connect define→expand and fragment→spine gap."],
                "pressure_points": ["If expansion disappears when definition tightens, the frame is wrong."],
            },
        }
    )
    runtime.ollama = FakeOllama([execution_json, execution_json])

    task = "What is the strongest interpretation, and what evidence is missing?"
    decision, _, _, handoff = runtime.execute(task=task, model="fake")

    assert decision.runner_up_regime == Stage.EPISTEMIC
    assert runtime.router_state is not None
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == Stage.ADVERSARIAL
    assert handoff.recommended_next_regime == Stage.ADVERSARIAL


def test_execute_records_completion_and_failure_signals_in_prior_regime_summary(synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json, synthesis_ok_json])

    runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    summary = runtime.router_state.prior_regimes[-1].outcome_summary
    assert "completion_signal=" in summary
    assert "failure_signal=" in summary


def test_invalid_structural_output_does_not_override_recommended_next_regime_or_handoff():
    runtime = CognitiveRuntime()
    invalid_structural = json.dumps(
        {
            "regime": "exploration",
            "purpose": "Explore frames before choosing.",
            "artifact_type": "decision_packet",
            "completion_signal": "decision_committed_with_actions",
            "failure_signal": "decision_not_actionable_under_constraints",
            "recommended_next_regime": "synthesis",
            "artifact": {
                "frames": ["Frame A", "Frame B"],
            },
        }
    )
    runtime.ollama = FakeOllama([invalid_structural, invalid_structural])

    decision, _, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert decision.runner_up_regime == Stage.ADVERSARIAL
    assert result.validation["is_valid"] is False
    assert runtime.router_state is not None
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == Stage.ADVERSARIAL
    assert handoff.recommended_next_regime == Stage.ADVERSARIAL
