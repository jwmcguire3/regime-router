import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector, RegimeOutputContract
from router.models import RegimeConfidenceResult, Stage
from router.routing import RegimeComposer
from router.state import RouterState


def _state_for(stage: Stage) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-test",
        task_summary="test",
        current_bottleneck="test bottleneck",
        current_regime=regime,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=[],
        uncertainties=[],
        contradictions=["Live contradiction"] if stage == Stage.SYNTHESIS else [],
        assumptions=["Assumption pending"] if stage == Stage.OPERATOR else [],
        risks=[],
        stage_goal="goal",
        switch_trigger="trigger",
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=0.0,
    )


def _output_for(stage: Stage, artifact: dict) -> RegimeOutputContract:
    payload = {
        "regime": "test",
        "stage": stage.value,
        "artifact_type": "test_artifact",
        "artifact": artifact,
    }
    return RegimeOutputContract(
        stage=stage,
        raw_response=json.dumps(payload),
        validation={"parsed": payload},
    )


def test_exploration_positive_detects_branch_sprawl_and_recommends_next_regime():
    detector = MisroutingDetector()
    state = _state_for(Stage.EXPLORATION)
    output = _output_for(
        Stage.EXPLORATION,
        {
            "candidate_frames": ["f1", "f2", "f3", "f4", "f5"],
            "selection_criteria": "",
            "unresolved_axes": ["a1", "a2", "a3"],
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is True
    assert result.still_productive is False
    assert result.recommended_next_regime == state.recommended_next_regime.stage


def test_exploration_negative_does_not_trigger_when_selection_criteria_present():
    detector = MisroutingDetector()
    state = _state_for(Stage.EXPLORATION)
    output = _output_for(
        Stage.EXPLORATION,
        {
            "candidate_frames": ["f1", "f2", "f3"],
            "selection_criteria": "Choose the frame that changes the decision boundary and narrows next action.",
            "unresolved_axes": ["a1"],
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is False
    assert result.recommended_next_regime is None


def test_synthesis_positive_detects_false_unification():
    detector = MisroutingDetector()
    state = _state_for(Stage.SYNTHESIS)
    output = _output_for(
        Stage.SYNTHESIS,
        {
            "central_claim": "A clean unified frame explains the whole pattern at once.",
            "organizing_idea": "Everything merges into one clean interpretation.",
            "supporting_structure": "Thin support only.",
            "pressure_points": "General pressure statement only.",
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is True
    assert result.justification == MisroutingDetector.JUSTIFICATION_SWITCH


def test_synthesis_negative_does_not_trigger_with_supported_frame_and_contradiction_handling():
    detector = MisroutingDetector()
    state = _state_for(Stage.SYNTHESIS)
    output = _output_for(
        Stage.SYNTHESIS,
        {
            "central_claim": "The frame only holds if contradictory evidence is explicitly tracked before commitment.",
            "organizing_idea": "Unification follows explicit contradiction bookkeeping.",
            "supporting_structure": "Two contradictory signals remain active, and each one updates whether the frame survives the next decision.",
            "pressure_points": "If contradictions remain unresolved, we do not unify and we re-open frame comparison.",
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is False
    assert result.still_productive is True


def test_operator_positive_detects_forced_closure():
    detector = MisroutingDetector()
    state = _state_for(Stage.OPERATOR)
    output = _output_for(
        Stage.OPERATOR,
        {
            "decision": "Pick option A now.",
            "rationale": "Fast choice.",
            "tradeoff_accepted": "Minimal.",
            "next_actions": ["Do it"],
            "fallback_trigger": "none",
            "review_point": "later",
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == state.recommended_next_regime.stage


def test_operator_negative_does_not_trigger_with_stable_tradeoffs():
    detector = MisroutingDetector()
    state = _state_for(Stage.OPERATOR)
    output = _output_for(
        Stage.OPERATOR,
        {
            "decision": "Choose option B.",
            "rationale": "Option B preserves reversibility while matching the near-term constraint profile in the current state.",
            "tradeoff_accepted": "We accept slower immediate throughput in exchange for preserving optionality over the next review window.",
            "next_actions": ["Run two-day trial", "Check trigger"],
            "fallback_trigger": "If throughput drops below threshold",
            "review_point": "72 hours",
        },
    )

    result = detector.detect(state, output)

    assert result.misrouting_detected is False


def test_detector_does_not_trigger_constantly_across_mixed_cases():
    detector = MisroutingDetector()
    triggered = 0

    for stage, artifact in [
        (
            Stage.EXPLORATION,
            {
                "candidate_frames": ["f1", "f2", "f3"],
                "selection_criteria": "Select based on decision boundary impact.",
                "unresolved_axes": ["a1"],
            },
        ),
        (
            Stage.SYNTHESIS,
            {
                "central_claim": "Claim with support.",
                "organizing_idea": "Frame with explicit limits.",
                "supporting_structure": "Support contains multiple specific constraints and checks that keep the frame grounded.",
                "pressure_points": "Contradictions are listed and remain live.",
            },
        ),
        (
            Stage.OPERATOR,
            {
                "decision": "Option B",
                "rationale": "Rationale includes assumptions, boundaries, and why this timing is acceptable.",
                "tradeoff_accepted": "Tradeoff is explicit and tied to known assumptions and review timing.",
                "next_actions": ["step1"],
                "fallback_trigger": "trigger",
                "review_point": "one week",
            },
        ),
    ]:
        state = _state_for(stage)
        result = detector.detect(state, _output_for(stage, artifact))
        if result.misrouting_detected:
            triggered += 1

    assert triggered < 2
