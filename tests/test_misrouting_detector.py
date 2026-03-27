import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector
from router.models import Stage
from router.state import RouterState


def _state(stage: Stage, *, runner_up: Stage | None = None, recommended: Stage | None = None) -> RouterState:
    return RouterState(
        task_id="task-1",
        task_summary="Test task",
        current_bottleneck="testing",
        current_regime=stage,
        runner_up_regime=runner_up,
        regime_confidence="medium",
        stage_goal="test",
        knowns=[],
        uncertainties=[],
        contradictions=[],
        assumptions=[],
        risks=[],
        decision_pressure=1,
        evidence_quality=1,
        recurrence_potential=1,
        prior_regimes=[],
        switch_trigger="trigger",
        recommended_next_regime=recommended,
    )


def test_detector_flags_synthesis_dominant_failure_and_recommends_switch():
    detector = MisroutingDetector()
    state = _state(Stage.SYNTHESIS, runner_up=Stage.EPISTEMIC, recommended=Stage.EPISTEMIC)

    output = (
        "This is a clean frame with coherent narrative and elegant frame language. "
        "The evidence is thin evidence and mostly speculative. "
        "We resolved contradiction immediately into a single theme without test."
    )
    validation = {"semantic_failures": ["artifact is not grounded in the task specifics"]}

    result = detector.detect(state=state, regime_output=output, validation=validation)

    assert result.current_regime == Stage.SYNTHESIS
    assert result.misrouting_detected is True
    assert result.dominant_failure_mode == "coherence polish outrunning evidential support"
    assert result.recommended_next_regime == Stage.EPISTEMIC
    assert result.switch_message == "Current regime is hitting its dominant failure mode. Switching is justified."


def test_detector_does_not_trigger_for_epistemic_when_clear_failure_pattern_is_absent():
    detector = MisroutingDetector()
    state = _state(Stage.EPISTEMIC, runner_up=Stage.OPERATOR, recommended=Stage.OPERATOR)

    output = (
        "Evidence summary: uncertainty remains in two sources, and we recommend collecting source A next. "
        "Immediate next step is to verify the missing dataset before selecting an option."
    )

    result = detector.detect(state=state, regime_output=output, validation={"semantic_failures": []})

    assert result.current_regime == Stage.EPISTEMIC
    assert result.misrouting_detected is False
    assert result.recommended_next_regime is None
    assert result.switch_message is None


def test_detector_flags_builder_when_architecture_is_premature():
    detector = MisroutingDetector()
    state = _state(Stage.BUILDER, runner_up=Stage.OPERATOR, recommended=Stage.OPERATOR)

    output = (
        "We should stand up a framework and platform before demand is known. "
        "This might recur someday, so add a new module before usage appears."
    )

    result = detector.detect(state=state, regime_output=output)

    assert result.current_regime == Stage.BUILDER
    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.OPERATOR
