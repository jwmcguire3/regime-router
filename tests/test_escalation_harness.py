import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import EscalationPolicy, MisroutingDetectionResult
from router.models import RegimeConfidenceResult, Severity, Stage
from router.routing import RegimeComposer
from router.state import RouterState


_POLICY = EscalationPolicy()
_COMPOSER = RegimeComposer()


def make_state(
    contradictions: list[str] | None = None,
    *,
    fragility_pressure: float = 0,
    possibility_space_need: float = 0,
) -> RouterState:
    current = _COMPOSER.compose(Stage.SYNTHESIS)
    runner_up = _COMPOSER.compose(Stage.EPISTEMIC)
    return RouterState(
        task_id="task-escalation",
        task_summary="escalation harness",
        current_bottleneck="test bottleneck",
        current_regime=current,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=["unknown"],
        contradictions=list(contradictions or []),
        assumptions=[],
        risks=["risk"],
        stage_goal="goal",
        switch_trigger=None,
        recommended_next_regime=runner_up,
        fragility_pressure=fragility_pressure,
        possibility_space_need=possibility_space_need,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=0.0,
    )


def make_misrouting(misrouting_detected: bool) -> MisroutingDetectionResult:
    return MisroutingDetectionResult(
        current_regime=_COMPOSER.compose(Stage.SYNTHESIS),
        dominant_failure_mode="test",
        still_productive=not misrouting_detected,
        misrouting_detected=misrouting_detected,
        justification="test",
        recommended_next_regime=_COMPOSER.compose(Stage.EPISTEMIC) if misrouting_detected else None,
    )


def eval_policy(
    *,
    task_text: str,
    state: RouterState | None = None,
    current_stage: Stage = Stage.SYNTHESIS,
    regime_confidence: RegimeConfidenceResult | None = None,
    misrouting_detected: bool = False,
):
    return _POLICY.evaluate(
        state=state,
        task_text=task_text,
        current_regime=_COMPOSER.compose(current_stage),
        regime_confidence=regime_confidence,
        misrouting_result=make_misrouting(True) if misrouting_detected else None,
    )


def test_stricter_from_high_fragility_pressure():
    result = eval_policy(task_text="generic planning task", state=make_state(fragility_pressure=2))
    assert result.escalation_direction == "stricter"
    assert result.switch_pressure_adjustment >= 2
    assert result.preferred_regime_biases.get(Stage.EPISTEMIC, 0) >= 1
    assert result.preferred_regime_biases.get(Stage.ADVERSARIAL, 0) >= 1


def test_stricter_from_deployment_or_production_text():
    result = eval_policy(task_text="prepare deployment checklist for production", state=make_state())
    assert result.escalation_direction == "stricter"
    assert result.switch_pressure_adjustment >= 2
    assert Stage.EPISTEMIC in result.preferred_regime_biases


def test_stricter_from_two_or_more_contradictions_biases_epistemic():
    result = eval_policy(
        task_text="resolve these conflicts",
        state=make_state(contradictions=["c1", "c2"]),
    )
    assert result.escalation_direction == "stricter"
    assert result.switch_pressure_adjustment >= 2
    assert result.preferred_regime_biases.get(Stage.EPISTEMIC, 0) >= 2


def test_stricter_from_prove_or_confidence_text():
    result = eval_policy(task_text="prove this claim with confidence", state=make_state())
    assert result.escalation_direction == "stricter"
    assert result.switch_pressure_adjustment >= 2
    assert result.preferred_regime_biases.get(Stage.EPISTEMIC, 0) >= 2


def test_stricter_combined_fragility_and_contradictions_increases_adjustment():
    result = eval_policy(
        task_text="safety review",
        state=make_state(contradictions=["c1", "c2"], fragility_pressure=2),
    )
    assert result.escalation_direction == "stricter"
    assert result.switch_pressure_adjustment == 3
    assert result.preferred_regime_biases.get(Stage.EPISTEMIC, 0) >= 3


def test_looser_from_brainstorm_or_map_space_with_high_possibility_need():
    result = eval_policy(
        task_text="brainstorm options and map the space before deciding",
        state=make_state(possibility_space_need=3),
    )
    assert result.escalation_direction == "looser"
    assert result.switch_pressure_adjustment <= -2
    assert result.preferred_regime_biases.get(Stage.EXPLORATION, 0) >= 2


def test_looser_from_before_narrowing_or_keep_it_open():
    result = eval_policy(task_text="keep it open before narrowing", state=make_state())
    assert result.escalation_direction == "looser"
    assert result.switch_pressure_adjustment <= -2
    assert result.preferred_regime_biases.get(Stage.EXPLORATION, 0) >= 1


def test_looser_from_lack_of_structure_signal():
    result = eval_policy(
        task_text="we have a lack of structure and cannot characterize the space",
        state=make_state(possibility_space_need=3),
    )
    assert result.escalation_direction == "looser"
    assert result.switch_pressure_adjustment <= -2
    assert result.preferred_regime_biases.get(Stage.EXPLORATION, 0) >= 2
    assert result.preferred_regime_biases.get(Stage.SYNTHESIS, 0) >= 1


def test_neutral_with_no_pressure_signals():
    result = eval_policy(task_text="summarize the current status", state=make_state())
    assert result.escalation_direction == "none"
    assert result.switch_pressure_adjustment == 0
    assert result.preferred_regime_biases == {}


def test_neutral_when_strict_and_loose_signals_are_balanced():
    result = eval_policy(task_text="prove it, but also brainstorm", state=make_state())
    assert result.escalation_direction == "none"
    assert result.switch_pressure_adjustment == 0
    assert result.preferred_regime_biases == {}


def test_looser_biases_exploration():
    result = eval_policy(task_text="before narrowing, map the space", state=make_state(possibility_space_need=3))
    assert result.escalation_direction == "looser"
    assert result.preferred_regime_biases.get(Stage.EXPLORATION, 0) > 0


def test_stricter_in_exploration_adds_synthesis_bias():
    result = eval_policy(
        task_text="prove this quickly",
        state=make_state(),
        current_stage=Stage.EXPLORATION,
    )
    assert result.escalation_direction == "stricter"
    assert result.preferred_regime_biases.get(Stage.SYNTHESIS, 0) >= 1


def test_scope_misrouting_detected_adds_strict_pressure():
    baseline = eval_policy(task_text="prove this", state=make_state())
    escalated = eval_policy(task_text="prove this", state=make_state(), misrouting_detected=True)
    assert baseline.escalation_direction == "stricter"
    assert escalated.escalation_direction == "stricter"
    assert escalated.switch_pressure_adjustment >= baseline.switch_pressure_adjustment
    assert "misrouting_detected" in escalated.debug_signals


def test_scope_low_confidence_with_strict_signals_adds_strict_pressure():
    high_conf = RegimeConfidenceResult(
        level=Severity.HIGH.value,
        rationale="test",
        top_stage_score=4,
        runner_up_score=1,
        score_gap=3,
        nontrivial_stage_count=2,
        weak_lexical_dependence=False,
        structural_feature_state="rich",
    )
    baseline = eval_policy(task_text="prove this", state=make_state(), regime_confidence=high_conf)
    low_conf = eval_policy(
        task_text="prove this",
        state=make_state(),
        regime_confidence=RegimeConfidenceResult.low_default(),
    )
    assert low_conf.escalation_direction == "stricter"
    assert low_conf.switch_pressure_adjustment >= baseline.switch_pressure_adjustment
    assert "low_confidence_requires_stricter_grounding" in low_conf.debug_signals
