import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeConfidenceResult, Stage
from router.orchestration.canonical_status import CanonicalStatus
from router.orchestration.escalation_policy import EscalationPolicyResult
from router.orchestration.misrouting_detector import MisroutingDetectionResult
from router.orchestration.output_contract import RegimeOutputContract
from router.orchestration.transition_rules import next_stage
from router.routing import RegimeComposer
from router.state import RouterState

_COMPOSER = RegimeComposer()
TerminalSignal = Literal["completion", "failure", "contradictory", "neither"]


def _state_for(stage: Stage) -> RouterState:
    regime = _COMPOSER.compose(stage)
    runner_up = _COMPOSER.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-transition",
        task_summary="transition tests",
        current_bottleneck="test bottleneck",
        current_regime=regime,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame="frame",
        knowns=["known"],
        uncertainties=["u1"],
        contradictions=[],
        assumptions=["a1"],
        substantive_assumptions=["a1"],
        risks=["risk"],
        stage_goal="goal",
        planned_switch_condition="planned",
        observed_switch_cause=None,
        switch_trigger=None,
        recommended_next_regime=None,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=1.0,
    )


def _detection_for(stage: Stage, *, recommended_next: Stage | None = None) -> MisroutingDetectionResult:
    return MisroutingDetectionResult(
        current_regime=_COMPOSER.compose(stage),
        dominant_failure_mode="mode",
        still_productive=True,
        misrouting_detected=False,
        justification="stay",
        recommended_next_regime=_COMPOSER.compose(recommended_next) if recommended_next else None,
    )


def _canonical(terminal_signal: TerminalSignal) -> CanonicalStatus:
    return CanonicalStatus(
        terminal_signal=terminal_signal,
        artifact_status="valid_complete",
        switch_posture="stay",
        completion_signal="done" if terminal_signal == "completion" else "",
        failure_signal="failed" if terminal_signal in {"failure", "contradictory"} else "",
        is_valid=True,
        structurally_valid=True,
        semantic_valid=True,
        control_conflict=False,
        recommended_next_stage=None,
    )


def _escalation_biases(preferred_regime_biases: dict[Stage, int]) -> EscalationPolicyResult:
    return EscalationPolicyResult(
        escalation_direction="stricter",
        justification="test",
        preferred_regime_biases=preferred_regime_biases,
        switch_pressure_adjustment=2,
        debug_signals=[],
    )


def _output(stage: Stage) -> RegimeOutputContract:
    return RegimeOutputContract(stage=stage, raw_response="{}", validation={"parsed": {"artifact": {}}})


def test_epistemic_completion_can_inject_forward_to_adversarial():
    chosen = next_stage(
        _state_for(Stage.EPISTEMIC),
        _detection_for(Stage.EPISTEMIC),
        _escalation_biases({Stage.ADVERSARIAL: 2}),
        _output(Stage.EPISTEMIC),
        canonical=_canonical("completion"),
    )
    assert chosen == Stage.ADVERSARIAL


def test_epistemic_to_adversarial_to_operator_path_with_stricter_escalation_bias():
    escalation = _escalation_biases({Stage.ADVERSARIAL: 3})
    first_hop = next_stage(
        _state_for(Stage.EPISTEMIC),
        _detection_for(Stage.EPISTEMIC),
        escalation,
        _output(Stage.EPISTEMIC),
        canonical=_canonical("completion"),
    )
    second_hop = next_stage(
        _state_for(Stage.ADVERSARIAL),
        _detection_for(Stage.ADVERSARIAL),
        escalation,
        _output(Stage.ADVERSARIAL),
        canonical=_canonical("completion"),
    )
    assert first_hop == Stage.ADVERSARIAL
    assert second_hop == Stage.OPERATOR


def test_exploration_completion_consults_escalation_preference_before_synthesis():
    chosen = next_stage(
        _state_for(Stage.EXPLORATION),
        _detection_for(Stage.EXPLORATION),
        _escalation_biases({Stage.EPISTEMIC: 3}),
        _output(Stage.EXPLORATION),
        canonical=_canonical("completion"),
    )
    assert chosen == Stage.EPISTEMIC


def test_adversarial_completion_keeps_operator_when_no_legal_forward_bias_exists():
    chosen = next_stage(
        _state_for(Stage.ADVERSARIAL),
        _detection_for(Stage.ADVERSARIAL),
        _escalation_biases({Stage.EPISTEMIC: 3}),
        _output(Stage.ADVERSARIAL),
        canonical=_canonical("completion"),
    )
    assert chosen == Stage.OPERATOR


def test_synthesis_failure_branch_remains_unchanged_by_escalation_injection():
    chosen = next_stage(
        _state_for(Stage.SYNTHESIS),
        _detection_for(Stage.SYNTHESIS),
        _escalation_biases({Stage.ADVERSARIAL: 3}),
        _output(Stage.SYNTHESIS),
        canonical=_canonical("failure"),
    )
    assert chosen == Stage.EPISTEMIC


def test_operator_completion_branch_remains_unchanged_by_escalation_injection():
    chosen = next_stage(
        _state_for(Stage.OPERATOR),
        _detection_for(Stage.OPERATOR),
        _escalation_biases({Stage.EPISTEMIC: 3}),
        _output(Stage.OPERATOR),
        canonical=_canonical("completion"),
    )
    assert chosen is None
