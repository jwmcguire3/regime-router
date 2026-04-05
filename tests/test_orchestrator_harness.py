import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import EscalationPolicyResult, MisroutingDetectionResult, RegimeOutputContract, SwitchOrchestrator
from router.models import RegimeConfidenceResult, Stage
from router.routing import RegimeComposer
from router.state import RouterState


def make_state(
    stage: Stage,
    *,
    assumptions: list[str] | None = None,
    contradictions: list[str] | None = None,
    recurrence_potential: float = 0.0,
) -> RouterState:
    composer = RegimeComposer()
    current = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id=f"task-{stage.value}",
        task_summary="switch orchestrator harness",
        current_bottleneck="test bottleneck",
        current_regime=current,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=["uncertain"],
        contradictions=list(contradictions or []),
        assumptions=list(assumptions or []),
        risks=["risk"],
        stage_goal="goal",
        planned_switch_condition="planned_condition",
        observed_switch_cause=None,
        switch_trigger=None,
        recommended_next_regime=None,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
    )


def make_output(stage: Stage, completion_signal: str = "", failure_signal: str = "") -> RegimeOutputContract:
    parsed = {
        "regime": stage.value,
        "purpose": "test",
        "artifact_type": "test_artifact",
        "artifact": {},
        "completion_signal": completion_signal,
        "failure_signal": failure_signal,
        "recommended_next_regime": "operator",
    }
    return RegimeOutputContract(stage=stage, raw_response=json.dumps(parsed), validation={"parsed": parsed})


def make_detection(
    stage: Stage,
    misrouting_detected: bool = False,
    recommended_next_stage: Stage | None = None,
) -> MisroutingDetectionResult:
    composer = RegimeComposer()
    return MisroutingDetectionResult(
        current_regime=composer.compose(stage),
        dominant_failure_mode="test_mode",
        still_productive=not misrouting_detected,
        misrouting_detected=misrouting_detected,
        justification="test",
        recommended_next_regime=composer.compose(recommended_next_stage) if recommended_next_stage else None,
    )


def test_allowed_pathways_harness():
    orchestrator = SwitchOrchestrator(RegimeComposer())

    cases = [
        (
            Stage.EXPLORATION,
            make_output(Stage.EXPLORATION, completion_signal="done"),
            make_detection(Stage.EXPLORATION),
            Stage.SYNTHESIS,
            "done",
            {},
        ),
        (
            Stage.SYNTHESIS,
            make_output(Stage.SYNTHESIS, failure_signal="failed"),
            make_detection(Stage.SYNTHESIS, misrouting_detected=True, recommended_next_stage=Stage.ADVERSARIAL),
            Stage.ADVERSARIAL,
            "failed",
            {},
        ),
        (
            Stage.EPISTEMIC,
            make_output(Stage.EPISTEMIC, completion_signal="supported"),
            make_detection(Stage.EPISTEMIC),
            Stage.OPERATOR,
            "supported",
            {},
        ),
        (
            Stage.ADVERSARIAL,
            make_output(Stage.ADVERSARIAL, completion_signal="stressed"),
            make_detection(Stage.ADVERSARIAL),
            Stage.OPERATOR,
            "stressed",
            {},
        ),
        (
            Stage.OPERATOR,
            make_output(Stage.OPERATOR, completion_signal="decision_ready"),
            make_detection(Stage.OPERATOR),
            Stage.BUILDER,
            "decision_ready",
            {"recurrence_potential": 2.0},
        ),
    ]

    for stage, output, detection, expected_next, expected_trigger, state_kwargs in cases:
        state = make_state(stage, **state_kwargs)
        result = orchestrator.orchestrate(state, output, detection, switches_used=0, max_switches=2)
        assert result.switch_recommended_now is True
        assert result.next_regime is not None
        assert result.next_regime.stage == expected_next
        assert result.updated_state.observed_switch_cause == expected_trigger
        assert result.updated_state.planned_switch_condition == "planned_condition"

    builder_state = make_state(Stage.BUILDER)
    builder_result = orchestrator.orchestrate(
        builder_state,
        make_output(Stage.BUILDER, completion_signal="builder_complete"),
        make_detection(Stage.BUILDER),
        switches_used=0,
        max_switches=2,
    )
    assert builder_result.switch_recommended_now is False
    assert builder_result.next_regime is None


def test_blocked_pathways_harness():
    orchestrator = SwitchOrchestrator(RegimeComposer())

    exploration_state = make_state(Stage.EXPLORATION)
    exploration_result = orchestrator.orchestrate(
        exploration_state,
        make_output(Stage.EXPLORATION),
        make_detection(Stage.EXPLORATION, misrouting_detected=True, recommended_next_stage=Stage.OPERATOR),
        switches_used=0,
        max_switches=2,
    )
    assert exploration_result.switch_recommended_now is False
    assert exploration_result.next_regime is None

    synthesis_state = make_state(Stage.SYNTHESIS)
    synthesis_result = orchestrator.orchestrate(
        synthesis_state,
        make_output(Stage.SYNTHESIS),
        make_detection(Stage.SYNTHESIS, misrouting_detected=True, recommended_next_stage=Stage.BUILDER),
        switches_used=0,
        max_switches=2,
    )
    assert synthesis_result.switch_recommended_now is False
    assert synthesis_result.next_regime is None


def test_switch_cap_harness():
    orchestrator = SwitchOrchestrator(RegimeComposer())

    state_at_cap = make_state(Stage.EXPLORATION)
    result_at_cap = orchestrator.orchestrate(
        state_at_cap,
        make_output(Stage.EXPLORATION, completion_signal="done"),
        make_detection(Stage.EXPLORATION),
        switches_used=2,
        max_switches=2,
    )
    assert result_at_cap.switch_recommended_now is False
    assert result_at_cap.next_regime is None

    state_can_switch = make_state(Stage.EXPLORATION)
    result_can_switch = orchestrator.orchestrate(
        state_can_switch,
        make_output(Stage.EXPLORATION, completion_signal="done"),
        make_detection(Stage.EXPLORATION),
        switches_used=1,
        max_switches=2,
    )
    assert result_can_switch.switch_recommended_now is True
    assert result_can_switch.next_regime is not None
    assert result_can_switch.next_regime.stage == Stage.SYNTHESIS
    assert result_can_switch.updated_state.observed_switch_cause == "done"

    zero_cap_state = make_state(Stage.EXPLORATION)
    zero_cap_result = orchestrator.orchestrate(
        zero_cap_state,
        make_output(Stage.EXPLORATION, completion_signal="done"),
        make_detection(Stage.EXPLORATION),
        switches_used=0,
        max_switches=0,
    )
    assert zero_cap_result.switch_recommended_now is False
    assert zero_cap_result.next_regime is None


def test_no_signal_harness():
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(
        make_state(Stage.EPISTEMIC),
        make_output(Stage.EPISTEMIC),
        make_detection(Stage.EPISTEMIC),
        switches_used=0,
        max_switches=2,
    )
    assert result.switch_recommended_now is False
    assert result.next_regime is None


def test_assumption_or_frame_collapse_harness():
    orchestrator = SwitchOrchestrator(RegimeComposer())

    assumption_collapse_result = orchestrator.orchestrate(
        make_state(Stage.OPERATOR, assumptions=["a1"], contradictions=["c1"]),
        make_output(Stage.OPERATOR, failure_signal="assumption collapse in active model"),
        make_detection(Stage.OPERATOR),
        switches_used=0,
        max_switches=2,
    )
    assert assumption_collapse_result.switch_recommended_now is True
    assert assumption_collapse_result.next_regime is not None
    assert assumption_collapse_result.next_regime.stage == Stage.EXPLORATION
    assert assumption_collapse_result.updated_state.observed_switch_cause == "assumption_or_frame_collapse"

    frame_collapse_result = orchestrator.orchestrate(
        make_state(Stage.SYNTHESIS, assumptions=["a1"], contradictions=["c1"]),
        make_output(Stage.SYNTHESIS, failure_signal="frame collapse detected"),
        make_detection(Stage.SYNTHESIS),
        switches_used=0,
        max_switches=2,
    )
    assert frame_collapse_result.switch_recommended_now is True
    assert frame_collapse_result.next_regime is not None
    assert frame_collapse_result.next_regime.stage == Stage.EXPLORATION
    assert frame_collapse_result.updated_state.observed_switch_cause == "assumption_or_frame_collapse"


def test_escalation_damping_harness():
    state = make_state(Stage.EXPLORATION)
    output = make_output(Stage.EXPLORATION)
    detection = make_detection(Stage.EXPLORATION, misrouting_detected=True, recommended_next_stage=Stage.SYNTHESIS)
    escalation = EscalationPolicyResult(
        escalation_direction="looser",
        justification="test damping",
        preferred_regime_biases={Stage.EXPLORATION: 1},
        switch_pressure_adjustment=-2,
    )

    result = SwitchOrchestrator(RegimeComposer()).orchestrate(
        state,
        output,
        detection,
        switches_used=0,
        max_switches=2,
        escalation=escalation,
    )
    assert result.switch_recommended_now is False
    assert result.next_regime is None
