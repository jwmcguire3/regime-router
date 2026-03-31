import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector, RegimeOutputContract, SwitchOrchestrator
from router.models import RegimeConfidenceResult, Stage
from router.orchestration.transition_rules import control_failure_regime_mismatch
from router.routing import RegimeComposer
from router.state import RouterState


def _state_for(stage: Stage, *, recurrence_potential: float = 0.0) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-orchestration-fix",
        task_summary="orchestration fixes",
        current_bottleneck="test bottleneck",
        current_regime=regime,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=["unknown"],
        contradictions=[],
        assumptions=[],
        risks=["risk"],
        stage_goal="goal",
        switch_trigger="trigger",
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
    )


def _operator_output(*, regime: str, control_failures: list[str] | None = None) -> RegimeOutputContract:
    payload = {
        "regime": regime,
        "purpose": "operator purpose",
        "artifact_type": "operator_decision",
        "completion_signal": "decision_ready_for_execution",
        "failure_signal": "",
        "recommended_next_regime": "operator",
        "artifact": {
            "decision": "Choose path A",
            "rationale": "Evidence supports path A under current constraints.",
            "tradeoff_accepted": "Accept slower speed for lower downside.",
            "next_actions": ["Do step 1", "Do step 2"],
            "fallback_trigger": "Fallback if metric drops by 15%.",
            "review_point": "Review in one week.",
        },
    }
    validation = {
        "parsed": payload,
        "valid_json": True,
        "required_keys_present": True,
        "artifact_fields_present": True,
        "artifact_type_matches": True,
        "contract_controls_valid": True,
        "semantic_valid": True,
        "control_failures": control_failures or [],
    }
    return RegimeOutputContract(stage=Stage.OPERATOR, raw_response=json.dumps(payload), validation=validation)


def test_regime_field_mismatch_triggers_misrouting_detection():
    state = _state_for(Stage.OPERATOR)
    output = _operator_output(regime="synthesis")

    result = MisroutingDetector().detect(state, output)

    assert result.misrouting_detected is True
    assert result.recommended_next_regime is not None
    assert result.recommended_next_regime.stage == Stage.SYNTHESIS
    assert "different regime" in result.justification.lower()


def test_control_failure_regime_mismatch_identifies_pattern():
    output = _operator_output(
        regime="operator",
        control_failures=["regime field mismatch: expected operator got synthesis"],
    )

    assert control_failure_regime_mismatch(output) is True



def test_switch_orchestrator_routes_operator_to_epistemic_on_regime_mismatch_control_failure():
    state = _state_for(Stage.OPERATOR)
    output = _operator_output(
        regime="operator",
        control_failures=["regime field mismatch: expected operator got synthesis"],
    )
    detection = MisroutingDetector().detect(state, output)

    result = SwitchOrchestrator().orchestrate(state, output, detection, switches_used=0, max_switches=2)

    assert result.switch_recommended_now is True
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.EPISTEMIC



def test_non_mismatch_cases_unchanged():
    state = _state_for(Stage.OPERATOR)
    output = _operator_output(regime="operator", control_failures=[])
    detection = MisroutingDetector().detect(state, output)

    assert detection.misrouting_detected is False
    assert control_failure_regime_mismatch(output) is False

    result = SwitchOrchestrator().orchestrate(state, output, detection, switches_used=0, max_switches=2)

    assert result.switch_recommended_now is False
    assert result.next_regime is None
