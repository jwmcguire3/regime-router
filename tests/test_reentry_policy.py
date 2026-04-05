import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetectionResult, RegimeOutputContract, SwitchOrchestrator
from router.models import ReentryJustification, RegimeConfidenceResult, RegimeExecutionResult, Stage
from router.orchestration.canonical_status import CanonicalStatus
from router.orchestration.transition_rules import next_stage
from router.routing import RegimeComposer
from router.runtime.session_runtime import SessionRuntime
from router.runtime.state_updater import update_router_state_from_execution
from router.state import RouterState


def _state_for(stage: Stage) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-reentry",
        task_summary="reentry policy tests",
        current_bottleneck="repeating operational workflow under changing constraints",
        current_regime=regime,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame="initial-frame",
        knowns=["known"],
        uncertainties=["u1"],
        contradictions=[],
        assumptions=["a1"],
        substantive_assumptions=["a1"],
        risks=["active risk: unresolved contract gap"],
        stage_goal="goal",
        planned_switch_condition="planned",
        observed_switch_cause=None,
        switch_trigger=None,
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=3.0,
    )


def _runtime() -> SessionRuntime:
    class _Stub:
        def detect(self, *args, **kwargs):
            raise NotImplementedError

        def evaluate(self, *args, **kwargs):
            raise NotImplementedError

        def should_stop(self, *args, **kwargs):
            raise NotImplementedError

    return SessionRuntime(
        misrouting_detector=_Stub(),  # type: ignore[arg-type]
        escalation_policy=_Stub(),  # type: ignore[arg-type]
        switch_orchestrator=_Stub(),  # type: ignore[arg-type]
        stop_policy=_Stub(),  # type: ignore[arg-type]
    )


def test_same_stage_retry_denied_without_state_delta():
    runtime = _runtime()
    state = _state_for(Stage.EPISTEMIC)
    state.executed_regime_stages = [Stage.EPISTEMIC]
    state.last_state_delta = "no_material_state_delta"
    state.last_reentry_justification = ReentryJustification(
        defect_class="evidence_failure",
        repair_target="epistemic:strengthen_evidence_quality",
        contract_delta="artifact_contract_advanced",
        state_delta="no_material_state_delta",
    )
    decision = runtime._evaluate_reentry(state=state, next_stage=Stage.EPISTEMIC, reason_for_switch="retry")
    assert decision.allowed is False


def test_same_stage_retry_allowed_with_contract_and_state_delta():
    runtime = _runtime()
    state = _state_for(Stage.EPISTEMIC)
    state.executed_regime_stages = [Stage.EPISTEMIC]
    state.last_state_delta = "recommended_next_stage_changed"
    state.last_reentry_justification = ReentryJustification(
        defect_class="evidence_failure",
        repair_target="epistemic:strengthen_evidence_quality",
        contract_delta="next_stage_contract_changed",
        state_delta="recommended_next_stage_changed",
    )
    decision = runtime._evaluate_reentry(state=state, next_stage=Stage.EPISTEMIC, reason_for_switch="retry with new evidence")
    assert decision.allowed is True


def test_prior_stage_reentry_allowed_when_defect_class_present():
    runtime = _runtime()
    state = _state_for(Stage.ADVERSARIAL)
    state.executed_regime_stages = [Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL]
    state.last_state_delta = "contradictions_changed"
    state.last_reentry_justification = ReentryJustification(
        defect_class="break_condition_discovery",
        repair_target="synthesis:incorporate_break_conditions",
        contract_delta="next_stage_contract_changed",
        state_delta="contradictions_changed",
    )
    decision = runtime._evaluate_reentry(state=state, next_stage=Stage.SYNTHESIS, reason_for_switch="repair synthesis")
    assert decision.allowed is True


def test_prior_stage_reentry_denied_without_justification():
    runtime = _runtime()
    state = _state_for(Stage.ADVERSARIAL)
    state.executed_regime_stages = [Stage.SYNTHESIS, Stage.ADVERSARIAL]
    state.last_state_delta = "contradictions_changed"
    state.last_reentry_justification = None
    decision = runtime._evaluate_reentry(state=state, next_stage=Stage.SYNTHESIS, reason_for_switch="repair synthesis")
    assert decision.allowed is False


def test_ping_pong_denied_for_repeated_same_cause():
    runtime = _runtime()
    state = _state_for(Stage.SYNTHESIS)
    state.last_state_delta = "no_material_state_delta"
    state.observed_switch_cause = "evidence_failure"
    state.record_switch_decision(
        switch_index=1,
        from_stage=Stage.SYNTHESIS,
        to_stage=Stage.EPISTEMIC,
        switch_recommended=True,
        switch_executed=True,
        reason="first",
        planned_switch_condition="planned",
        observed_switch_cause="evidence_failure",
    )
    state.record_switch_decision(
        switch_index=2,
        from_stage=Stage.EPISTEMIC,
        to_stage=Stage.SYNTHESIS,
        switch_recommended=True,
        switch_executed=True,
        reason="second",
        planned_switch_condition="planned",
        observed_switch_cause="evidence_failure",
    )
    state.last_reentry_justification = ReentryJustification(
        defect_class="evidence_failure",
        repair_target="epistemic:strengthen_evidence_quality",
        contract_delta="next_stage_contract_changed",
        state_delta="no_material_state_delta",
    )
    decision = runtime._evaluate_reentry(state=state, next_stage=Stage.EPISTEMIC, reason_for_switch="third")
    assert decision.allowed is False


def test_collapse_is_one_reentry_class_not_the_only_one():
    state = _state_for(Stage.OPERATOR)
    detection = MisroutingDetectionResult(
        current_regime=state.current_regime,
        dominant_failure_mode="forced closure",
        still_productive=False,
        misrouting_detected=True,
        justification="switch",
        recommended_next_regime=state.resolve_regime(Stage.EPISTEMIC, RegimeComposer().compose),
    )
    output = RegimeOutputContract(
        stage=Stage.OPERATOR,
        raw_response="{}",
        validation={
            "parsed": {
                "completion_signal": "",
                "failure_signal": "decision_not_actionable_under_constraints",
                "recommended_next_regime": "epistemic",
                "artifact": {},
            },
            "valid_json": True,
            "required_keys_present": True,
            "artifact_fields_present": True,
            "artifact_type_matches": True,
            "contract_controls_valid": True,
        },
    )
    state.last_state_delta = "recommended_next_stage_changed"
    state.last_contract_delta = "contract_invalidated_by_control_conflict"
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, detection, switches_used=0, max_switches=3)
    assert result.switch_recommended_now is True
    assert state.last_reentry_justification is not None
    assert state.last_reentry_justification.defect_class == "decision_non_actionable"


def test_last_state_and_contract_delta_storage():
    state = _state_for(Stage.SYNTHESIS)
    result = RegimeExecutionResult(
        task="task",
        model="m",
        regime_name=state.current_regime.name,
        stage=Stage.SYNTHESIS,
        system_prompt="",
        user_prompt="",
        raw_response="{}",
        artifact_text="",
        validation={
            "parsed": {
                "completion_signal": "coherent_frame_stable",
                "failure_signal": "",
                "recommended_next_regime": "epistemic",
                "artifact": {"central_claim": "new frame"},
            },
            "valid_json": True,
            "required_keys_present": True,
            "artifact_fields_present": True,
            "artifact_type_matches": True,
            "contract_controls_valid": True,
            "is_valid": True,
            "semantic_failures": [],
        },
    )
    update_router_state_from_execution(state, result, reason_entered="test", composer=RegimeComposer())
    assert state.last_state_delta in {"dominant_frame_changed", "recommended_next_stage_changed"}
    assert state.last_contract_delta in {"next_stage_contract_changed", "artifact_contract_advanced"}


def test_builder_not_forced_from_operator_by_recurrence_alone():
    state = _state_for(Stage.OPERATOR)
    detection = MisroutingDetectionResult(
        current_regime=state.current_regime,
        dominant_failure_mode="forced closure",
        still_productive=True,
        misrouting_detected=False,
        justification="stay",
        recommended_next_regime=None,
    )
    output = RegimeOutputContract(
        stage=Stage.OPERATOR,
        raw_response=json.dumps({"artifact": {"decision": "ship one-off fix"}}),
        validation={"parsed": {"artifact": {"decision": "ship one-off fix"}}},
    )
    chosen = next_stage(
        state,
        detection,
        None,
        output,
        canonical=CanonicalStatus(
            terminal_signal="completion",
            artifact_status="valid_complete",
            switch_posture="stay",
            completion_signal="decision_ready_for_execution",
            failure_signal="",
            is_valid=True,
            structurally_valid=True,
            semantic_valid=True,
            control_conflict=False,
            recommended_next_stage=None,
        ),
    )
    assert chosen is None


def test_justified_reentry_from_semantic_invalidation_with_sparse_signals():
    state = _state_for(Stage.OPERATOR)
    state.assumptions = []
    state.contradictions = []
    state.last_state_delta = "semantic_failures_introduced"
    state.last_contract_delta = "contract_invalidated_by_semantic_failure"
    detection = MisroutingDetectionResult(
        current_regime=state.current_regime,
        dominant_failure_mode="forced closure",
        still_productive=False,
        misrouting_detected=False,
        justification="incomplete",
        recommended_next_regime=None,
    )
    output = RegimeOutputContract(
        stage=Stage.OPERATOR,
        raw_response="{}",
        validation={
            "parsed": {
                "completion_signal": "",
                "failure_signal": "",
                "recommended_next_regime": "epistemic",
                "artifact": {
                    "decision": "Proceed now",
                    "rationale": "Placeholder rationale",
                    "tradeoff_accepted": "Accept known downside",
                    "next_actions": ["act"],
                    "fallback_trigger": "if fails",
                    "review_point": "soon",
                },
            },
            "semantic_failures": ["decision field uses placeholder token"],
            "valid_json": True,
            "required_keys_present": True,
            "artifact_fields_present": True,
            "artifact_type_matches": True,
            "contract_controls_valid": True,
            "semantic_valid": False,
        },
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, detection, switches_used=0, max_switches=3)
    assert result.switch_recommended_now is True
    assert result.next_regime is not None
    assert result.next_regime.stage in {Stage.EPISTEMIC, Stage.EXPLORATION}
    assert state.observed_switch_cause in {"contract_invalidated", "assumption_or_frame_collapse"}
