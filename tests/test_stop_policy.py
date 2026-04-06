import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import ARTIFACT_HINTS, RegimeConfidenceResult, RegimeExecutionResult, RoutingDecision, Stage
from router.orchestration.stop_policy import StopPolicy
from router.orchestration.switch_orchestrator import SwitchOrchestrationResult
from router.routing import RegimeComposer
from router.runtime.session_runtime import SessionRuntime
from router.state import Handoff, RouterState


def _state_for(
    stage: Stage,
    *,
    recurrence_potential: float = 0.0,
    task_summary: str = "stop policy test",
    task_classification_endpoint: Stage = Stage.OPERATOR,
) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-stop-policy",
        task_summary=task_summary,
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
        switch_trigger=None,
        recommended_next_regime=None,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
        task_classification={"likely_endpoint_regime": task_classification_endpoint.value},
    )


def _decision(endpoint: Stage) -> RoutingDecision:
    return RoutingDecision(
        bottleneck="test",
        primary_regime=Stage.EXPLORATION,
        runner_up_regime=Stage.SYNTHESIS,
        why_primary_wins_now="test",
        switch_trigger="test",
        likely_endpoint_regime=endpoint.value,
    )


def _result(stage: Stage, *, is_valid: bool, completion_signal: str, failure_signal: str) -> RegimeExecutionResult:
    parsed = {
        "regime": stage.value,
        "purpose": "test purpose",
        "artifact_type": ARTIFACT_HINTS[stage],
        "artifact": {},
        "completion_signal": completion_signal,
        "failure_signal": failure_signal,
        "recommended_next_regime": "operator",
    }
    return RegimeExecutionResult(
        task="task",
        model="model",
        regime_name="Regime",
        stage=stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(parsed),
        artifact_text="{}",
        validation={"is_valid": is_valid, "parsed": parsed},
    )


class _NoopDetector:
    def detect(self, state, output):
        return object()


class _NoopEscalation:
    def evaluate(self, **kwargs):
        class _E:
            escalation_direction = "neutral"
            justification = "none"
            preferred_regime_biases = {}
            switch_pressure_adjustment = 0
            debug_signals = {}

        return _E()


class _StaticOrchestrator:
    def __init__(self, next_stage: Stage | None):
        self.next_stage = next_stage
        self._composer = RegimeComposer()

    def orchestrate(self, state, output, detection, **kwargs):
        if self.next_stage is None:
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="No switch",
                updated_state=state,
            )
        next_regime = self._composer.compose(self.next_stage)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(
            next_regime=next_regime,
            switch_recommended_now=True,
            reason_for_switch=f"Switch to {self.next_stage.value}",
            updated_state=state,
        )


def _runtime_with(next_stage: Stage | None) -> SessionRuntime:
    return SessionRuntime(
        misrouting_detector=_NoopDetector(),
        escalation_policy=_NoopEscalation(),
        switch_orchestrator=_StaticOrchestrator(next_stage),
        stop_policy=StopPolicy(),
    )


def _run_loop(state: RouterState, initial_result: RegimeExecutionResult, *, max_switches: int, endpoint: Stage) -> RegimeExecutionResult:
    runtime = _runtime_with(Stage.BUILDER)
    composer = RegimeComposer()

    def execute_regime_once(**kwargs):
        next_regime = kwargs["regime"]
        return _result(next_regime.stage, is_valid=False, completion_signal="", failure_signal="")

    def update_state_from_execution(st, result, *, reason_entered):
        st.current_regime = composer.compose(result.stage)
        st.record_regime_step(
            regime=st.current_regime,
            reason_entered=reason_entered,
            completion_signal_seen=bool(result.validation.get("is_valid", False)),
            failure_signal_seen=not bool(result.validation.get("is_valid", False)),
            outcome_summary=reason_entered,
        )

    return runtime.run_orchestration_loop(
        state=state,
        task="task",
        model="model",
        initial_result=initial_result,
        max_switches=max_switches,
        routing_decision=_decision(endpoint),
        execute_regime_once=execute_regime_once,
        update_state_from_execution=update_state_from_execution,
        handoff_from_state=lambda s: Handoff("", "", [], [], [], [], "", None, ""),
        compute_forward_handoff=lambda result, st, regime: Handoff("", "", [], [], [], [], "", None, ""),
    )



def test_artifact_aware_completion_framework_request_not_satisfied_by_dominant_frame():
    state = _state_for(
        Stage.SYNTHESIS,
        task_summary="Deliver a finished framework document with reusable sections and operating guidance.",
        task_classification_endpoint=Stage.SYNTHESIS,
    )
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(
            Stage.SYNTHESIS,
            is_valid=True,
            completion_signal="coherent_frame_stable",
            failure_signal="",
        ).validation,
        routing_decision=_decision(Stage.SYNTHESIS),
        current_stage=Stage.SYNTHESIS,
    )
    assert decision.should_stop is False


def test_artifact_aware_completion_worksheet_or_spec_not_satisfied_by_candidate_frame_set():
    state = _state_for(
        Stage.EXPLORATION,
        task_summary="Produce a final worksheet spec the team can execute directly.",
        task_classification_endpoint=Stage.EXPLORATION,
    )
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(
            Stage.EXPLORATION,
            is_valid=True,
            completion_signal="exploration_ready_for_selection",
            failure_signal="",
        ).validation,
        routing_decision=_decision(Stage.EXPLORATION),
        current_stage=Stage.EXPLORATION,
    )
    assert decision.should_stop is False


def test_artifact_aware_completion_task_explicitly_rejects_frame_as_final_output():
    state = _state_for(
        Stage.SYNTHESIS,
        task_summary=(
            "A frame or interpretation is not the final output. Provide the final deliverable as a completed memo."
        ),
        task_classification_endpoint=Stage.SYNTHESIS,
    )
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(
            Stage.SYNTHESIS,
            is_valid=True,
            completion_signal="coherent_frame_stable",
            failure_signal="",
        ).validation,
        routing_decision=_decision(Stage.SYNTHESIS),
        current_stage=Stage.SYNTHESIS,
    )
    assert decision.should_stop is False


def test_stop_policy_respects_requested_deliverable_beyond_decision_packet():
    state = _state_for(
        Stage.OPERATOR,
        task_summary="Return a reusable implementation spec document, not only an operator decision packet.",
        task_classification_endpoint=Stage.OPERATOR,
    )
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(
            Stage.OPERATOR,
            is_valid=True,
            completion_signal="decision_committed_with_actions",
            failure_signal="",
        ).validation,
        routing_decision=_decision(Stage.OPERATOR),
        current_stage=Stage.OPERATOR,
    )
    assert decision.should_stop is False


def test_stop_at_operator_default():
    state = _state_for(Stage.OPERATOR)
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(Stage.OPERATOR, is_valid=True, completion_signal="decision_ready", failure_signal="").validation,
        routing_decision=_decision(Stage.OPERATOR),
        current_stage=Stage.OPERATOR,
    )
    assert decision.should_stop is True


def test_stop_does_not_fire_before_endpoint():
    state = _state_for(Stage.EXPLORATION)
    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_result(Stage.EXPLORATION, is_valid=True, completion_signal="exploration_ready", failure_signal="").validation,
        routing_decision=_decision(Stage.OPERATOR),
        current_stage=Stage.EXPLORATION,
    )
    assert decision.should_stop is False


def test_builder_not_blocked_low_recurrence():
    state = _state_for(Stage.OPERATOR, recurrence_potential=3)
    result = _run_loop(
        state,
        _result(Stage.OPERATOR, is_valid=False, completion_signal="", failure_signal=""),
        max_switches=2,
        endpoint=Stage.BUILDER,
    )
    assert result.stage == Stage.BUILDER
    assert state.current_regime.stage == Stage.BUILDER


def test_builder_allowed_high_recurrence():
    state = _state_for(Stage.OPERATOR, recurrence_potential=8)
    result = _run_loop(
        state,
        _result(Stage.OPERATOR, is_valid=False, completion_signal="", failure_signal=""),
        max_switches=2,
        endpoint=Stage.BUILDER,
    )
    assert result.stage == Stage.BUILDER
    assert state.current_regime.stage == Stage.BUILDER


def test_max_switches_still_hard_ceiling():
    state = _state_for(Stage.EXPLORATION)
    state.switches_executed = 1
    result = _run_loop(
        state,
        _result(Stage.EXPLORATION, is_valid=False, completion_signal="", failure_signal=""),
        max_switches=1,
        endpoint=Stage.BUILDER,
    )
    assert result.stage == Stage.EXPLORATION
    assert state.orchestration_stop_reason == "switch_limit_reached"


def test_stop_reason_recorded():
    state = _state_for(Stage.OPERATOR)
    runtime = _runtime_with(None)
    initial_result = _result(Stage.OPERATOR, is_valid=True, completion_signal="decision_ready", failure_signal="")
    runtime.run_orchestration_loop(
        state=state,
        task="task",
        model="model",
        initial_result=initial_result,
        max_switches=2,
        routing_decision=_decision(Stage.OPERATOR),
        execute_regime_once=lambda **kwargs: initial_result,
        update_state_from_execution=lambda *args, **kwargs: None,
        handoff_from_state=lambda s: Handoff("", "", [], [], [], [], "", None, ""),
        compute_forward_handoff=lambda result, st, regime: Handoff("", "", [], [], [], [], "", None, ""),
    )
    assert state.orchestration_stop_reason == "artifact_complete_at_or_past_endpoint:operator"


def test_orchestration_execution_inputs_come_from_state():
    state = _state_for(Stage.OPERATOR)
    state.structural_signals = ["sig_from_state"]
    state.risk_tags = {"risk_from_state"}
    runtime = _runtime_with(Stage.BUILDER)
    captured: dict[str, object] = {}

    def execute_regime_once(**kwargs):
        captured["task_signals"] = kwargs["task_signals"]
        captured["risk_profile"] = kwargs["risk_profile"]
        next_regime = kwargs["regime"]
        return _result(next_regime.stage, is_valid=False, completion_signal="", failure_signal="")

    runtime.run_orchestration_loop(
        state=state,
        task="task",
        model="model",
        initial_result=_result(Stage.OPERATOR, is_valid=False, completion_signal="", failure_signal=""),
        max_switches=1,
        routing_decision=_decision(Stage.BUILDER),
        execute_regime_once=execute_regime_once,
        update_state_from_execution=lambda *args, **kwargs: None,
        handoff_from_state=lambda s: Handoff("", "", [], [], [], [], "", None, ""),
        compute_forward_handoff=lambda result, st, regime: Handoff("", "", [], [], [], [], "", None, ""),
    )

    assert captured["task_signals"] == ["sig_from_state"]
    assert captured["risk_profile"] == {"risk_from_state"}
