import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeConfidenceResult, RegimeExecutionResult, RoutingDecision, Stage
from router.orchestration.stop_policy import StopPolicy
from router.orchestration.switch_orchestrator import SwitchOrchestrationResult
from router.orchestration.transition_rules import assumption_or_frame_collapse
from router.routing import RegimeComposer
from router.runtime.session_runtime import SessionRuntime
from router.runtime.state_updater import build_router_state, update_router_state_from_execution
from router.state import Handoff, RouterState


class _NoopDetector:
    def detect(self, state, output):
        return SimpleNamespace(misrouting_detected=False)


class _NoopEscalation:
    def evaluate(self, **kwargs):
        return SimpleNamespace(
            escalation_direction="neutral",
            justification="none",
            preferred_regime_biases={},
            switch_pressure_adjustment=0,
            debug_signals={},
        )


class _CountingOrchestrator:
    def __init__(self, next_stage: Stage | None):
        self.next_stage = next_stage
        self.calls = 0
        self.composer = RegimeComposer()

    def orchestrate(self, state, output, detection, **kwargs):
        self.calls += 1
        if self.next_stage is None:
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="No switch",
                updated_state=state,
            )
        next_regime = self.composer.compose(self.next_stage)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(
            next_regime=next_regime,
            switch_recommended_now=True,
            reason_for_switch=f"Switch to {self.next_stage.value}",
            updated_state=state,
        )


def _make_state(stage: Stage, *, assumptions=None, contradictions=None, executed_stages=None) -> RouterState:
    composer = RegimeComposer()
    return RouterState(
        task_id="task-fallback-regressions",
        task_summary="fallback regression coverage",
        current_bottleneck="test bottleneck",
        current_regime=composer.compose(stage),
        runner_up_regime=composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC),
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=["unknown"],
        contradictions=list(contradictions or []),
        assumptions=list(assumptions or []),
        risks=["risk"],
        stage_goal="goal",
        switch_trigger="switch-trigger",
        recommended_next_regime=None,
        executed_regime_stages=list(executed_stages or []),
        task_classification={"likely_endpoint_regime": Stage.OPERATOR.value},
    )


def _make_result(stage: Stage, *, is_valid: bool, completion_signal: str, failure_signal: str) -> RegimeExecutionResult:
    parsed = {
        "regime": stage.value,
        "purpose": "test",
        "artifact_type": "test_artifact",
        "artifact": {},
        "completion_signal": completion_signal,
        "failure_signal": failure_signal,
        "recommended_next_regime": "operator",
    }
    return RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name=f"{stage.value.title()} Core",
        stage=stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(parsed),
        artifact_text="{}",
        validation={"is_valid": is_valid, "parsed": parsed},
    )


def _run_loop(runtime: SessionRuntime, state: RouterState, initial_result: RegimeExecutionResult) -> RegimeExecutionResult:
    return runtime.run_orchestration_loop(
        state=state,
        task="task",
        model="fake",
        initial_result=initial_result,
        task_signals=[],
        risk_profile=set(),
        routing_features=SimpleNamespace(),
        max_switches=3,
        routing_decision=RoutingDecision(
            bottleneck="test",
            primary_regime=Stage.EXPLORATION,
            runner_up_regime=Stage.SYNTHESIS,
            why_primary_wins_now="test",
            switch_trigger="test",
            likely_endpoint_regime=Stage.OPERATOR.value,
        ),
        execute_regime_once=lambda **kwargs: initial_result,
        update_state_from_execution=lambda *args, **kwargs: None,
        handoff_from_state=lambda s: Handoff("", "", [], [], [], [], "", None, ""),
        compute_forward_handoff=lambda result, st, regime: Handoff("", "", [], [], [], [], "", None, ""),
    )


def test_fallback_stop_policy_preempts_collapse_signal():
    state = _make_state(Stage.OPERATOR, assumptions=["a1"], contradictions=["c1"])
    orchestrator = _CountingOrchestrator(next_stage=Stage.EXPLORATION)
    runtime = SessionRuntime(
        misrouting_detector=_NoopDetector(),
        escalation_policy=_NoopEscalation(),
        switch_orchestrator=orchestrator,
        stop_policy=StopPolicy(),
    )
    initial_result = _make_result(
        Stage.OPERATOR,
        is_valid=True,
        completion_signal="decision_ready",
        failure_signal="assumption collapse detected",
    )

    _run_loop(runtime, state, initial_result)

    assert orchestrator.calls == 0
    assert state.orchestration_stop_reason == "artifact_complete_at_or_past_endpoint:operator"


@pytest.mark.xfail(reason="Gap 2: collapse detection relies on brittle keyword matching")
def test_collapse_semantic_phrase_should_trigger_but_is_missed():
    state = _make_state(Stage.SYNTHESIS, assumptions=["a1"], contradictions=["c1"])
    assert assumption_or_frame_collapse(state, "Core premise invalidated by evidence") is True


def test_collapse_semantic_phrase_currently_missed():
    state = _make_state(Stage.SYNTHESIS, assumptions=["a1"], contradictions=["c1"])
    assert assumption_or_frame_collapse(state, "Core premise invalidated by evidence") is False


def test_collapse_assumption_guard_is_noop_after_state_build():
    composer = RegimeComposer()
    decision = RoutingDecision(
        bottleneck="test",
        primary_regime=Stage.EXPLORATION,
        runner_up_regime=Stage.SYNTHESIS,
        why_primary_wins_now="test",
        switch_trigger="test",
    )

    state_without_signals = build_router_state(
        bottleneck="test",
        decision=decision,
        regime=composer.compose(Stage.EXPLORATION),
        signals=[],
        risks=set(),
        features=SimpleNamespace(decision_pressure=0, evidence_demand=0, recurrence_potential=0),
        composer=composer,
    )
    state_with_signals = build_router_state(
        bottleneck="test",
        decision=decision,
        regime=composer.compose(Stage.EXPLORATION),
        signals=["marker-a"],
        risks=set(),
        features=SimpleNamespace(decision_pressure=0, evidence_demand=0, recurrence_potential=0),
        composer=composer,
    )

    assert bool(state_without_signals.assumptions) is True
    assert bool(state_with_signals.assumptions) is True


def test_reentry_exploration_fallback_requires_complete_justification():
    state = _make_state(
        Stage.SYNTHESIS,
        assumptions=["a1"],
        contradictions=["c1"],
        executed_stages=[Stage.EXPLORATION],
    )
    runtime = SessionRuntime(
        misrouting_detector=_NoopDetector(),
        escalation_policy=_NoopEscalation(),
        switch_orchestrator=_CountingOrchestrator(next_stage=Stage.EXPLORATION),
        stop_policy=StopPolicy(),
    )
    initial_result = _make_result(
        Stage.SYNTHESIS,
        is_valid=False,
        completion_signal="",
        failure_signal="assumption collapse",
    )

    _run_loop(runtime, state, initial_result)

    assert state.orchestration_stop_reason == "loop_prevented_reentry"
    assert "previously visited stage without material state delta" in state.switch_history[-1].reason


def test_invalid_output_recovery_empty_output_does_not_count_as_progress_and_falls_back():
    state = _make_state(Stage.OPERATOR)
    runtime = SessionRuntime(
        misrouting_detector=_NoopDetector(),
        escalation_policy=_NoopEscalation(),
        switch_orchestrator=_CountingOrchestrator(next_stage=None),
        stop_policy=StopPolicy(),
    )
    initial_result = _make_result(
        Stage.OPERATOR,
        is_valid=False,
        completion_signal="",
        failure_signal="",
    )

    _run_loop(runtime, state, initial_result)

    assert state.orchestration_stop_reason != "switch_not_recommended"
    assert any(record.switch_recommended and record.to_stage == Stage.EXPLORATION for record in state.switch_history)


def test_unusable_output_preserves_truthful_failure_bookkeeping():
    composer = RegimeComposer()
    state = _make_state(Stage.OPERATOR)
    unusable_result = _make_result(
        Stage.OPERATOR,
        is_valid=True,
        completion_signal="",
        failure_signal="decision_not_actionable_under_constraints",
    )

    update_router_state_from_execution(
        state,
        unusable_result,
        reason_entered="switch",
        composer=composer,
    )

    assert state.prior_regimes
    step = state.prior_regimes[-1]
    assert step.completion_signal_seen is False
    assert step.failure_signal_seen is True
