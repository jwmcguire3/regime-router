from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import ARTIFACT_HINTS, ReentryJustification, RegimeConfidenceResult, Stage
from router.orchestration.stop_policy import StopPolicy
from router.routing import RegimeComposer
from router.state import RouterState


def _state_for(stage: Stage) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    return RouterState(
        task_id="task-stop-policy-vnext",
        task_summary="control surface stop policy test",
        current_bottleneck="test bottleneck",
        current_regime=regime,
        runner_up_regime=composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC),
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=[],
        uncertainties=[],
        contradictions=[],
        assumptions=[],
        risks=[],
        stage_goal="goal",
        switch_trigger=None,
        recommended_next_regime=None,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=0.0,
        task_classification={"likely_endpoint_regime": Stage.OPERATOR.value},
    )


def _valid_validation(stage: Stage) -> dict[str, object]:
    return {
        "is_valid": True,
        "parsed": {
            "regime": stage.value,
            "artifact_type": ARTIFACT_HINTS[stage],
            "artifact": {},
            "completion_signal": "done",
            "failure_signal": "",
        },
    }


def test_stop_policy_does_not_block_builder_by_recurrence_threshold() -> None:
    state = _state_for(Stage.OPERATOR)
    state.recurrence_potential = 1.0
    state.recommended_next_regime = RegimeComposer().compose(Stage.BUILDER)

    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result={"is_valid": False, "parsed": {"completion_signal": "", "failure_signal": ""}},
        routing_decision=None,
        current_stage=Stage.OPERATOR,
    )

    assert decision.should_stop is False
    assert decision.reason == "artifact_incomplete"


def test_stop_policy_refuses_endpoint_completion_on_artifact_stage_mismatch() -> None:
    state = _state_for(Stage.OPERATOR)
    validation = {
        "is_valid": True,
        "parsed": {
            "regime": Stage.EPISTEMIC.value,
            "artifact_type": ARTIFACT_HINTS[Stage.EPISTEMIC],
            "artifact": {},
            "completion_signal": "done",
            "failure_signal": "",
        },
    }

    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=validation,
        routing_decision=None,
        current_stage=Stage.OPERATOR,
    )

    assert decision.should_stop is False
    assert decision.reason == "artifact_complete_but_stage_artifact_mismatch"


def test_stop_policy_allows_justified_reentry_to_defer_stop() -> None:
    state = _state_for(Stage.SYNTHESIS)
    state.recommended_next_regime = RegimeComposer().compose(Stage.EPISTEMIC)
    state.last_reentry_justification = ReentryJustification(
        defect_class="contract_invalidated",
        repair_target="epistemic:evidence_refresh",
        contract_delta="next_stage_contract_changed",
        state_delta="recommended_next_stage_changed",
    )

    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_valid_validation(Stage.SYNTHESIS),
        routing_decision=None,
        current_stage=Stage.SYNTHESIS,
    )

    assert decision.should_stop is False
    assert decision.reason == "forward_progress_recommended"


def test_stop_policy_endpoint_completion_wins_over_justified_reentry() -> None:
    state = _state_for(Stage.OPERATOR)
    state.recommended_next_regime = RegimeComposer().compose(Stage.EPISTEMIC)
    state.last_reentry_justification = ReentryJustification(
        defect_class="contract_invalidated",
        repair_target="epistemic:evidence_refresh",
        contract_delta="next_stage_contract_changed",
        state_delta="recommended_next_stage_changed",
    )

    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_valid_validation(Stage.OPERATOR),
        routing_decision=None,
        current_stage=Stage.OPERATOR,
    )

    assert decision.should_stop is True
    assert decision.reason == "artifact_complete_at_or_past_endpoint:operator"


def test_stop_policy_does_not_defer_for_unjustified_same_or_lower_recommendation() -> None:
    state = _state_for(Stage.OPERATOR)
    state.recommended_next_regime = RegimeComposer().compose(Stage.EPISTEMIC)

    decision = StopPolicy().should_stop(
        router_state=state,
        validation_result=_valid_validation(Stage.OPERATOR),
        routing_decision=None,
        current_stage=Stage.OPERATOR,
    )

    assert decision.should_stop is True
    assert decision.reason == "artifact_complete_at_or_past_endpoint:operator"
