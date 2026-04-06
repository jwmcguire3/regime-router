import json

from router.models import CANONICAL_FAILURE_IF_OVERUSED, RegimeConfidenceResult, RegimeExecutionResult, RoutingDecision, Stage
from router.routing import RegimeComposer
from router.runtime.state_updater import build_router_state, update_router_state_from_execution
from router.state import RouterState, router_state_from_jsonable, to_jsonable


def _decision() -> RoutingDecision:
    return RoutingDecision(
        bottleneck="decide a resilient plan",
        primary_regime=Stage.SYNTHESIS,
        runner_up_regime=Stage.EPISTEMIC,
        why_primary_wins_now="synthesis pressure dominates",
        switch_trigger="frame_collapses_under_pressure_points",
        confidence=RegimeConfidenceResult.low_default(),
    )


def _result(parsed: dict) -> RegimeExecutionResult:
    return RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name="Synthesis Core",
        stage=Stage.SYNTHESIS,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(parsed),
        artifact_text=json.dumps(parsed.get("artifact", {})),
        validation={"parsed": parsed, "is_valid": True},
    )


def _state_with_pressures() -> RouterState:
    composer = RegimeComposer()
    return RouterState(
        task_id="task-heuristics",
        task_summary="heuristics",
        current_bottleneck="heuristics",
        current_regime=composer.compose(Stage.SYNTHESIS),
        runner_up_regime=composer.compose(Stage.EPISTEMIC),
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=[],
        uncertainties=[],
        contradictions=[],
        assumptions=[],
        risks=["risk"],
        stage_goal="goal",
        decision_pressure=5.0,
        fragility_pressure=5.0,
        possibility_space_need=5.0,
        synthesis_pressure=5.0,
        evidence_demand=2.0,
    )


def test_build_router_state_populates_pressures_and_risk_tags_from_analyzer(analyzer_output_fixture):
    composer = RegimeComposer()

    state = build_router_state(
        bottleneck="decide a resilient plan",
        decision=_decision(),
        regime=composer.compose(Stage.SYNTHESIS),
        composer=composer,
        analyzer_result=analyzer_output_fixture,
    )

    assert state.structural_signals == analyzer_output_fixture.structural_signals
    assert state.decision_pressure == float(analyzer_output_fixture.decision_pressure)
    assert state.fragility_pressure == float(analyzer_output_fixture.fragility_pressure)
    assert state.possibility_space_need == float(analyzer_output_fixture.possibility_space_need)
    assert state.synthesis_pressure == float(analyzer_output_fixture.synthesis_pressure)
    assert state.recurrence_potential == float(analyzer_output_fixture.recurrence_potential)
    assert state.risk_tags == set(analyzer_output_fixture.risk_tags)
    assert "High stakes" in state.risks
    assert CANONICAL_FAILURE_IF_OVERUSED[Stage.SYNTHESIS] in state.risks


def test_build_router_state_defaults_when_analyzer_result_is_absent():
    composer = RegimeComposer()

    state = build_router_state(
        bottleneck="decide a resilient plan",
        decision=_decision(),
        regime=composer.compose(Stage.SYNTHESIS),
        composer=composer,
        analyzer_result=None,
    )

    assert state.structural_signals == []
    assert state.risk_tags == set()
    assert state.decision_pressure == 0.0
    assert state.fragility_pressure == 0.0
    assert state.possibility_space_need == 0.0
    assert state.synthesis_pressure == 0.0
    assert state.evidence_demand == 0.0
    assert state.detected_markers == {}


def test_router_state_serialization_round_trip_preserves_new_fields():
    composer = RegimeComposer()
    state = _state_with_pressures()
    state.risk_tags = {"high_stakes", "operator_error_intolerant"}

    serialized = to_jsonable(state)
    restored = router_state_from_jsonable(serialized, composer.compose)

    assert restored is not None
    assert restored.synthesis_pressure == state.synthesis_pressure
    assert restored.risk_tags == state.risk_tags


def test_execution_heuristics_candidate_frames_increase_synthesis_and_reduce_possibility_need():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"candidate_frames": ["A", "B", "C"]}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.synthesis_pressure == 6.0
    assert state.possibility_space_need == 4.0


def test_execution_heuristics_central_claim_lowers_synthesis_pressure():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"central_claim": "Choose frame alpha"}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.synthesis_pressure == 4.0


def test_execution_heuristics_contradictions_increase_evidence_demand():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"contradictions": ["c1", "c2"]}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.evidence_demand == 3.0


def test_execution_heuristics_decision_and_tradeoff_lower_decision_pressure():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"decision": "Proceed", "tradeoff_accepted": "Accept slower speed"}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.decision_pressure == 4.0


def test_execution_heuristics_destabilizers_raise_fragility_pressure():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"top_destabilizers": ["hidden dependency"]}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.fragility_pressure == 6.0


def test_execution_heuristics_survivable_revisions_lower_fragility_pressure():
    state = _state_with_pressures()

    update_router_state_from_execution(
        state,
        _result({"artifact": {"survivable_revisions": ["add retry fallback"]}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert state.fragility_pressure == 4.0


def test_execution_heuristics_empty_artifact_leaves_pressures_unchanged():
    state = _state_with_pressures()
    baseline = (
        state.synthesis_pressure,
        state.possibility_space_need,
        state.evidence_demand,
        state.decision_pressure,
        state.fragility_pressure,
    )

    update_router_state_from_execution(
        state,
        _result({"artifact": {}}),
        reason_entered="initial",
        composer=RegimeComposer(),
    )

    assert (
        state.synthesis_pressure,
        state.possibility_space_need,
        state.evidence_demand,
        state.decision_pressure,
        state.fragility_pressure,
    ) == baseline
