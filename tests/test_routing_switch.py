import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeConfidenceResult, Stage
from router.routing import RegimeComposer, Router
from router.state import RouterState


def _state(
    *,
    current_stage: Stage = Stage.EXPLORATION,
    decision_pressure: float = 0.0,
    synthesis_pressure: float = 0.0,
    possibility_space_need: float = 0.0,
    evidence_demand: float = 0.0,
    fragility_pressure: float = 0.0,
    recurrence_potential: float = 0.0,
    executed_regime_stages: list[Stage] | None = None,
) -> RouterState:
    composer = RegimeComposer()
    current_regime = composer.compose(current_stage)
    runner_up_stage = Stage.SYNTHESIS if current_stage != Stage.SYNTHESIS else Stage.EPISTEMIC
    return RouterState(
        task_id="task-test",
        task_summary="routing switch test task",
        current_bottleneck="switch bottleneck",
        current_regime=current_regime,
        runner_up_regime=composer.compose(runner_up_stage),
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=[],
        contradictions=[],
        assumptions=[],
        risks=["risk"],
        stage_goal="goal",
        decision_pressure=decision_pressure,
        fragility_pressure=fragility_pressure,
        possibility_space_need=possibility_space_need,
        synthesis_pressure=synthesis_pressure,
        evidence_demand=evidence_demand,
        recurrence_potential=recurrence_potential,
        executed_regime_stages=list(executed_regime_stages or []),
    )


def test_route_switch_high_decision_pressure_prefers_operator() -> None:
    decision = Router().route_switch(
        _state(
            decision_pressure=6.0,
            synthesis_pressure=2.0,
            possibility_space_need=1.0,
        )
    )
    assert decision.primary_regime == Stage.OPERATOR


def test_route_switch_high_synthesis_pressure_prefers_synthesis() -> None:
    decision = Router().route_switch(
        _state(
            synthesis_pressure=6.0,
            decision_pressure=2.0,
            possibility_space_need=1.0,
        )
    )
    assert decision.primary_regime == Stage.SYNTHESIS


def test_route_switch_all_zero_scores_falls_back_to_exploration() -> None:
    decision = Router().route_switch(_state())
    assert decision.primary_regime == Stage.EXPLORATION
    assert decision.inference_quality == "state_led"


def test_route_switch_penalizes_last_executed_stage_unless_pressure_overcomes_penalty() -> None:
    router = Router()
    state = _state(decision_pressure=2.0, synthesis_pressure=2.0, executed_regime_stages=[Stage.OPERATOR])
    decision = router.route_switch(state)
    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.deterministic_stage_scores[Stage.OPERATOR] == 1

    recovered = router.route_switch(
        _state(decision_pressure=4.0, synthesis_pressure=2.0, executed_regime_stages=[Stage.OPERATOR])
    )
    assert recovered.primary_regime == Stage.OPERATOR
    assert recovered.deterministic_stage_scores[Stage.OPERATOR] == 3


def test_route_switch_responds_to_updated_state_pressures() -> None:
    router = Router()
    state = _state(possibility_space_need=3.0, synthesis_pressure=1.0)
    first_decision = router.route_switch(state)
    assert first_decision.primary_regime == Stage.EXPLORATION

    state.possibility_space_need = 0.0
    state.synthesis_pressure = 5.0
    switched_decision = router.route_switch(state)
    assert switched_decision.primary_regime == Stage.SYNTHESIS
