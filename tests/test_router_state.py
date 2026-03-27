import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeConfidenceResult, Stage
from router.routing import RegimeComposer
from router.state import RouterState


def _make_state() -> RouterState:
    composer = RegimeComposer()
    current = composer.compose(Stage.SYNTHESIS, risk_profile={"false_unification"}, handoff_expected=True)
    runner_up = composer.compose(Stage.ADVERSARIAL, risk_profile={"false_unification"}, handoff_expected=True)
    return RouterState(
        task_id="task-123",
        task_summary="Find strongest interpretation",
        current_bottleneck="Find strongest interpretation with constraints",
        current_regime=current,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["Known A"],
        uncertainties=["Unknown A"],
        contradictions=["Contradiction A"],
        assumptions=["Assumption A"],
        risks=["Risk A"],
        stage_goal="Produce typed artifact",
        switch_trigger="If evidence breaks frame",
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=1.0,
    )


def test_record_regime_step_appends_prior_regimes():
    state = _make_state()
    assert state.prior_regimes == []

    state.record_regime_step(
        regime=Stage.SYNTHESIS,
        reason_entered="Primary route won.",
        completion_signal_seen=True,
        failure_signal_seen=False,
        outcome_summary="Produced useful frame.",
    )

    assert len(state.prior_regimes) == 1
    assert state.prior_regimes[0].regime == Stage.SYNTHESIS
    assert state.prior_regimes[0].completion_signal_seen is True


def test_apply_dominant_frame_updates_value():
    state = _make_state()
    assert state.dominant_frame is None

    state.apply_dominant_frame("Frame A")
    assert state.dominant_frame == "Frame A"


def test_update_inference_state_preserves_and_updates_fields():
    state = _make_state()
    original_uncertainties = list(state.uncertainties)

    state.update_inference_state(
        contradictions=state.contradictions + ["Contradiction B"],
        assumptions=state.assumptions + ["Assumption B"],
    )

    assert "Contradiction A" in state.contradictions
    assert "Contradiction B" in state.contradictions
    assert "Assumption A" in state.assumptions
    assert "Assumption B" in state.assumptions
    assert state.uncertainties == original_uncertainties
