import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector, RegimeOutputContract
from router.models import Regime, RegimeConfidenceResult, Stage
from router.routing import RegimeComposer
from router.state import RouterState


def make_state(
    stage: Stage,
    contradictions: list[str] | None = None,
    assumptions: list[str] | None = None,
    recurrence_potential: float = 0.0,
) -> RouterState:
    composer = RegimeComposer()
    current = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id=f"task-{stage.value}",
        task_summary=f"misrouting harness for {stage.value}",
        current_bottleneck="test bottleneck",
        current_regime=current,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known"],
        uncertainties=["unknown"],
        contradictions=list(contradictions or []),
        assumptions=list(assumptions or []),
        risks=["risk"],
        stage_goal="goal",
        switch_trigger=None,
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
    )


def make_output(
    stage: Stage,
    artifact: dict,
    completion_signal: str = "",
    failure_signal: str = "",
    recommended_next_regime: str = "synthesis",
) -> RegimeOutputContract:
    payload = {
        "regime": stage.value,
        "purpose": "test",
        "artifact_type": "test_artifact",
        "artifact": artifact,
        "completion_signal": completion_signal,
        "failure_signal": failure_signal,
        "recommended_next_regime": recommended_next_regime,
    }
    return RegimeOutputContract(
        stage=stage,
        raw_response=json.dumps(payload),
        validation={"parsed": payload},
    )


def _assert_recommendation_contract(result, *, misrouted: bool) -> None:
    if misrouted:
        assert isinstance(result.recommended_next_regime, Regime)
        assert result.recommended_next_regime.stage in set(Stage)
    else:
        assert result.recommended_next_regime is None


def test_exploration_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.EXPLORATION),
        make_output(
            Stage.EXPLORATION,
            {
                "candidate_frames": [
                    "Focus on onboarding funnel friction reduction",
                    "Focus on pricing clarity and packaging",
                    "Focus on post-signup activation loops",
                ],
                "selection_criteria": "Choose frame with strongest near-term retention lift.",
                "unresolved_axes": ["SMB vs enterprise", "self-serve vs assisted"],
            },
        ),
    )
    assert healthy.still_productive is True
    assert healthy.misrouting_detected is False
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.EXPLORATION),
        make_output(
            Stage.EXPLORATION,
            {
                "candidate_frames": [
                    "same frame",
                    "same frame",
                    "same frame",
                    "same frame",
                    "same frame",
                    "same frame",
                ],
                "selection_criteria": "",
                "unresolved_axes": "",
            },
        ),
    )
    assert failing.misrouting_detected is True
    assert failing.recommended_next_regime is not None
    _assert_recommendation_contract(failing, misrouted=True)


def test_synthesis_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.SYNTHESIS),
        make_output(
            Stage.SYNTHESIS,
            {
                "central_claim": "Retention is constrained by time-to-value.",
                "organizing_idea": "Collapse setup overhead before first success.",
                "supporting_structure": ["Template defaults", "Guided milestone flow"],
                "pressure_points": ["Fails if templates are low quality"],
            },
        ),
    )
    assert healthy.still_productive is True
    assert healthy.misrouting_detected is False
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.SYNTHESIS),
        make_output(
            Stage.SYNTHESIS,
            {
                "central_claim": "One policy explains all outcomes.",
                "organizing_idea": "Single global mechanism.",
                "supporting_structure": "",
                "pressure_points": "",
            },
        ),
    )
    assert failing.misrouting_detected is True
    _assert_recommendation_contract(failing, misrouted=True)


def test_epistemic_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.EPISTEMIC),
        make_output(
            Stage.EPISTEMIC,
            {
                "supported_claims": ["Cohort A has higher activation after guided setup."],
                "plausible_but_unproven": ["Higher activation will improve renewal."],
                "contradictions": ["Enterprise cohort behaves differently."],
            },
        ),
    )
    assert healthy.still_productive is True
    assert healthy.misrouting_detected is False
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.EPISTEMIC, contradictions=[], assumptions=[]),
        make_output(
            Stage.EPISTEMIC,
            {
                "supported_claims": "",
                "plausible_but_unproven": "",
                "contradictions": "",
            },
        ),
    )
    assert failing.misrouting_detected is True
    _assert_recommendation_contract(failing, misrouted=True)


def test_adversarial_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.ADVERSARIAL),
        make_output(
            Stage.ADVERSARIAL,
            {
                "top_destabilizers": ["objection A"],
                "survivable_revisions": ["revision A"],
                "residual_risks": ["risk A"],
            },
        ),
    )
    assert healthy.misrouting_detected is False
    assert healthy.still_productive is True
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.ADVERSARIAL),
        make_output(
            Stage.ADVERSARIAL,
            {
                "top_destabilizers": ["objection A"],
                "survivable_revisions": "",
                "residual_risks": ["objection A"],
            },
        ),
    )
    assert failing.misrouting_detected is True
    _assert_recommendation_contract(failing, misrouted=True)


def test_operator_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.OPERATOR),
        make_output(
            Stage.OPERATOR,
            {
                "decision": "Ship staged rollout to paid tier first.",
                "rationale": "Limits blast radius while validating conversion impact.",
                "tradeoff_accepted": "Lower short-term reach for operational safety.",
                "next_actions": ["Deploy to staging", "Monitor conversion for 7 days"],
                "fallback_trigger": "Rollback if trial-to-paid conversion drops below baseline.",
                "review_point": "Review after first 7-day measurement window.",
            },
        ),
    )
    assert healthy.misrouting_detected is False
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.OPERATOR),
        make_output(
            Stage.OPERATOR,
            {
                "decision": "Ship immediately.",
                "rationale": "",
                "tradeoff_accepted": "",
                "fallback_trigger": "",
            },
        ),
    )
    assert failing.misrouting_detected is True
    _assert_recommendation_contract(failing, misrouted=True)


def test_builder_healthy_and_failing_pairs():
    detector = MisroutingDetector(RegimeComposer())

    healthy = detector.detect(
        make_state(Stage.BUILDER, recurrence_potential=3.0),
        make_output(
            Stage.BUILDER,
            {
                "modules": ["intake", "planner", "executor"],
                "interfaces": ["intake->planner", "planner->executor"],
            },
        ),
    )
    assert healthy.misrouting_detected is False
    _assert_recommendation_contract(healthy, misrouted=False)

    failing = detector.detect(
        make_state(Stage.BUILDER, recurrence_potential=0.0),
        make_output(
            Stage.BUILDER,
            {
                "modules": ["intake", "planner", "executor"],
                "interfaces": ["intake->planner", "planner->executor"],
            },
        ),
    )
    assert failing.misrouting_detected is True
    _assert_recommendation_contract(failing, misrouted=True)


def test_assumption_collapse_triggers_exploration_fallback():
    detector = MisroutingDetector(RegimeComposer())
    result = detector.detect(
        make_state(Stage.OPERATOR, assumptions=["Demand remains stable"]),
        make_output(
            Stage.OPERATOR,
            {
                "decision": "Expand to enterprise tier now.",
                "rationale": "",
                "tradeoff_accepted": "",
                "fallback_trigger": "",
                "hidden_assumptions": ["Demand remains stable"],
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime is not None
    assert result.recommended_next_regime.stage == Stage.EXPLORATION
