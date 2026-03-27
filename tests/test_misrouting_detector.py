import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector, RegimeOutputContract
from router.models import RegimeConfidenceResult, Stage
from router.routing import RegimeComposer
from router.state import RouterState


def _state_for(
    stage: Stage,
    *,
    contradictions: list[str] | None = None,
    assumptions: list[str] | None = None,
    recurrence_potential: float = 0.0,
) -> RouterState:
    composer = RegimeComposer()
    regime = composer.compose(stage)
    runner_up = composer.compose(Stage.SYNTHESIS if stage != Stage.SYNTHESIS else Stage.EPISTEMIC)
    return RouterState(
        task_id="task-test",
        task_summary="test",
        current_bottleneck="test bottleneck",
        current_regime=regime,
        runner_up_regime=runner_up,
        regime_confidence=RegimeConfidenceResult.low_default(),
        dominant_frame=None,
        knowns=["known fact"],
        uncertainties=["uncertain fact"],
        contradictions=list(contradictions or []),
        assumptions=list(assumptions or []),
        risks=["baseline risk"],
        stage_goal="goal",
        switch_trigger="trigger",
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
    )


def _output_for(stage: Stage, artifact: dict) -> RegimeOutputContract:
    payload = {
        "regime": "test",
        "stage": stage.value,
        "artifact_type": "test_artifact",
        "artifact": artifact,
    }
    return RegimeOutputContract(
        stage=stage,
        raw_response=json.dumps(payload),
        validation={"parsed": payload},
    )


def test_exploration_positive_detects_branch_sprawl_without_selection_criteria():
    detector = MisroutingDetector()
    state = _state_for(Stage.EXPLORATION)
    result = detector.detect(
        state,
        _output_for(
            Stage.EXPLORATION,
            {
                "candidate_frames": ["f1", "f2", "f3", "f4", "f5"],
                "selection_criteria": "",
                "unresolved_axes": ["a1", "a2", "a3"],
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.SYNTHESIS


def test_exploration_negative_does_not_trigger_with_two_or_three_frames_and_criteria():
    detector = MisroutingDetector()
    state = _state_for(Stage.EXPLORATION)
    result = detector.detect(
        state,
        _output_for(
            Stage.EXPLORATION,
            {
                "candidate_frames": ["f1", "f2", "f3"],
                "selection_criteria": ["criterion A"],
                "unresolved_axes": ["a1"],
            },
        ),
    )

    assert result.misrouting_detected is False
    assert result.recommended_next_regime is None


def test_synthesis_positive_detects_unsupported_unification_and_flattened_contradictions():
    detector = MisroutingDetector()
    state = _state_for(Stage.SYNTHESIS, contradictions=["Signal A conflicts with signal B"])
    result = detector.detect(
        state,
        _output_for(
            Stage.SYNTHESIS,
            {
                "central_claim": "One elegant frame explains everything.",
                "organizing_idea": "The pattern is unified by one mechanism.",
                "supporting_structure": "",
                "pressure_points": "",
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.EPISTEMIC


def test_epistemic_positive_detects_support_map_without_decision_useful_conclusion():
    detector = MisroutingDetector()
    state = _state_for(Stage.EPISTEMIC)
    result = detector.detect(
        state,
        _output_for(
            Stage.EPISTEMIC,
            {
                "supported_claims": ["Claim A is supported"],
                "plausible_but_unproven": ["Claim B may hold"],
                "contradictions": ["Signal C vs D"],
                "omitted_due_to_insufficient_support": ["Claim E"],
                "decision_relevant_conclusions": "",
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.OPERATOR


def test_adversarial_positive_detects_repetitive_objections_and_defaults_to_operator():
    detector = MisroutingDetector()
    state = _state_for(Stage.ADVERSARIAL)
    objections = ["Objection A", "Objection B"]
    result = detector.detect(
        state,
        _output_for(
            Stage.ADVERSARIAL,
            {
                "top_destabilizers": objections,
                "hidden_assumptions": ["Assumption 1"],
                "break_conditions": ["Break if X"],
                "survivable_revisions": "",
                "residual_risks": objections,
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.OPERATOR


def test_operator_positive_detects_forced_closure_and_routes_to_epistemic_when_support_is_missing():
    detector = MisroutingDetector()
    state = _state_for(Stage.OPERATOR, assumptions=["Assume demand stays flat"])
    result = detector.detect(
        state,
        _output_for(
            Stage.OPERATOR,
            {
                "decision": "Choose option A",
                "rationale": "",
                "tradeoff_accepted": "",
                "next_actions": ["Do it"],
                "fallback_trigger": "",
                "review_point": "later",
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.EPISTEMIC


def test_builder_positive_detects_premature_architecture_with_weak_recurrence():
    detector = MisroutingDetector()
    state = _state_for(Stage.BUILDER, recurrence_potential=0.0)
    result = detector.detect(
        state,
        _output_for(
            Stage.BUILDER,
            {
                "reusable_pattern": "",
                "modules": ["Module A", "Module B"],
                "interfaces": ["A->B"],
                "required_inputs": ["input1"],
                "produced_outputs": ["output1"],
                "implementation_sequence": ["step1"],
                "compounding_path": "",
            },
        ),
    )

    assert result.misrouting_detected is True
    assert result.recommended_next_regime == Stage.OPERATOR


def test_detector_does_not_trigger_constantly_across_balanced_cases():
    detector = MisroutingDetector()
    triggered = 0
    for stage, artifact, kwargs in [
        (
            Stage.EXPLORATION,
            {
                "candidate_frames": ["f1", "f2", "f3"],
                "selection_criteria": "Select by decision impact.",
                "unresolved_axes": ["a1"],
            },
            {},
        ),
        (
            Stage.SYNTHESIS,
            {
                "central_claim": "Frame with explicit limits.",
                "organizing_idea": "Mechanism tied to input signals.",
                "supporting_structure": ["support point 1", "support point 2"],
                "pressure_points": ["If X fails, frame fails"],
            },
            {"contradictions": ["c1"]},
        ),
        (
            Stage.OPERATOR,
            {
                "decision": "Option B",
                "rationale": "Reversible and resource-feasible now.",
                "tradeoff_accepted": "Sacrifice short-term throughput for optionality.",
                "next_actions": ["step1"],
                "fallback_trigger": "If metric M drops",
                "review_point": "in one week",
            },
            {"assumptions": ["a1"]},
        ),
    ]:
        state = _state_for(stage, **kwargs)
        result = detector.detect(state, _output_for(stage, artifact))
        if result.misrouting_detected:
            triggered += 1

    assert triggered <= 1


def test_any_regime_can_fallback_to_exploration_on_assumption_collapse():
    detector = MisroutingDetector()
    for stage in [Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR, Stage.BUILDER]:
        state = _state_for(stage, assumptions=["Core assumption"])
        # Ensure each stage has its stage-specific failure signal active.
        stage_artifact = {
            Stage.SYNTHESIS: {
                "central_claim": "unified claim",
                "organizing_idea": "single mechanism",
                "supporting_structure": "",
                "pressure_points": "",
                "hidden_assumptions": ["A1"],
            },
            Stage.EPISTEMIC: {
                "supported_claims": ["supported"],
                "plausible_but_unproven": ["unproven"],
                "contradictions": ["c1"],
                "omitted_due_to_insufficient_support": ["o1"],
                "decision_relevant_conclusions": "",
            },
            Stage.ADVERSARIAL: {
                "top_destabilizers": ["D1"],
                "hidden_assumptions": ["A1"],
                "break_conditions": ["B1"],
                "survivable_revisions": "",
                "residual_risks": ["D1"],
            },
            Stage.OPERATOR: {
                "decision": "Do X",
                "rationale": "",
                "tradeoff_accepted": "",
                "next_actions": ["step"],
                "fallback_trigger": "",
                "review_point": "soon",
                "hidden_assumptions": ["A1"],
            },
            Stage.BUILDER: {
                "reusable_pattern": "",
                "modules": ["M1"],
                "interfaces": ["I1"],
                "required_inputs": ["in"],
                "produced_outputs": ["out"],
                "implementation_sequence": ["s1"],
                "compounding_path": "",
                "hidden_assumptions": ["A1"],
            },
        }[stage]

        result = detector.detect(state, _output_for(stage, stage_artifact))
        assert result.misrouting_detected is True
        assert result.recommended_next_regime == Stage.EXPLORATION


def test_deterministic_transition_defaults_are_spec_aligned():
    detector = MisroutingDetector()

    # Exploration completion-like failure signal defaults to synthesis.
    exploration = detector.detect(
        _state_for(Stage.EXPLORATION),
        _output_for(Stage.EXPLORATION, {"candidate_frames": ["a", "b", "c", "d"], "selection_criteria": "", "unresolved_axes": ["u"]}),
    )
    assert exploration.recommended_next_regime == Stage.SYNTHESIS

    # Synthesis failure can branch to epistemic (default) or adversarial (explicit pressure points).
    synthesis_default = detector.detect(
        _state_for(Stage.SYNTHESIS),
        _output_for(Stage.SYNTHESIS, {"central_claim": "c", "organizing_idea": "o", "supporting_structure": "", "pressure_points": ""}),
    )
    assert synthesis_default.recommended_next_regime == Stage.EPISTEMIC

    synthesis_stress = detector.detect(
        _state_for(Stage.SYNTHESIS),
        _output_for(Stage.SYNTHESIS, {"central_claim": "c", "organizing_idea": "o", "supporting_structure": "", "pressure_points": ["stress this"]}),
    )
    assert synthesis_stress.recommended_next_regime == Stage.ADVERSARIAL

    epistemic = detector.detect(
        _state_for(Stage.EPISTEMIC),
        _output_for(
            Stage.EPISTEMIC,
            {
                "supported_claims": ["S1"],
                "plausible_but_unproven": ["P1"],
                "contradictions": ["C1"],
                "omitted_due_to_insufficient_support": ["O1"],
                "decision_relevant_conclusions": "",
            },
        ),
    )
    assert epistemic.recommended_next_regime == Stage.OPERATOR

    adversarial = detector.detect(
        _state_for(Stage.ADVERSARIAL),
        _output_for(
            Stage.ADVERSARIAL,
            {
                "top_destabilizers": ["D1"],
                "hidden_assumptions": ["A1"],
                "break_conditions": ["B1"],
                "survivable_revisions": "",
                "residual_risks": ["D1"],
            },
        ),
    )
    assert adversarial.recommended_next_regime == Stage.OPERATOR

    operator_builder = detector.detect(
        _state_for(Stage.OPERATOR, assumptions=["A1"], recurrence_potential=2.5),
        _output_for(
            Stage.OPERATOR,
            {
                "decision": "Do X",
                "rationale": "",
                "tradeoff_accepted": "",
                "next_actions": ["step"],
                "fallback_trigger": "",
                "review_point": "soon",
            },
        ),
    )
    assert operator_builder.recommended_next_regime == Stage.BUILDER


def test_orchestration_path_is_bounded_and_preserves_state_fields_without_loops():
    detector = MisroutingDetector()
    max_steps = 6
    visited: list[Stage] = []
    state = _state_for(Stage.EXPLORATION, assumptions=[])

    artifacts = {
        Stage.EXPLORATION: {"candidate_frames": ["f1", "f2", "f3", "f4"], "selection_criteria": "", "unresolved_axes": ["u1"]},
        Stage.SYNTHESIS: {"central_claim": "c", "organizing_idea": "o", "supporting_structure": "", "pressure_points": ""},
        Stage.EPISTEMIC: {
            "supported_claims": ["S1"],
            "plausible_but_unproven": ["P1"],
            "contradictions": ["C1"],
            "omitted_due_to_insufficient_support": ["O1"],
            "decision_relevant_conclusions": "",
        },
        Stage.ADVERSARIAL: {
            "top_destabilizers": ["D1"],
            "hidden_assumptions": ["A1"],
            "break_conditions": ["B1"],
            "survivable_revisions": "",
            "residual_risks": ["D1"],
        },
        Stage.OPERATOR: {
            "decision": "Do X",
            "rationale": "Decision is supported by explicit throughput constraints.",
            "tradeoff_accepted": "Accept lower short-term speed to keep reversibility.",
            "next_actions": ["step"],
            "fallback_trigger": "Re-open if throughput drops below threshold.",
            "review_point": "soon",
        },
        Stage.BUILDER: {
            "reusable_pattern": "",
            "modules": ["M1"],
            "interfaces": ["I1"],
            "required_inputs": ["in"],
            "produced_outputs": ["out"],
            "implementation_sequence": ["s1"],
            "compounding_path": "",
        },
    }

    for _ in range(max_steps):
        stage = state.current_regime.stage
        visited.append(stage)
        result = detector.detect(state, _output_for(stage, artifacts[stage]))
        state.record_regime_step(
            regime=stage,
            reason_entered="test transition",
            completion_signal_seen=not result.misrouting_detected,
            failure_signal_seen=result.misrouting_detected,
            outcome_summary=result.justification,
        )

        if result.recommended_next_regime is None:
            break

        state.current_regime = RegimeComposer().compose(result.recommended_next_regime)

    assert len(visited) <= max_steps
    assert len(state.prior_regimes) == len(visited)
    assert state.knowns == ["known fact"]
    assert state.uncertainties == ["uncertain fact"]
    assert state.risks == ["baseline risk"]
    assert all(step.regime == stage for step, stage in zip(state.prior_regimes, visited))
    # bounded-step guard; if a loop appeared, we would hit max_steps with no break
    assert len(visited) < max_steps
