import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import MisroutingDetector, RegimeOutputContract, SwitchOrchestrator
from router.models import RegimeConfidenceResult, RegimeExecutionResult, Stage
from router.routing import RegimeComposer
from router.runtime import CognitiveRuntime
from router.state import RouterState


def _state_for(stage: Stage, *, recurrence_potential: float = 0.0, assumptions=None, contradictions=None) -> RouterState:
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
        substantive_assumptions=list(assumptions or []),
        risks=["baseline risk"],
        stage_goal="goal",
        planned_switch_condition="planned_switch_from_router",
        observed_switch_cause=None,
        switch_trigger="trigger",
        recommended_next_regime=runner_up,
        decision_pressure=1.0,
        evidence_quality=1.0,
        recurrence_potential=recurrence_potential,
    )


def _output(
    stage: Stage,
    artifact: dict,
    *,
    completion_signal: str,
    failure_signal: str,
    validation_overrides: dict | None = None,
) -> RegimeOutputContract:
    parsed = {
        "regime": "test",
        "purpose": "test purpose",
        "artifact_type": "test_artifact",
        "completion_signal": completion_signal,
        "failure_signal": failure_signal,
        "recommended_next_regime": "operator",
        "artifact": artifact,
    }
    validation = {"parsed": parsed}
    if validation_overrides:
        validation.update(validation_overrides)
    return RegimeOutputContract(stage=stage, raw_response=json.dumps(parsed), validation=validation)


def _detect(state: RouterState, output: RegimeOutputContract):
    return MisroutingDetector(RegimeComposer()).detect(state, output)


def test_exploration_completion_moves_to_synthesis():
    state = _state_for(Stage.EXPLORATION)
    output = _output(
        Stage.EXPLORATION,
        {"candidate_frames": ["a b", "c d", "e f"], "selection_criteria": "best criterion", "unresolved_axes": []},
        completion_signal="exploration_ready_for_selection",
        failure_signal="branch_sprawl_without_differentiation",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.switch_recommended_now is True
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.SYNTHESIS


def test_synthesis_failure_moves_to_epistemic_or_adversarial():
    orchestrator = SwitchOrchestrator(RegimeComposer())
    adversarial_state = _state_for(Stage.SYNTHESIS)
    adversarial_output = _output(
        Stage.SYNTHESIS,
        {
            "central_claim": "Frame claim",
            "organizing_idea": "Frame organizing logic",
            "supporting_structure": "",
            "pressure_points": ["Strong break point"],
        },
        completion_signal="coherent_frame_stable",
        failure_signal="frame_collapses_under_pressure_points",
    )
    adversarial_result = orchestrator.orchestrate(
        adversarial_state, adversarial_output, _detect(adversarial_state, adversarial_output), switches_used=0, max_switches=2
    )
    assert adversarial_result.next_regime is not None
    assert adversarial_result.next_regime.stage == Stage.ADVERSARIAL

    epistemic_state = _state_for(Stage.SYNTHESIS, contradictions=["c1"])
    epistemic_output = _output(
        Stage.SYNTHESIS,
        {
            "central_claim": "Frame claim",
            "organizing_idea": "Frame organizing logic",
            "supporting_structure": "",
            "pressure_points": "",
        },
        completion_signal="coherent_frame_stable",
        failure_signal="frame_collapses_under_pressure_points",
    )
    epistemic_result = orchestrator.orchestrate(
        epistemic_state, epistemic_output, _detect(epistemic_state, epistemic_output), switches_used=0, max_switches=2
    )
    assert epistemic_result.next_regime is not None
    assert epistemic_result.next_regime.stage == Stage.EPISTEMIC


def test_epistemic_completion_moves_to_operator():
    state = _state_for(Stage.EPISTEMIC)
    output = _output(
        Stage.EPISTEMIC,
        {
            "supported_claims": ["Supported claim with evidence"],
            "plausible_but_unproven": ["Unproven but plausible claim"],
            "omitted_due_to_insufficient_support": ["Omitted claim"],
            "contradictions": ["conflict found"],
            "decision_relevant_conclusions": ["what is true enough now"],
        },
        completion_signal="support_separation_completed",
        failure_signal="evidence_quality_insufficient",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.OPERATOR


def test_operator_completion_with_recurrence_moves_to_builder():
    state = _state_for(Stage.OPERATOR, recurrence_potential=2.0)
    output = _output(
        Stage.OPERATOR,
        {
            "decision": "Choose path A now",
            "rationale": "Rationale ties to evidence and risk.",
            "tradeoff_accepted": "Accept slower speed for better reversibility.",
            "next_actions": ["Action one", "Action two"],
            "fallback_trigger": "If metric drops, revert.",
            "review_point": "Review in two weeks.",
        },
        completion_signal="decision_ready_for_execution",
        failure_signal="forced_closure_without_real_tradeoff",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.BUILDER


def test_operator_failure_moves_to_epistemic():
    state = _state_for(Stage.OPERATOR)
    output = _output(
        Stage.OPERATOR,
        {
            "decision": "",
            "rationale": "",
            "tradeoff_accepted": "",
            "next_actions": [],
            "fallback_trigger": "",
            "review_point": "",
        },
        completion_signal="decision_committed_with_actions",
        failure_signal="decision_not_actionable_under_constraints",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.switch_recommended_now is True
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.EPISTEMIC


def test_operator_semantic_failure_moves_to_epistemic():
    state = _state_for(Stage.OPERATOR)
    output = _output(
        Stage.OPERATOR,
        {
            "decision": "Proceed with guarded offer acceptance.",
            "rationale": "Expected value is positive if one missing diligence check passes.",
            "tradeoff_accepted": "Accept slower commitment in exchange for lower downside.",
            "next_actions": ["Run final diligence check.", "Collect counterparty confirmation."],
            "fallback_trigger": "Reconsider if new information affects understanding of risk.",
            "review_point": "Review after diligence result.",
        },
        completion_signal="decision_committed_with_actions",
        failure_signal="decision_not_actionable_under_constraints",
        validation_overrides={
            "is_valid": False,
            "valid_json": True,
            "required_keys_present": True,
            "artifact_fields_present": True,
            "artifact_type_matches": True,
            "contract_controls_valid": True,
            "semantic_valid": False,
            "semantic_failures": ["fallback_trigger contains generic filler: understanding"],
        },
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.switch_recommended_now is True
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.EPISTEMIC


def test_valid_operator_output_with_no_recurrence_does_not_switch():
    state = _state_for(Stage.OPERATOR, recurrence_potential=0.0)
    output = _output(
        Stage.OPERATOR,
        {
            "decision": "Choose path A now",
            "rationale": "Path A preserves the highest value under current constraints.",
            "tradeoff_accepted": "Accept delayed optionality review for immediate clarity.",
            "next_actions": ["Notify team", "Start execution"],
            "fallback_trigger": "If key metric worsens by 20%.",
            "review_point": "Review in one week.",
        },
        completion_signal="decision_committed_with_actions",
        failure_signal="decision_not_actionable_under_constraints",
        validation_overrides={
            "is_valid": True,
            "semantic_failures": [],
        },
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.switch_recommended_now is False
    assert result.next_regime is None


def test_assumption_collapse_triggers_exploration_fallback():
    state = _state_for(Stage.OPERATOR, assumptions=["a1"], contradictions=["c1"])
    output = _output(
        Stage.OPERATOR,
        {
            "decision": "Choose path A now",
            "rationale": "Rationale missing stability assumptions.",
            "tradeoff_accepted": "Accept slower speed for better reversibility.",
            "next_actions": ["Action one", "Action two"],
            "fallback_trigger": "If metric drops, revert.",
            "review_point": "Review in two weeks.",
        },
        completion_signal="decision_ready_for_execution",
        failure_signal="frame_collapses_due_to_assumption_collapse",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=0, max_switches=2)
    assert result.next_regime is not None
    assert result.next_regime.stage == Stage.EXPLORATION


def test_max_step_bound_is_enforced():
    state = _state_for(Stage.EXPLORATION)
    output = _output(
        Stage.EXPLORATION,
        {"candidate_frames": ["a b", "c d", "e f"], "selection_criteria": "criterion", "unresolved_axes": ["x"]},
        completion_signal="exploration_ready_for_selection",
        failure_signal="branch_sprawl_without_differentiation",
    )
    result = SwitchOrchestrator(RegimeComposer()).orchestrate(state, output, _detect(state, output), switches_used=2, max_switches=2)
    assert result.switch_recommended_now is False
    assert result.next_regime is None


def test_runtime_bounded_mode_updates_prior_regimes_without_infinite_loop(monkeypatch):
    runtime = CognitiveRuntime(provider="ollama")
    scripted = [
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Exploration Core",
            stage=Stage.EXPLORATION,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "exploration_ready_for_selection",
                    "failure_signal": "branch_sprawl_without_differentiation",
                    "recommended_next_regime": "synthesis",
                    "artifact": {"candidate_frames": ["a b", "c d", "e f"], "selection_criteria": "criterion", "unresolved_axes": ["u"]},
                },
            },
        ),
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Synthesis Core",
            stage=Stage.SYNTHESIS,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "coherent_frame_stable",
                    "failure_signal": "frame_collapses_under_pressure_points",
                    "recommended_next_regime": "adversarial",
                    "artifact": {
                        "central_claim": "claim text here",
                        "organizing_idea": "organizing text here",
                        "supporting_structure": "",
                        "pressure_points": ["pressure point"],
                    },
                },
            },
        ),
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Adversarial Core",
            stage=Stage.ADVERSARIAL,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "stress_test_completed",
                    "failure_signal": "attack_loop_without_revisions",
                    "recommended_next_regime": "operator",
                    "artifact": {
                        "top_destabilizers": ["d1"],
                        "hidden_assumptions": ["h1"],
                        "break_conditions": ["b1"],
                        "survivable_revisions": ["revise one"],
                        "residual_risks": ["r1"],
                    },
                },
            },
        ),
    ]
    call_count = {"n": 0}

    def fake_execute_once(self, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return scripted[idx]

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    runtime.execute(
        task="Brainstorm some options.",
        model="fake",
        bounded_orchestration=True,
        max_switches=2,
    )

    assert runtime.router_state is not None
    assert call_count["n"] == 3
    assert len(runtime.router_state.prior_regimes) == 3
    assert runtime.router_state.switches_executed == 2
    assert runtime.router_state.orchestration_enabled is True
    assert runtime.router_state.orchestration_stop_reason == "switch_limit_reached"


def test_runtime_single_step_mode_preserves_old_behavior(monkeypatch):
    runtime = CognitiveRuntime(provider="ollama")
    scripted = [
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Exploration Core",
            stage=Stage.EXPLORATION,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={"is_valid": True, "semantic_failures": [], "parsed": {"artifact": {}}},
        )
    ]
    call_count = {"n": 0}

    def fake_execute_once(self, **kwargs):
        call_count["n"] += 1
        return scripted[0]

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    runtime.execute(task="Simple task", model="fake", bounded_orchestration=False)

    assert call_count["n"] == 1
    assert runtime.router_state is not None
    assert len(runtime.router_state.prior_regimes) == 1
    assert runtime.router_state.switches_attempted == 0
    assert runtime.router_state.switches_executed == 0
    assert runtime.router_state.orchestration_stop_reason == "single_step_mode"


def test_runtime_allows_single_collapse_reentry_then_stops(monkeypatch):
    runtime = CognitiveRuntime(provider="ollama")
    scripted = [
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Exploration Core",
            stage=Stage.EXPLORATION,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "exploration_ready_for_selection",
                    "failure_signal": "",
                    "recommended_next_regime": "synthesis",
                    "artifact": {"candidate_frames": ["a b", "c d", "e f"], "selection_criteria": "criterion"},
                },
            },
        ),
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Synthesis Core",
            stage=Stage.SYNTHESIS,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "",
                    "failure_signal": "assumption collapse",
                    "recommended_next_regime": "exploration",
                    "artifact": {
                        "central_claim": "claim",
                        "organizing_idea": "idea",
                        "supporting_structure": "",
                        "hidden_assumptions": ["This claim is unsupported by evidence."],
                    },
                },
            },
        ),
        RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name="Exploration Core",
            stage=Stage.EXPLORATION,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={
                "is_valid": True,
                "semantic_failures": [],
                "parsed": {
                    "completion_signal": "",
                    "failure_signal": "",
                    "recommended_next_regime": "exploration",
                    "artifact": {"candidate_frames": ["a b"], "selection_criteria": "criterion", "unresolved_axes": []},
                },
            },
        ),
    ]
    call_count = {"n": 0}

    def fake_execute_once(self, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return scripted[idx]

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    runtime.execute(task="Brainstorm options.", model="fake", bounded_orchestration=True, max_switches=3)

    assert runtime.router_state is not None
    assert call_count["n"] == 3
    assert runtime.router_state.switches_executed == 2
    assert runtime.router_state.collapse_reentries == 1
    assert runtime.router_state.orchestration_stop_reason == "switch_not_recommended"
    assert runtime.router_state.switch_history[-1].switch_executed is False
    assert runtime.router_state.switch_history[-1].planned_switch_condition == runtime.router_state.planned_switch_condition
    assert runtime.router_state.switch_history[1].observed_switch_cause == "assumption_or_frame_collapse"
    assert runtime.router_state.switch_history[1].reason == "collapse_recovery"
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == Stage.EXPLORATION
