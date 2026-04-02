import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeExecutionResult, RoutingDecision, Stage
from router.prompts import PromptBuilder
from router.routing import RegimeComposer
from router.runtime import CognitiveRuntime
from router.runtime.state_updater import compute_forward_handoff, update_router_state_from_execution
from router.state import Handoff


class _NoopAnalyzer:
    def __init__(self, decision: RoutingDecision):
        self._decision = decision

    def propose_route(self, task, routing_features, task_signals, risk_profile):
        return self._decision


def _decision(primary: Stage, runner_up: Stage) -> RoutingDecision:
    return RoutingDecision(
        bottleneck="task",
        primary_regime=primary,
        runner_up_regime=runner_up,
        why_primary_wins_now="test",
        switch_trigger="test",
    )


def test_initial_proposal_stage_matches_step1_composed_and_executed_stage(monkeypatch):
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))

    observed = {"planned": None, "executed": []}
    original_plan = runtime.plan

    def wrapped_plan(task, **kwargs):
        decision, regime, handoff = original_plan(task, **kwargs)
        observed["planned"] = decision.primary_regime
        assert regime.stage == decision.primary_regime
        return decision, regime, handoff

    def fake_execute_once(self, **kwargs):
        regime = kwargs["regime"]
        observed["executed"].append(regime.stage)
        return RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name=f"{regime.stage.value}-core",
            stage=regime.stage,
            system_prompt="",
            user_prompt="",
            raw_response="{}",
            artifact_text="{}",
            validation={"is_valid": True, "parsed": {"artifact": {}}},
        )

    monkeypatch.setattr(runtime, "plan", wrapped_plan)
    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)

    runtime.execute(task="check continuity", model="fake", bounded_orchestration=False)

    assert observed["planned"] == Stage.EPISTEMIC
    assert observed["executed"] == [Stage.EPISTEMIC]


def test_switch_target_stage_matches_next_step_execution_stage(monkeypatch):
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))

    observed = {"executed": [], "switch_targets": []}
    original_orchestrate = runtime.switch_orchestrator.orchestrate

    def wrapped_orchestrate(state, output, detection, **kwargs):
        result = original_orchestrate(state, output, detection, **kwargs)
        if result.switch_recommended_now and result.next_regime is not None:
            observed["switch_targets"].append(result.next_regime.stage)
        return result

    def fake_execute_once(self, **kwargs):
        regime = kwargs["regime"]
        observed["executed"].append(regime.stage)

        if regime.stage == Stage.EPISTEMIC:
            payload = {
                "completion_signal": "evidence_state_decision_ready",
                "failure_signal": "",
                "recommended_next_regime": "operator",
                "artifact": {
                    "supported_claims": ["c1"],
                    "plausible_but_unproven": ["c2"],
                    "contradictions": ["u1"],
                },
            }
        else:
            payload = {"completion_signal": "", "failure_signal": "", "recommended_next_regime": "", "artifact": {}}

        return RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name=f"{regime.stage.value}-core",
            stage=regime.stage,
            system_prompt="",
            user_prompt="",
            raw_response=json.dumps(payload),
            artifact_text=json.dumps(payload),
            validation={"is_valid": True, "parsed": payload},
        )

    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", wrapped_orchestrate)
    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)

    runtime.execute(task="check switch continuity", model="fake", bounded_orchestration=True, max_switches=1)

    assert observed["switch_targets"] == [Stage.OPERATOR]
    assert observed["executed"] == [Stage.EPISTEMIC, Stage.OPERATOR]


def test_validator_uses_active_step_stage_each_execution():
    from router.execution.executor import RegimeExecutor

    class FakeModelClient:
        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            return {"response": "{}"}

    class SpyValidator:
        def __init__(self):
            self.stages = []

        def validate(self, stage, raw_text, **kwargs):
            self.stages.append(stage)
            return {"is_valid": True, "parsed": {"artifact": {}}}

    class FakePromptBuilder:
        REPAIR_MODE_SEMANTIC = "semantic"

        def build_system_prompt(self, regime, **kwargs):
            return "system"

        def build_user_prompt(self, task, regime, **kwargs):
            return "user"

        def build_repair_prompt(self, *args, **kwargs):
            return "repair"

    validator = SpyValidator()
    executor = RegimeExecutor(model_client=FakeModelClient(), prompt_builder=FakePromptBuilder(), validator=validator)

    for stage in (Stage.EPISTEMIC, Stage.OPERATOR):
        regime = CognitiveRuntime().composer.compose(stage)
        executor.execute_once(task="task", model="fake", regime=regime, task_signals=[], risk_profile=set())

    assert validator.stages == [Stage.EPISTEMIC, Stage.OPERATOR]


def test_first_stage_no_handoff_in_prompt():
    from router.execution.executor import RegimeExecutor

    class FakeModelClient:
        def __init__(self):
            self.last_prompt = ""

        def generate(self, **kwargs):
            self.last_prompt = kwargs["prompt"]
            return {"response": "{}"}

    class FakeValidator:
        def validate(self, *args, **kwargs):
            return {"is_valid": True, "parsed": {"artifact": {}}}

    model_client = FakeModelClient()
    regime = RegimeComposer().compose(Stage.EPISTEMIC)
    executor = RegimeExecutor(model_client=model_client, prompt_builder=PromptBuilder(), validator=FakeValidator())
    executor.execute_once(task="task", model="fake", regime=regime, task_signals=[], risk_profile=set(), prior_handoff=None)

    assert "## Prior Stage Context" not in model_client.last_prompt


def test_second_stage_has_handoff_in_prompt(monkeypatch):
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))
    seen_handoffs = []

    def fake_execute_once(self, **kwargs):
        regime = kwargs["regime"]
        seen_handoffs.append(kwargs.get("prior_handoff"))

        if regime.stage == Stage.EPISTEMIC:
            payload = {
                "completion_signal": "evidence_state_decision_ready",
                "failure_signal": "",
                "recommended_next_regime": "operator",
                "artifact": {
                    "supported_claims": ["Known claim"],
                    "plausible_but_unproven": ["Uncertain claim"],
                    "contradictions": ["Contradiction A"],
                    "omitted_due_to_insufficient_support": ["Assumption A"],
                    "decision_relevant_conclusions": ["Risk A"],
                },
            }
        else:
            payload = {
                "completion_signal": "",
                "failure_signal": "",
                "recommended_next_regime": "",
                "artifact": {"decision": "Do it now"},
            }

        return RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name=f"{regime.stage.value}-core",
            stage=regime.stage,
            system_prompt="",
            user_prompt="",
            raw_response=json.dumps(payload),
            artifact_text=json.dumps(payload),
            validation={"is_valid": True, "parsed": payload},
        )

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    runtime.execute(task="check switch continuity", model="fake", bounded_orchestration=True, max_switches=1)

    assert len(seen_handoffs) == 2
    assert seen_handoffs[0] is None
    second_handoff = seen_handoffs[1]
    assert isinstance(second_handoff, Handoff)

    operator_regime = runtime.composer.compose(Stage.OPERATOR)
    prompt = PromptBuilder.build_user_prompt(
        "task",
        operator_regime,
        task_signals=[],
        risk_profile=set(),
        prior_handoff=second_handoff,
    )
    assert "## Prior Stage Context" in prompt
    assert second_handoff.dominant_frame in prompt


def test_handoff_fields_appear_in_prompt():
    composer = RegimeComposer()
    prior_handoff = Handoff(
        current_bottleneck="bottleneck",
        dominant_frame="Frame X",
        what_is_known=["Known 1", "Known 2"],
        what_remains_uncertain=["Unknown 1"],
        active_contradictions=["Contradiction 1"],
        assumptions_in_play=["Assumption 1"],
        main_risk_if_continue="Risk 1",
        recommended_next_regime=Stage.OPERATOR,
        minimum_useful_artifact="Artifact 1",
        recommended_next_regime_full=composer.compose(Stage.OPERATOR),
    )

    prompt = PromptBuilder.build_user_prompt(
        "task",
        composer.compose(Stage.SYNTHESIS),
        prior_handoff=prior_handoff,
    )

    assert "## Prior Stage Context" in prompt
    assert "Frame X" in prompt
    assert "Known 1" in prompt
    assert "Known 2" in prompt
    assert "Unknown 1" in prompt
    assert "Contradiction 1" in prompt
    assert "Assumption 1" in prompt
    assert "Risk 1" in prompt
    assert "Artifact 1" in prompt


def test_forward_handoff_includes_artifact_summary(monkeypatch):
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))

    def fake_execute_once(self, **kwargs):
        regime = kwargs["regime"]
        payload = {
            "completion_signal": "evidence_state_decision_ready",
            "failure_signal": "",
            "recommended_next_regime": "operator",
            "artifact": {
                "supported_claims": ["Known claim"],
                "plausible_but_unproven": ["Unknown claim"],
                "contradictions": ["Contradiction A"],
            },
        }
        return RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name=f"{regime.stage.value}-core",
            stage=regime.stage,
            system_prompt="",
            user_prompt="",
            raw_response=json.dumps(payload),
            artifact_text=json.dumps(payload),
            validation={"is_valid": True, "parsed": payload},
        )

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    decision, regime, _result, handoff = runtime.execute(task="forward handoff summary", model="fake", bounded_orchestration=False)

    assert decision.primary_regime == regime.stage
    assert handoff.prior_artifact_summary.strip()


def test_forward_handoff_accumulates_knowns():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))
    decision, regime, _ = runtime.plan("knowns accumulation task")
    assert decision.primary_regime == regime.stage
    assert runtime.router_state is not None
    runtime.router_state.knowns = ["Known A", "Known B"]

    payload = {
        "completion_signal": "evidence_state_decision_ready",
        "failure_signal": "",
        "recommended_next_regime": "operator",
        "artifact": {
            "supported_claims": ["Claim C"],
            "plausible_but_unproven": [],
            "contradictions": [],
        },
    }
    result = RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name=f"{regime.stage.value}-core",
        stage=regime.stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(payload),
        artifact_text=json.dumps(payload),
        validation={"is_valid": True, "parsed": payload},
    )
    forward_handoff = compute_forward_handoff(result, runtime.router_state, regime)
    assert len(forward_handoff.what_is_known) == 1


def test_forward_handoff_is_deterministic():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))
    decision, regime, _ = runtime.plan("deterministic handoff task")
    assert decision.primary_regime == regime.stage
    assert runtime.router_state is not None

    payload = {
        "completion_signal": "evidence_state_decision_ready",
        "failure_signal": "",
        "recommended_next_regime": "operator",
        "artifact": {
            "supported_claims": ["Claim 1"],
            "plausible_but_unproven": ["Unknown 1"],
            "contradictions": ["Contra 1"],
        },
    }
    result = RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name=f"{regime.stage.value}-core",
        stage=regime.stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(payload),
        artifact_text=json.dumps(payload),
        validation={"is_valid": True, "parsed": payload},
    )
    assert compute_forward_handoff(result, runtime.router_state, regime) == compute_forward_handoff(
        result,
        runtime.router_state,
        regime,
    )


def test_artifact_summary_in_second_stage_prompt(monkeypatch):
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))
    captured = []

    def fake_execute_once(self, **kwargs):
        regime = kwargs["regime"]
        if kwargs.get("prior_handoff") is not None:
            captured.append(kwargs["prior_handoff"])
        payload = {
            "completion_signal": "evidence_state_decision_ready",
            "failure_signal": "",
            "recommended_next_regime": "operator",
            "artifact": {
                "supported_claims": ["Known claim"],
                "plausible_but_unproven": ["Unknown claim"],
                "contradictions": ["Contradiction A"],
                "decision_relevant_conclusions": ["Action now"],
            },
        }
        if regime.stage == Stage.OPERATOR:
            payload["artifact"] = {"decision": "Ship now"}
        return RegimeExecutionResult(
            task="task",
            model="fake",
            regime_name=f"{regime.stage.value}-core",
            stage=regime.stage,
            system_prompt="",
            user_prompt="",
            raw_response=json.dumps(payload),
            artifact_text=json.dumps(payload),
            validation={"is_valid": True, "parsed": payload},
        )

    monkeypatch.setattr(CognitiveRuntime, "_execute_regime_once", fake_execute_once)
    runtime.execute(task="summary into second prompt", model="fake", bounded_orchestration=True, max_switches=1)
    assert captured
    prompt = PromptBuilder.build_user_prompt(
        "task",
        runtime.composer.compose(Stage.OPERATOR),
        prior_handoff=captured[0],
    )
    assert "Prior artifact summary:" in prompt
    assert captured[0].prior_artifact_summary in prompt


def _execution_result_for_handoff(stage: Stage, artifact: dict, failure_signal: str = "") -> RegimeExecutionResult:
    payload = {
        "completion_signal": "done",
        "failure_signal": failure_signal,
        "recommended_next_regime": "operator",
        "artifact": artifact,
    }
    return RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name=f"{stage.value}-core",
        stage=stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(payload),
        artifact_text=json.dumps(payload),
        validation={"is_valid": True, "parsed": payload},
    )


def _execution_result_for_state_update(stage: Stage, artifact: dict) -> RegimeExecutionResult:
    payload = {
        "completion_signal": "done",
        "failure_signal": "",
        "recommended_next_regime": "operator",
        "artifact": artifact,
    }
    return RegimeExecutionResult(
        task="task",
        model="fake",
        regime_name=f"{stage.value}-core",
        stage=stage,
        system_prompt="",
        user_prompt="",
        raw_response=json.dumps(payload),
        artifact_text=json.dumps(payload),
        validation={
            "is_valid": True,
            "valid_json": True,
            "required_keys_present": True,
            "artifact_fields_present": True,
            "artifact_type_matches": True,
            "contract_controls_valid": True,
            "parsed": payload,
        },
    )


def test_state_dominant_frame_updates_from_synthesis_central_claim():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.SYNTHESIS, Stage.OPERATOR))
    _planned_decision, regime, _ = runtime.plan("synthesis frame update")
    assert runtime.router_state is not None
    result = _execution_result_for_state_update(
        regime.stage,
        {"central_claim": "A tighter synthesis truth."},
    )

    update_router_state_from_execution(
        runtime.router_state,
        result,
        reason_entered="test",
        composer=runtime.composer,
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert runtime.router_state.dominant_frame == "A tighter synthesis truth."
    assert handoff.dominant_frame == runtime.router_state.dominant_frame


def test_state_dominant_frame_updates_from_operator_decision():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _planned_decision, regime, _ = runtime.plan("operator frame update")
    assert runtime.router_state is not None
    result = _execution_result_for_state_update(
        regime.stage,
        {"decision": "Commit to option A now."},
    )

    update_router_state_from_execution(
        runtime.router_state,
        result,
        reason_entered="test",
        composer=runtime.composer,
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert runtime.router_state.dominant_frame == "Commit to option A now."
    assert handoff.dominant_frame == runtime.router_state.dominant_frame


def test_state_dominant_frame_updates_from_builder_reusable_pattern():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.BUILDER, Stage.OPERATOR))
    _planned_decision, regime, _ = runtime.plan("builder frame update")
    assert runtime.router_state is not None
    result = _execution_result_for_state_update(
        regime.stage,
        {"reusable_pattern": "Codify rollout checklist as a reusable template."},
    )

    update_router_state_from_execution(
        runtime.router_state,
        result,
        reason_entered="test",
        composer=runtime.composer,
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert runtime.router_state.dominant_frame == "Codify rollout checklist as a reusable template."
    assert handoff.dominant_frame == runtime.router_state.dominant_frame


def test_handoff_bottleneck_not_raw_task():
    runtime = CognitiveRuntime()
    decision = RoutingDecision(
        bottleneck="prioritization under constrained weekly capacity",
        primary_regime=Stage.OPERATOR,
        runner_up_regime=Stage.EPISTEMIC,
        why_primary_wins_now="test",
        switch_trigger="test",
    )
    runtime.task_analyzer = _NoopAnalyzer(decision)
    planned_decision, regime, _ = runtime.plan("Long raw task text that should never be copied as bottleneck.")
    assert runtime.router_state is not None
    result = _execution_result_for_handoff(regime.stage, {"decision": "Prioritize proposal first", "rationale": "Friday deadline"})
    handoff = compute_forward_handoff(result, runtime.router_state, regime)
    assert handoff.current_bottleneck != "Long raw task text that should never be copied as bottleneck."
    assert handoff.current_bottleneck == planned_decision.bottleneck


def test_handoff_no_system_boilerplate():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("boilerplate check task")
    assert runtime.router_state is not None
    result = _execution_result_for_handoff(
        regime.stage,
        {"decision": "Do A", "rationale": "Deadline", "tradeoff_accepted": "Delay B"},
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)
    payload = " ".join(
        [
            handoff.current_bottleneck,
            handoff.dominant_frame,
            handoff.prior_artifact_summary,
            *handoff.what_is_known,
            *handoff.what_remains_uncertain,
            *handoff.active_contradictions,
            *handoff.assumptions_in_play,
        ]
    )
    assert "dominant failure mode" not in payload
    assert "Soft LLM behavior" not in payload
    assert "bottleneck has been classified" not in payload
    assert "This plan assumes" not in payload


def test_handoff_summary_under_500_chars():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("summary size task")
    assert runtime.router_state is not None
    long_text = "x" * 1200
    result = _execution_result_for_handoff(
        regime.stage,
        {"decision": long_text, "rationale": long_text, "tradeoff_accepted": long_text},
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)
    assert len(handoff.prior_artifact_summary) <= 500


def test_handoff_knowns_max_five():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("knowns cap task")
    assert runtime.router_state is not None
    result = _execution_result_for_handoff(
        regime.stage,
        {
            "decision": "Do A",
            "rationale": "Reason A",
            "tradeoff_accepted": "Delay B",
            "next_actions": ["Step 1", "Step 2"],
            "fallback_trigger": "If blocked",
            "failure_signal": "extra field ignored",
        },
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)
    assert len(handoff.what_is_known) <= 5


def test_handoff_knowns_not_field_dumps():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("known formatting task")
    assert runtime.router_state is not None
    result = _execution_result_for_handoff(
        regime.stage,
        {
            "decision": "Prioritize Friday proposal",
            "rationale": "Hard external deadline",
            "tradeoff_accepted": "Delay internal tooling",
            "next_actions": ["Draft proposal"],
            "fallback_trigger": "No response by Thursday",
        },
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)
    blocked_prefixes = ("decision:", "rationale:", "tradeoff_accepted:", "next_actions:", "fallback_trigger:")
    assert not any(item.lower().startswith(blocked_prefixes) for item in handoff.what_is_known)


def test_operator_tradeoff_not_reclassified_as_contradiction():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("tradeoff contradiction hygiene")
    assert runtime.router_state is not None
    runtime.router_state.contradictions = ["Carry-forward contradiction"]

    result = _execution_result_for_handoff(
        regime.stage,
        {"decision": "Do A", "tradeoff_accepted": "Delay B"},
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.active_contradictions == ["Carry-forward contradiction"]
    assert any("accepted tradeoff" in finding.lower() for finding in handoff.what_is_known)


def test_operator_rationale_and_fallback_do_not_become_assumptions():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("assumption hygiene from prose")
    assert runtime.router_state is not None
    runtime.router_state.assumptions = ["Carry-forward assumption"]

    result = _execution_result_for_handoff(
        regime.stage,
        {
            "decision": "Do A",
            "rationale": "Deadline pressure",
            "fallback_trigger": "If blocked",
            "tradeoff_accepted": "Delay B",
        },
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.assumptions_in_play == ["Carry-forward assumption"]


def test_explicit_contradictions_override_state_fallback():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.EPISTEMIC, Stage.OPERATOR))
    _decision_used, regime, _ = runtime.plan("explicit contradiction passthrough")
    assert runtime.router_state is not None
    runtime.router_state.contradictions = ["Carry-forward contradiction"]

    result = _execution_result_for_handoff(
        regime.stage,
        {
            "supported_claims": ["Claim"],
            "plausible_but_unproven": ["Unknown"],
            "contradictions": ["Explicit contradiction"],
            "decision_relevant_conclusions": ["Conclusion"],
        },
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.active_contradictions == ["Explicit contradiction."]


def test_contradictions_fallback_when_no_explicit_content():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("contradiction fallback")
    assert runtime.router_state is not None
    runtime.router_state.contradictions = ["Carry-forward contradiction"]

    result = _execution_result_for_handoff(regime.stage, {"decision": "Do A"})
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.active_contradictions == ["Carry-forward contradiction"]


def test_assumptions_fallback_when_no_explicit_assumption_content():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.OPERATOR, Stage.EPISTEMIC))
    _decision_used, regime, _ = runtime.plan("assumption fallback")
    assert runtime.router_state is not None
    runtime.router_state.assumptions = ["Carry-forward assumption"]

    result = _execution_result_for_handoff(
        regime.stage,
        {"decision": "Do A", "rationale": "Reason", "fallback_trigger": "If blocked"},
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.assumptions_in_play == ["Carry-forward assumption"]


def test_explicit_hidden_assumptions_override_state_fallback():
    runtime = CognitiveRuntime()
    runtime.task_analyzer = _NoopAnalyzer(_decision(Stage.ADVERSARIAL, Stage.OPERATOR))
    _decision_used, regime, _ = runtime.plan("explicit hidden assumptions")
    assert runtime.router_state is not None
    runtime.router_state.assumptions = ["Carry-forward assumption"]

    result = _execution_result_for_handoff(
        regime.stage,
        {
            "top_destabilizers": ["D1"],
            "hidden_assumptions": ["Explicit hidden assumption"],
            "break_conditions": ["B1"],
            "survivable_revisions": ["R1"],
            "residual_risks": ["Risk"],
        },
    )
    handoff = compute_forward_handoff(result, runtime.router_state, regime)

    assert handoff.assumptions_in_play == ["Explicit hidden assumption."]
