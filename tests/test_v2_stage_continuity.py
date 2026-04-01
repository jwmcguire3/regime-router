import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import RegimeExecutionResult, RoutingDecision, Stage
from router.prompts import PromptBuilder
from router.routing import RegimeComposer
from router.runtime import CognitiveRuntime
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
