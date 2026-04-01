import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.orchestration.switch_orchestrator import SwitchOrchestrationResult
from router.runtime import CognitiveRuntime


class FakeModelClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more mocked model responses available")
        return {"response": self._responses.pop(0)}

    def list_models(self):
        return {"models": []}


@pytest.fixture
def runtime_factory(monkeypatch):
    runtimes = []

    def _make(responses):
        runtime = CognitiveRuntime(provider="ollama")
        fake_client = FakeModelClient(responses)
        runtime.ollama = fake_client

        def _always_valid(stage, raw_text, **_kwargs):
            parsed = json.loads(raw_text)
            return {
                "is_valid": True,
                "valid_json": True,
                "required_keys_present": True,
                "artifact_fields_present": True,
                "artifact_type_matches": True,
                "contract_controls_valid": True,
                "semantic_valid": True,
                "parsed": parsed,
            }

        monkeypatch.setattr(runtime.validator, "validate", _always_valid)
        runtimes.append((runtime, fake_client))
        return runtime, fake_client

    return _make


def _analyzer_json(primary: Stage, endpoint: Stage, recurrence: int = 3):
    return json.dumps(
        {
            "bottleneck_label": "test task",
            "candidate_regimes": [primary.value],
            "stage_scores": {
                "exploration": 0.1,
                "synthesis": 0.2,
                "epistemic": 0.1,
                "adversarial": 0.1,
                "operator": 0.1,
                "builder": 0.1,
                primary.value: 0.95,
            },
            "structural_signals": ["needs structure"],
            "decision_pressure": 6,
            "evidence_quality": 5,
            "recurrence_potential": recurrence,
            "confidence": 0.92,
            "rationale": "Mocked analyzer result",
            "likely_endpoint_regime": endpoint.value,
            "endpoint_confidence": 0.9,
        }
    )


def _stage_payload(stage: Stage, next_stage: Stage | None):
    artifact = {
        Stage.EXPLORATION: {
            "candidate_frames": ["frame a", "frame b"],
            "selection_criteria": ["criterion"],
            "unresolved_axes": ["unknown axis"],
        },
        Stage.SYNTHESIS: {
            "central_claim": "A cohesive frame",
            "organizing_idea": "Organize around constraints",
            "key_tensions": ["speed vs depth"],
            "supporting_structure": ["support item"],
            "pressure_points": ["invalid if assumptions fail"],
        },
        Stage.EPISTEMIC: {
            "supported_claims": ["supported"],
            "plausible_but_unproven": ["plausible"],
            "contradictions": ["contradiction"],
            "omitted_due_to_insufficient_support": ["omitted"],
            "decision_relevant_conclusions": ["conclusion"],
        },
        Stage.OPERATOR: {
            "decision": "Proceed",
            "rationale": "Best available choice",
            "tradeoff_accepted": "accept risk",
            "next_actions": ["step 1"],
            "fallback_trigger": "quality drops",
            "review_point": "after milestone",
        },
        Stage.BUILDER: {
            "reusable_pattern": "Reusable pattern",
            "modules": ["module1"],
            "interfaces": ["interface1"],
            "required_inputs": ["input1"],
            "produced_outputs": ["output1"],
            "implementation_sequence": ["phase1"],
            "compounding_path": "reuse over time",
        },
    }[stage]
    return json.dumps(
        {
            "regime": stage.value,
            "purpose": f"{stage.value} purpose",
            "artifact_type": "artifact",
            "artifact": artifact,
            "completion_signal": f"{stage.value}_complete",
            "failure_signal": "",
            "recommended_next_regime": next_stage.value if next_stage else stage.value,
        }
    )


def test_every_task_reaches_analyzer(runtime_factory, monkeypatch):
    tasks = [
        "Implement exactly this pattern in Python.",
        "I have a vague sense this is wrong, can you help?",
        "Analyze this architecture and identify its core tensions.",
        "Choose whether we should ship now or delay.",
        "Build a reusable pipeline template for similar projects.",
    ]
    runtime, _client = runtime_factory([_analyzer_json(Stage.EXPLORATION, Stage.OPERATOR) for _ in tasks])

    analyzed = []
    original_analyze = runtime.task_analyzer.analyze

    def spy_analyze(task, **kwargs):
        analyzed.append(task)
        return original_analyze(task, **kwargs)

    classified = []
    original_classify = runtime.task_classifier.classify

    def spy_classify(task):
        classified.append(task)
        return original_classify(task)

    monkeypatch.setattr(runtime.task_analyzer, "analyze", spy_analyze)
    monkeypatch.setattr(runtime.task_classifier, "classify", spy_classify)

    for task in tasks:
        runtime.plan(task)

    assert analyzed == tasks
    assert set(tasks).issubset(set(classified))


def test_handoff_continuity_across_switch(runtime_factory, monkeypatch):
    runtime, client = runtime_factory(
        [
            _analyzer_json(Stage.EXPLORATION, Stage.SYNTHESIS),
            _stage_payload(Stage.EXPLORATION, Stage.SYNTHESIS),
            _stage_payload(Stage.SYNTHESIS, Stage.SYNTHESIS),
        ]
    )

    def orchestrate_to_synthesis(state, output, detection, **_kwargs):
        next_regime = runtime.composer.compose(Stage.SYNTHESIS)
        state.recommended_next_regime = next_regime
        state.switch_trigger = "test_switch"
        return SwitchOrchestrationResult(next_regime, True, "test exploration to synthesis", state)

    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", orchestrate_to_synthesis)

    runtime.execute("Explore this space before converging", model="fake", bounded_orchestration=True, max_switches=2)

    synthesis_prompt = client.calls[-1]["prompt"]
    assert "## Prior Stage Context" in synthesis_prompt
    assert "Dominant frame:" in synthesis_prompt
    assert "What is known:" in synthesis_prompt
    assert "What remains uncertain:" in synthesis_prompt
    assert "Prior artifact summary:" in synthesis_prompt


def test_handoff_not_present_in_first_stage(runtime_factory):
    runtime, client = runtime_factory(
        [
            _analyzer_json(Stage.EXPLORATION, Stage.EXPLORATION),
            _stage_payload(Stage.EXPLORATION, Stage.EXPLORATION),
        ]
    )

    runtime.execute("Single stage task", model="fake", bounded_orchestration=False)

    first_stage_prompt = client.calls[-1]["prompt"]
    assert "## Prior Stage Context" not in first_stage_prompt


def test_stop_at_operator_integration(runtime_factory, monkeypatch):
    runtime, _client = runtime_factory(
        [
            _analyzer_json(Stage.SYNTHESIS, Stage.OPERATOR),
            _stage_payload(Stage.SYNTHESIS, Stage.OPERATOR),
            _stage_payload(Stage.OPERATOR, Stage.OPERATOR),
        ]
    )

    def orchestrate_to_operator(state, output, detection, **_kwargs):
        next_regime = runtime.composer.compose(Stage.OPERATOR)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(next_regime, True, "force operator", state)

    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", orchestrate_to_operator)

    runtime.execute("Decision task ending at operator", model="fake", bounded_orchestration=True, max_switches=3)

    assert runtime.router_state is not None
    assert runtime.router_state.executed_regime_stages == [Stage.SYNTHESIS, Stage.OPERATOR]
    assert "artifact_complete_at_or_past_endpoint:operator" in (runtime.router_state.orchestration_stop_reason or "")


def test_builder_only_when_justified_integration(runtime_factory, monkeypatch):
    runtime, _client = runtime_factory(
        [
            _analyzer_json(Stage.SYNTHESIS, Stage.BUILDER, recurrence=2),
            _stage_payload(Stage.SYNTHESIS, Stage.OPERATOR),
            _stage_payload(Stage.OPERATOR, Stage.BUILDER),
        ]
    )

    def orchestrate(state, output, detection, **_kwargs):
        if state.current_regime.stage == Stage.SYNTHESIS:
            next_regime = runtime.composer.compose(Stage.OPERATOR)
        else:
            next_regime = runtime.composer.compose(Stage.BUILDER)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(next_regime, True, "test switch", state)

    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", orchestrate)

    runtime.execute("One-off decision with low recurrence", model="fake", bounded_orchestration=True, max_switches=3)

    assert runtime.router_state is not None
    assert runtime.router_state.executed_regime_stages == [Stage.SYNTHESIS, Stage.OPERATOR]
    assert "Builder blocked" in (runtime.router_state.orchestration_stop_reason or "")


def test_builder_entered_when_justified(runtime_factory, monkeypatch):
    runtime, _client = runtime_factory(
        [
            _analyzer_json(Stage.SYNTHESIS, Stage.BUILDER, recurrence=9),
            _stage_payload(Stage.SYNTHESIS, Stage.OPERATOR),
            _stage_payload(Stage.OPERATOR, Stage.BUILDER),
            _stage_payload(Stage.BUILDER, Stage.BUILDER),
        ]
    )

    original_update = runtime._update_router_state_from_execution

    def update_with_high_recurrence(state, result, *, reason_entered):
        original_update(state, result, reason_entered=reason_entered)
        if state is not None:
            state.recurrence_potential = 9.0

    def orchestrate(state, output, detection, **_kwargs):
        if state.current_regime.stage == Stage.SYNTHESIS:
            next_regime = runtime.composer.compose(Stage.OPERATOR)
        elif state.current_regime.stage == Stage.OPERATOR:
            next_regime = runtime.composer.compose(Stage.BUILDER)
        else:
            return SwitchOrchestrationResult(None, False, "done", state)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(next_regime, True, "test switch", state)

    monkeypatch.setattr(runtime, "_update_router_state_from_execution", update_with_high_recurrence)
    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", orchestrate)

    runtime.execute("Recurring workflow that merits builder", model="fake", bounded_orchestration=True, max_switches=4)

    assert runtime.router_state is not None
    assert runtime.router_state.executed_regime_stages == [Stage.SYNTHESIS, Stage.OPERATOR, Stage.BUILDER]


def test_no_independent_stage_reruns(runtime_factory, monkeypatch):
    runtime, client = runtime_factory(
        [
            _analyzer_json(Stage.EXPLORATION, Stage.SYNTHESIS),
            _stage_payload(Stage.EXPLORATION, Stage.SYNTHESIS),
            _stage_payload(Stage.SYNTHESIS, Stage.SYNTHESIS),
        ]
    )

    def orchestrate_to_synthesis(state, output, detection, **_kwargs):
        next_regime = runtime.composer.compose(Stage.SYNTHESIS)
        state.recommended_next_regime = next_regime
        return SwitchOrchestrationResult(next_regime, True, "forced for continuity", state)

    monkeypatch.setattr(runtime.switch_orchestrator, "orchestrate", orchestrate_to_synthesis)

    original_task = "Explore and then synthesize this framing"
    runtime.execute(original_task, model="fake", bounded_orchestration=True, max_switches=2)

    second_prompt = client.calls[-1]["prompt"]
    assert original_task in second_prompt
    assert "## Prior Stage Context" in second_prompt
    assert "Build on this context. Do not re-derive what is already established." in second_prompt
