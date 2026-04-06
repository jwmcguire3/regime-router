from __future__ import annotations

import json
from typing import Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.models import Stage, TaskAnalyzerOutput


class StubModelClient:
    def __init__(self, responses: List[Dict[str, object]]) -> None:
        self._responses = list(responses)

    def generate(self, **_: object) -> Dict[str, object]:
        if not self._responses:
            raise AssertionError("No stub responses configured")
        return self._responses.pop(0)

    def list_models(self) -> Dict[str, object]:
        return {"models": []}


class AnalyzerWithStubbedAnalyze(TaskAnalyzer):
    def __init__(self) -> None:
        super().__init__(model_client=StubModelClient([]), model="stub")
        self.stub_result: TaskAnalyzerOutput | None = None

    def analyze(
        self,
        task: str,
    ) -> TaskAnalyzerOutput | None:
        return self.stub_result


def _analysis_output(
    *,
    confidence: float = 0.6,
    top_stage: Stage = Stage.SYNTHESIS,
    decision_pressure: int = 5,
    fragility_pressure: int = 5,
    recurrence_potential: int = 5,
) -> Dict[str, object]:
    scores = {
        Stage.EXPLORATION.value: 0.1,
        Stage.SYNTHESIS.value: 0.2,
        Stage.EPISTEMIC.value: 0.3,
        Stage.ADVERSARIAL.value: 0.4,
        Stage.OPERATOR.value: 0.5,
        Stage.BUILDER.value: 0.6,
    }
    scores[top_stage.value] = 0.9
    return {
        "bottleneck_label": "need immediate call",
        "candidate_regimes": [top_stage.value, Stage.SYNTHESIS.value],
        "stage_scores": scores,
        "structural_signals": ["decision_tradeoff_commitment"],
        "decision_pressure": decision_pressure,
        "fragility_pressure": fragility_pressure,
        "possibility_space_need": 4,
        "synthesis_pressure": 3,
        "evidence_quality": 5,
        "recurrence_potential": recurrence_potential,
        "confidence": confidence,
        "rationale": "Task language explicitly prioritizes this bottleneck.",
        "risk_tags": ["high_stakes"],
        "likely_endpoint_regime": Stage.OPERATOR.value,
        "endpoint_confidence": 0.7,
    }


def test_propose_route_returns_exploration_fallback_when_analyzer_fails() -> None:
    analyzer = AnalyzerWithStubbedAnalyze()
    analyzer.stub_result = None
    analyzer.last_error_summary = "Analyzer malformed JSON response."

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.EXPLORATION
    assert decision.runner_up_regime == Stage.SYNTHESIS
    assert decision.confidence.level == "low"
    assert decision.analyzer_summary == "Analyzer malformed JSON response."


def test_propose_route_ranks_stages_from_analyzer_output() -> None:
    client = StubModelClient([{"response": json.dumps(_analysis_output(top_stage=Stage.EPISTEMIC))}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.EPISTEMIC
    assert decision.runner_up_regime == Stage.BUILDER


def test_operator_primary_not_hard_demoted_when_decision_pressure_is_zero() -> None:
    client = StubModelClient(
        [{"response": json.dumps(_analysis_output(top_stage=Stage.OPERATOR, decision_pressure=0))}]
    )
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.OPERATOR
    assert "operator support weak; soft guardrail only" in decision.policy_warnings
    assert "demoted to exploration" not in (decision.analyzer_summary or "")


def test_builder_primary_not_hard_demoted_when_recurrence_potential_is_zero() -> None:
    client = StubModelClient(
        [{"response": json.dumps(_analysis_output(top_stage=Stage.BUILDER, recurrence_potential=0))}]
    )
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.BUILDER
    assert "builder support weak; advisory only" in decision.policy_warnings


def test_high_confidence_threshold_works() -> None:
    client = StubModelClient([{"response": json.dumps(_analysis_output(confidence=0.85, top_stage=Stage.SYNTHESIS))}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.confidence.level == "high"


def test_analyzer_summary_is_populated() -> None:
    client = StubModelClient([{"response": json.dumps(_analysis_output(confidence=0.62, top_stage=Stage.SYNTHESIS))}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.analyzer_summary is not None
    assert "Analyzer confidence=0.62" in decision.analyzer_summary
    assert "candidates=" in decision.analyzer_summary


def test_analyzer_prompt_is_task_only() -> None:
    client = StubModelClient([{"response": json.dumps(_analysis_output())}])
    captured: Dict[str, object] = {}
    original_generate = client.generate

    def _capture_generate(**kwargs: object) -> Dict[str, object]:
        captured["prompt"] = kwargs["prompt"]
        return original_generate(**kwargs)

    analyzer = TaskAnalyzer(model_client=client, model="stub")
    analyzer.model_client.generate = _capture_generate  # type: ignore[method-assign]
    analyzer.analyze("write a function")

    prompt = str(captured["prompt"])
    system_prompt = analyzer._build_system_prompt()
    assert "Classifier assessment:" not in prompt
    assert "deterministic_features:" not in prompt
    assert "Estimate which regime will produce the minimum useful artifact for this task." in system_prompt


def test_endpoint_defaults_to_operator() -> None:
    payload = _analysis_output(top_stage=Stage.SYNTHESIS)
    payload.pop("likely_endpoint_regime")
    payload.pop("endpoint_confidence")
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.likely_endpoint_regime == Stage.OPERATOR.value
    assert decision.endpoint_confidence == 0.7


def test_builder_endpoint_softened_without_recurrence_when_confidence_not_high() -> None:
    payload = _analysis_output(top_stage=Stage.SYNTHESIS)
    payload["likely_endpoint_regime"] = Stage.BUILDER.value
    payload["endpoint_confidence"] = 0.9
    payload["recurrence_potential"] = 0
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.likely_endpoint_regime == Stage.OPERATOR.value
    assert "builder endpoint softened to operator" in decision.policy_actions


def test_endpoint_in_routing_decision() -> None:
    payload = _analysis_output(top_stage=Stage.EPISTEMIC)
    payload["likely_endpoint_regime"] = Stage.OPERATOR.value
    payload["endpoint_confidence"] = 0.83
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.likely_endpoint_regime == Stage.OPERATOR.value
    assert decision.endpoint_confidence == 0.83


def test_endpoint_cannot_precede_primary() -> None:
    payload = _analysis_output(top_stage=Stage.SYNTHESIS)
    payload["likely_endpoint_regime"] = Stage.EXPLORATION.value
    payload["endpoint_confidence"] = 0.8
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.likely_endpoint_regime == Stage.SYNTHESIS.value


def test_missing_new_fields_default_during_transition() -> None:
    payload = _analysis_output(top_stage=Stage.SYNTHESIS)
    payload.pop("fragility_pressure")
    payload.pop("possibility_space_need")
    payload.pop("synthesis_pressure")
    payload.pop("risk_tags")
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    result = analyzer.analyze("task")

    assert result is not None
    assert result.fragility_pressure == 0
    assert result.possibility_space_need == 0
    assert result.synthesis_pressure == 0
    assert result.risk_tags == []


def test_risk_tags_vocabulary_is_preserved_from_analyzer_output() -> None:
    payload = _analysis_output(top_stage=Stage.SYNTHESIS)
    payload["risk_tags"] = [
        "sprawl",
        "need_reframing",
        "single_point_failure",
        "elegant_theory_drift",
        "coherence_over_truth",
        "false_unification",
        "high_stakes",
        "abstract_structural_task",
    ]
    client = StubModelClient([{"response": json.dumps(payload)}])
    analyzer = TaskAnalyzer(model_client=client, model="stub")

    result = analyzer.analyze("task")

    assert result is not None
    assert result.risk_tags == payload["risk_tags"]
