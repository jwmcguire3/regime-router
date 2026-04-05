from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.classifier import TaskClassifier
from router.control import EscalationPolicy
from router.models import RegimeConfidenceResult, RoutingDecision, RoutingFeatures, Stage, TaskAnalyzerOutput
from router.routing import RegimeComposer, Router, extract_routing_features
from router.runtime.planner import RuntimePlanner


class _NoModelClient:
    def generate(self, **_: object):  # pragma: no cover - safety guard
        raise AssertionError("No model calls are expected in these deterministic routing tests")

    def list_models(self):
        return {"models": []}


class _FixedDecisionAnalyzer:
    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: TaskAnalyzerOutput | None,
        routing_features: RoutingFeatures,
    ) -> RoutingDecision:
        return self._decision


def _planner() -> RuntimePlanner:
    return RuntimePlanner(
        router=Router(),
        composer=RegimeComposer(),
        escalation_policy=EscalationPolicy(),
        task_classifier=TaskClassifier(),
    )


def _analyzer() -> TaskAnalyzer:
    return TaskAnalyzer(model_client=_NoModelClient(), model="unused")


def _base_stage_scores(primary: Stage, runner_up: Stage) -> dict[Stage, float]:
    scores = {stage: 0.05 for stage in Stage}
    scores[primary] = 0.95
    scores[runner_up] = 0.6
    return scores


# Family A — analyzer-led path


def test_analyzer_led_high_decision_low_uncertainty_prefers_operator() -> None:
    task = "Decide now between options this week and make a call."
    features = extract_routing_features(task)
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.OPERATOR, Stage.SYNTHESIS],
        stage_scores=_base_stage_scores(Stage.OPERATOR, Stage.SYNTHESIS),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=features.recurrence_potential,
        confidence=0.83,
        rationale="Decision commitment dominates.",
        likely_endpoint_regime=Stage.OPERATOR,
        endpoint_confidence=0.88,
    )

    decision = _analyzer().decision_from_analysis(task=task, analyzer_result=output, routing_features=features)

    assert features.decision_pressure > 0
    assert features.evidence_demand == 0
    assert decision.primary_regime == Stage.OPERATOR


def test_analyzer_led_high_fragility_low_recurrence_prefers_adversarial_with_operator_endpoint() -> None:
    task = "Stress test this launch plan and list failure modes before production deployment."
    features = extract_routing_features(task)
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.ADVERSARIAL, Stage.EPISTEMIC],
        stage_scores=_base_stage_scores(Stage.ADVERSARIAL, Stage.EPISTEMIC),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=0,
        confidence=0.86,
        rationale="Fragility pressure is explicit and recurrence is low.",
        likely_endpoint_regime=Stage.OPERATOR,
        endpoint_confidence=0.87,
    )

    decision = _analyzer().decision_from_analysis(task=task, analyzer_result=output, routing_features=features)

    assert features.fragility_pressure > 0
    assert features.recurrence_potential == 0
    assert decision.primary_regime == Stage.ADVERSARIAL
    assert decision.likely_endpoint_regime == Stage.OPERATOR.value


def test_analyzer_led_confidence_below_half_maps_to_low() -> None:
    task = "Map options before deciding."
    features = extract_routing_features(task)
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.SYNTHESIS, Stage.EXPLORATION],
        stage_scores=_base_stage_scores(Stage.SYNTHESIS, Stage.EXPLORATION),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=features.recurrence_potential,
        confidence=0.49,
        rationale="Confidence is explicitly low.",
        likely_endpoint_regime=Stage.OPERATOR,
        endpoint_confidence=0.6,
    )

    decision = _analyzer().decision_from_analysis(task=task, analyzer_result=output, routing_features=features)

    assert decision.confidence.level == "low"


def test_analyzer_led_builder_endpoint_retained_when_confidence_is_high() -> None:
    task = "Choose one plan now."
    features = extract_routing_features(task)
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.OPERATOR, Stage.SYNTHESIS],
        stage_scores=_base_stage_scores(Stage.OPERATOR, Stage.SYNTHESIS),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=0,
        confidence=0.8,
        rationale="Builder endpoint remains allowed at high confidence.",
        likely_endpoint_regime=Stage.BUILDER,
        endpoint_confidence=0.75,
    )

    decision = _analyzer().decision_from_analysis(task=task, analyzer_result=output, routing_features=features)

    assert decision.likely_endpoint_regime == Stage.BUILDER.value


def test_analyzer_led_endpoint_is_clamped_when_it_precedes_primary() -> None:
    task = "Decide now between options."
    features = extract_routing_features(task)
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.OPERATOR, Stage.SYNTHESIS],
        stage_scores=_base_stage_scores(Stage.OPERATOR, Stage.SYNTHESIS),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=features.recurrence_potential,
        confidence=0.9,
        rationale="Primary is late-stage operator.",
        likely_endpoint_regime=Stage.SYNTHESIS,
        endpoint_confidence=0.8,
    )

    decision = _analyzer().decision_from_analysis(task=task, analyzer_result=output, routing_features=features)

    assert decision.primary_regime == Stage.OPERATOR
    assert decision.likely_endpoint_regime == Stage.OPERATOR.value


# Family B — feature-led fallback path (Router.route placeholder)


def test_feature_led_decision_scores_are_feature_grounded() -> None:
    planner = _planner()

    decision, *_ = planner.plan(
        "Decide now between options this week and make a call.",
        router_state=None,
        use_task_analyzer=False,
        task_analyzer=None,
    )

    assert decision.primary_regime == Stage.OPERATOR
    assert decision.deterministic_stage_scores
    assert decision.deterministic_stage_scores[Stage.OPERATOR] >= decision.deterministic_stage_scores[Stage.EXPLORATION]
    assert decision.inference_quality == "feature_led"


def test_feature_led_decision_has_non_analyzer_quality_marker() -> None:
    planner = _planner()

    decision, *_ = planner.plan(
        "Stress test this launch plan and list failure modes before production deployment.",
        router_state=None,
        use_task_analyzer=False,
        task_analyzer=None,
    )

    assert decision.analyzer_used is False
    assert decision.analyzer_summary is not None
    assert "feature_led" in decision.analyzer_summary
    assert decision.inference_quality == "feature_led"


def test_feature_led_endpoint_confidence_is_lower_than_analyzer_led_for_same_task() -> None:
    task = "Decide now between options this week and make a call."
    features = extract_routing_features(task)
    analyzer_output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[Stage.OPERATOR, Stage.SYNTHESIS],
        stage_scores=_base_stage_scores(Stage.OPERATOR, Stage.SYNTHESIS),
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=features.recurrence_potential,
        confidence=0.9,
        rationale="Operator confidence is moderate.",
        likely_endpoint_regime=Stage.OPERATOR,
        endpoint_confidence=0.62,
    )
    analyzer_decision = _analyzer().decision_from_analysis(
        task=task,
        analyzer_result=analyzer_output,
        routing_features=features,
    )

    planner = _planner()
    fallback_decision, *_ = planner.plan(
        task,
        router_state=None,
        use_task_analyzer=False,
        task_analyzer=None,
    )

    assert fallback_decision.endpoint_confidence < analyzer_decision.endpoint_confidence


def test_feature_led_score_summary_has_no_placeholder_text() -> None:
    planner = _planner()

    decision, *_ = planner.plan(
        "Stress test this launch plan and list failure modes before production deployment.",
        router_state=None,
        use_task_analyzer=False,
        task_analyzer=None,
    )

    summary = " ".join(
        [
            decision.why_primary_wins_now,
            decision.switch_trigger,
            decision.deterministic_score_summary,
        ]
    ).lower()
    assert "awaiting llm proposer" not in summary
    assert "placeholder" not in summary


# Fast-path gate regression


def test_fastpath_blocks_direct_when_fragility_pressure_is_positive() -> None:
    planner = _planner()
    fixed_decision = RoutingDecision(
        bottleneck="task",
        primary_regime=Stage.SYNTHESIS,
        runner_up_regime=Stage.EXPLORATION,
        why_primary_wins_now="fixed analyzer-led decision",
        switch_trigger="fixed",
        confidence=RegimeConfidenceResult.low_default(),
        analyzer_enabled=True,
        analyzer_used=True,
    )
    analyzer = _FixedDecisionAnalyzer(fixed_decision)
    analyzer_result = TaskAnalyzerOutput(
        bottleneck_label="task",
        candidate_regimes=[Stage.SYNTHESIS],
        stage_scores=_base_stage_scores(Stage.SYNTHESIS, Stage.EXPLORATION),
        structural_signals=[],
        decision_pressure=0,
        evidence_quality=0,
        recurrence_potential=0,
        confidence=0.99,
        rationale="single candidate high confidence",
    )

    decision, regime, *_ = planner.plan(
        "Write a script and stress test failure modes before deployment.",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=analyzer,
        analyzer_result=analyzer_result,
    )

    assert decision.primary_regime == Stage.SYNTHESIS
    assert regime.name != "Direct Passthrough"
