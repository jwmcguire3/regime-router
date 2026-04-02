from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.classifier import TaskClassifier
from router.control import EscalationPolicy
from router.llm import ModelClient
from router.models import RegimeConfidenceResult, RoutingDecision, RoutingFeatures, Stage, TaskAnalyzerOutput
from router.routing import RegimeComposer, Router
from router.runtime.planner import RuntimePlanner


class FixedDecisionAnalyzer:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision

    def analyze(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
        classifier_signal: Optional[Dict[str, object]] = None,
    ) -> Optional[TaskAnalyzerOutput]:
        return None

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
        routing_features: RoutingFeatures,
    ) -> RoutingDecision:
        return self.decision


class ExplodingModelClient(ModelClient):
    def generate(self, **_: object) -> Dict[str, object]:
        raise AssertionError("model client should not be called by RuntimePlanner")

    def list_models(self) -> Dict[str, object]:
        return {"models": []}


def _planner() -> RuntimePlanner:
    return RuntimePlanner(
        router=Router(),
        composer=RegimeComposer(),
        escalation_policy=EscalationPolicy(),
        task_classifier=TaskClassifier(),
    )


def _decision() -> RoutingDecision:
    return RoutingDecision(
        bottleneck="task",
        primary_regime=Stage.SYNTHESIS,
        runner_up_regime=Stage.EXPLORATION,
        why_primary_wins_now="deterministic analyzer decision",
        switch_trigger="deterministic switch",
        confidence=RegimeConfidenceResult.low_default(),
        analyzer_enabled=True,
        analyzer_used=True,
    )


def _analysis() -> TaskAnalyzerOutput:
    return TaskAnalyzerOutput(
        bottleneck_label="task",
        candidate_regimes=[Stage.SYNTHESIS],
        stage_scores={
            Stage.EXPLORATION: 0.1,
            Stage.SYNTHESIS: 0.9,
            Stage.EPISTEMIC: 0.0,
            Stage.ADVERSARIAL: 0.0,
            Stage.OPERATOR: 0.0,
            Stage.BUILDER: 0.0,
        },
        structural_signals=[],
        decision_pressure=0,
        evidence_quality=0,
        recurrence_potential=0,
        confidence=0.95,
        rationale="stable",
    )


def test_planner_deterministic() -> None:
    planner = _planner()
    analyzer = FixedDecisionAnalyzer(_decision())
    args = dict(
        bottleneck="Write a deterministic parser.",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=analyzer,
        risk_profile={"high_stakes"},
        handoff_expected=True,
        task_signals=["explicit_output"],
        risks_inferred=True,
        analyzer_result=_analysis(),
    )

    first_decision, first_regime, first_handoff, first_state, _ = planner.plan(**args)
    second_decision, second_regime, second_handoff, second_state, _ = planner.plan(**args)

    assert first_decision == second_decision
    assert first_regime == second_regime
    assert first_handoff == second_handoff
    assert first_state == second_state
    assert first_state.planned_switch_condition == "deterministic switch"
    assert first_state.observed_switch_cause is None


def test_planner_no_model_calls() -> None:
    planner = _planner()
    analyzer = TaskAnalyzer(model_client=ExplodingModelClient(), model="unused")

    decision, regime, handoff, state, _ = planner.plan(
        "Summarize this meeting notes list.",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=analyzer,
        analyzer_result=_analysis(),
    )

    assert decision is not None
    assert regime is not None
    assert handoff is not None
    assert state is not None
