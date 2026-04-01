from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.classifier import TaskClassifier
from router.control import EscalationPolicy
from router.models import RegimeConfidenceResult, RoutingDecision, RoutingFeatures, Stage, TaskAnalyzerOutput
from router.routing import RegimeComposer, Router
from router.runtime.planner import RuntimePlanner


class StubAnalyzer:
    def __init__(self, output: Optional[TaskAnalyzerOutput], decision: RoutingDecision) -> None:
        self.output = output
        self.decision = decision
        self.analyze_calls: List[Dict[str, object]] = []
        self.decision_calls: List[Dict[str, object]] = []

    def analyze(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
        classifier_signal: Optional[Dict[str, object]] = None,
    ) -> Optional[TaskAnalyzerOutput]:
        self.analyze_calls.append(
            {
                "task": task,
                "routing_features": routing_features,
                "task_signals": task_signals,
                "risk_profile": risk_profile,
                "classifier_signal": classifier_signal,
            }
        )
        return self.output

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
        routing_features: RoutingFeatures,
    ) -> RoutingDecision:
        self.decision_calls.append(
            {
                "task": task,
                "analyzer_result": analyzer_result,
                "routing_features": routing_features,
            }
        )
        return self.decision


def _decision(primary: Stage = Stage.SYNTHESIS) -> RoutingDecision:
    return RoutingDecision(
        bottleneck="analysis",
        primary_regime=primary,
        runner_up_regime=Stage.EXPLORATION,
        why_primary_wins_now="analyzer-selected",
        switch_trigger="switch",
        confidence=RegimeConfidenceResult(
            level="high",
            rationale="Analyzer confidence=0.95.",
            top_stage_score=9,
            runner_up_score=2,
            score_gap=7,
            nontrivial_stage_count=2,
            weak_lexical_dependence=False,
            structural_feature_state="rich",
        ),
        analyzer_enabled=True,
        analyzer_used=True,
    )


def _analysis(*, confidence: float, candidates: List[Stage]) -> TaskAnalyzerOutput:
    return TaskAnalyzerOutput(
        bottleneck_label="analysis",
        candidate_regimes=candidates,
        stage_scores={
            Stage.EXPLORATION: 0.1,
            Stage.SYNTHESIS: 0.9,
            Stage.EPISTEMIC: 0.2,
            Stage.ADVERSARIAL: 0.0,
            Stage.OPERATOR: 0.0,
            Stage.BUILDER: 0.0,
        },
        structural_signals=[],
        decision_pressure=0,
        evidence_quality=0,
        recurrence_potential=0,
        confidence=confidence,
        rationale="stub",
    )


def _planner() -> RuntimePlanner:
    return RuntimePlanner(
        router=Router(),
        composer=RegimeComposer(),
        escalation_policy=EscalationPolicy(),
        task_classifier=TaskClassifier(),
    )


def test_analyzer_decision_consumes_supplied_analysis() -> None:
    analyzer = StubAnalyzer(output=_analysis(confidence=0.7, candidates=[Stage.SYNTHESIS]), decision=_decision())
    planner = _planner()
    analyzer_result = _analysis(confidence=0.7, candidates=[Stage.SYNTHESIS])

    planner.plan(
        "write a function",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=analyzer,
        analyzer_result=analyzer_result,
    )

    assert len(analyzer.analyze_calls) == 0
    assert len(analyzer.decision_calls) == 1
    assert analyzer.decision_calls[0]["analyzer_result"] == analyzer_result


def test_direct_fastpath_requires_all_three() -> None:
    planner = _planner()

    direct_analyzer = StubAnalyzer(output=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]), decision=_decision())
    decision, regime, *_ = planner.plan(
        "write a function",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=direct_analyzer,
        analyzer_result=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]),
    )
    assert decision.primary_regime is None
    assert regime.name == "Direct Passthrough"

    low_conf_analyzer = StubAnalyzer(output=_analysis(confidence=0.9, candidates=[Stage.SYNTHESIS]), decision=_decision())
    decision, regime, *_ = planner.plan(
        "write a function",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=low_conf_analyzer,
        analyzer_result=_analysis(confidence=0.9, candidates=[Stage.SYNTHESIS]),
    )
    assert decision.primary_regime == Stage.SYNTHESIS
    assert regime.name != "Direct Passthrough"

    multi_candidate_analyzer = StubAnalyzer(
        output=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS, Stage.EXPLORATION]), decision=_decision()
    )
    decision, regime, *_ = planner.plan(
        "write a function",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=multi_candidate_analyzer,
        analyzer_result=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS, Stage.EXPLORATION]),
    )
    assert decision.primary_regime == Stage.SYNTHESIS
    assert regime.name != "Direct Passthrough"

    structural_tension_analyzer = StubAnalyzer(output=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]), decision=_decision())
    decision, regime, *_ = planner.plan(
        "write a function and decide now using evidence",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=structural_tension_analyzer,
        analyzer_result=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]),
    )
    assert decision.primary_regime == Stage.SYNTHESIS
    assert regime.name != "Direct Passthrough"


def test_direct_pattern_task_with_structural_tension_not_fastpathed_even_when_classifier_direct() -> None:
    analyzer = StubAnalyzer(output=_analysis(confidence=0.99, candidates=[Stage.SYNTHESIS]), decision=_decision())
    planner = RuntimePlanner(
        router=Router(),
        composer=RegimeComposer(),
        escalation_policy=EscalationPolicy(),
        task_classifier=TaskClassifier(),
    )

    decision, regime, *_ = planner.plan(
        "write a function and decide now using evidence",
        router_state=None,
        use_task_analyzer=True,
        task_analyzer=analyzer,
        analyzer_result=_analysis(confidence=0.99, candidates=[Stage.SYNTHESIS]),
    )

    assert decision.primary_regime == Stage.SYNTHESIS
    assert regime.name != "Direct Passthrough"
