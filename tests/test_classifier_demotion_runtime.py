from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.control import EscalationPolicy
from router.models import RegimeConfidenceResult, RoutingDecision, Stage, TaskAnalyzerOutput
from router.routing import RegimeComposer, Router
from router.runtime.planner import RuntimePlanner


class StubAnalyzer:
    def __init__(self, output: Optional[TaskAnalyzerOutput], decision: RoutingDecision) -> None:
        self.output = output
        self.decision = decision
        self.analyze_calls: List[Dict[str, object]] = []
        self.decision_calls: List[Dict[str, object]] = []

    def analyze(self, task: str) -> Optional[TaskAnalyzerOutput]:
        self.analyze_calls.append(
            {
                "task": task,
            }
        )
        return self.output

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
    ) -> RoutingDecision:
        self.decision_calls.append(
            {
                "task": task,
                "analyzer_result": analyzer_result,
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
        fragility_pressure=0,
        possibility_space_need=0,
        synthesis_pressure=0,
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
    )


def test_analyzer_decision_consumes_supplied_analysis() -> None:
    analyzer = StubAnalyzer(output=_analysis(confidence=0.7, candidates=[Stage.SYNTHESIS]), decision=_decision())
    planner = _planner()
    analyzer_result = _analysis(confidence=0.7, candidates=[Stage.SYNTHESIS])

    planner.plan(
        "write a function",
        router_state=None,
        task_analyzer=analyzer,
        analyzer_result=analyzer_result,
    )

    assert len(analyzer.analyze_calls) == 0
    assert len(analyzer.decision_calls) == 1
    assert analyzer.decision_calls[0]["analyzer_result"] == analyzer_result


def test_audit_adds_warning_and_softens_confidence() -> None:
    planner = _planner()
    analyzer = StubAnalyzer(output=_analysis(confidence=0.95, candidates=[Stage.OPERATOR]), decision=_decision(primary=Stage.OPERATOR))
    decision, *_ = planner.plan(
        "Decide now using evidence from multiple conflicting sources.",
        router_state=None,
        task_analyzer=analyzer,
        analyzer_result=_analysis(confidence=0.95, candidates=[Stage.OPERATOR]),
    )

    assert decision.confidence.level == "medium"
    assert decision.policy_warnings


def test_audit_keeps_confidence_when_no_warning_triggered() -> None:
    planner = _planner()
    analyzer = StubAnalyzer(output=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]), decision=_decision())

    decision, *_ = planner.plan(
        "Summarize these ideas into a coherent framing.",
        router_state=None,
        task_analyzer=analyzer,
        analyzer_result=_analysis(confidence=0.95, candidates=[Stage.SYNTHESIS]),
    )

    assert decision.confidence.level == "high"
    assert decision.policy_warnings == []
