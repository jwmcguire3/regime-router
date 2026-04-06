from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.control import EscalationPolicy
from router.llm import ModelClient
from router.models import RegimeConfidenceResult, RoutingDecision, Stage, TaskAnalyzerOutput
from router.routing import RegimeComposer, Router
from router.runtime.planner import RuntimePlanner


class FixedDecisionAnalyzer:
    def __init__(self, decision: RoutingDecision) -> None:
        self.decision = decision

    def analyze(
        self,
        task: str,
    ) -> Optional[TaskAnalyzerOutput]:
        return None

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
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
        fragility_pressure=0,
        possibility_space_need=0,
        synthesis_pressure=0,
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
        task_analyzer=analyzer,
        handoff_expected=True,
        analyzer_result=_analysis(),
    )

    first_decision, first_regime, first_handoff, first_state = planner.plan(**args)
    second_decision, second_regime, second_handoff, second_state = planner.plan(**args)

    assert first_decision == second_decision
    assert first_regime == second_regime
    assert first_handoff == second_handoff
    assert first_state == second_state
    assert first_state.planned_switch_condition == "deterministic switch"
    assert first_state.observed_switch_cause is None


def test_planner_no_model_calls() -> None:
    planner = _planner()
    analyzer = TaskAnalyzer(model_client=ExplodingModelClient(), model="unused")

    decision, regime, handoff, state = planner.plan(
        "Summarize this meeting notes list.",
        router_state=None,
        task_analyzer=analyzer,
        analyzer_result=_analysis(),
    )

    assert decision is not None
    assert regime is not None
    assert handoff is not None
    assert state is not None


def test_planner_threads_analyzer_evidence_quality_into_router_state() -> None:
    planner = _planner()
    analyzer = FixedDecisionAnalyzer(_decision())
    analyzer_result = _analysis()
    analyzer_result = TaskAnalyzerOutput(
        bottleneck_label=analyzer_result.bottleneck_label,
        candidate_regimes=analyzer_result.candidate_regimes,
        stage_scores=analyzer_result.stage_scores,
        structural_signals=analyzer_result.structural_signals,
        decision_pressure=analyzer_result.decision_pressure,
        fragility_pressure=analyzer_result.fragility_pressure,
        possibility_space_need=analyzer_result.possibility_space_need,
        synthesis_pressure=analyzer_result.synthesis_pressure,
        evidence_quality=7,
        recurrence_potential=analyzer_result.recurrence_potential,
        confidence=analyzer_result.confidence,
        rationale=analyzer_result.rationale,
        likely_endpoint_regime=analyzer_result.likely_endpoint_regime,
        endpoint_confidence=analyzer_result.endpoint_confidence,
    )

    _, _, _, state = planner.plan(
        "Write a deterministic parser.",
        router_state=None,
        task_analyzer=analyzer,
        analyzer_result=analyzer_result,
    )

    assert state.evidence_quality == 7.0
