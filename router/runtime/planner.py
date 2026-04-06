from __future__ import annotations

from typing import Optional, Tuple

from ..analyzer import TaskAnalyzer
from ..control import EscalationPolicy
from ..models import Regime, RegimeConfidenceResult, RoutingDecision, Severity, Stage, TaskAnalyzerOutput
from ..routing import RegimeComposer, Router, extract_routing_features
from ..state import Handoff, RouterState
from .state_updater import build_router_state, handoff_from_state


class RuntimePlanner:
    """RuntimePlanner is deterministic.

    Given identical inputs, it always produces identical outputs. It never calls
    the model client or uses non-deterministic values.
    """

    def __init__(
        self,
        *,
        router: Router,
        composer: RegimeComposer,
        escalation_policy: EscalationPolicy,
    ) -> None:
        self.router = router
        self.composer = composer
        self.escalation_policy = escalation_policy

    def plan(
        self,
        bottleneck: str,
        *,
        router_state: Optional[RouterState],
        task_analyzer: Optional[TaskAnalyzer],
        handoff_expected: bool = True,
        analyzer_result: Optional[TaskAnalyzerOutput] = None,
    ) -> Tuple[RoutingDecision, Regime, Handoff, RouterState]:
        features = extract_routing_features(bottleneck)
        escalation = self.escalation_policy.evaluate(
            state=router_state,
            routing_features=features,
            task_text=bottleneck,
            current_regime=router_state.current_regime if router_state else None,
            regime_confidence=router_state.regime_confidence if router_state else None,
            misrouting_result=None,
        )
        if analyzer_result is not None and task_analyzer is not None:
            decision = task_analyzer.decision_from_analysis(task=bottleneck, analyzer_result=analyzer_result)
            self._audit_analyzer_decision(decision, bottleneck)
        else:
            decision = self.router.route(
                bottleneck,
                task_signals=features.structural_signals,
                risk_profile=set(),
                routing_features=features,
                escalation_policy_result=escalation,
            )
        regime = self.composer.compose(decision.primary_regime, risk_profile=set(), handoff_expected=handoff_expected)
        state = build_router_state(
            bottleneck=decision.bottleneck,
            decision=decision,
            regime=regime,
            signals=features.structural_signals,
            risks=set(),
            features=features,
            composer=self.composer,
            analyzer_result=analyzer_result,
        )
        state.escalation_debug = {
            "direction": escalation.escalation_direction,
            "justification": escalation.justification,
            "biases": {stage.value: v for stage, v in escalation.preferred_regime_biases.items()},
            "switch_pressure_adjustment": escalation.switch_pressure_adjustment,
            "signals": escalation.debug_signals,
        }
        handoff = handoff_from_state(state)
        return decision, regime, handoff, state

    def _audit_analyzer_decision(self, decision: RoutingDecision, task: str) -> None:
        features = extract_routing_features(task)
        warnings: list[str] = []
        if decision.primary_regime == Stage.OPERATOR and features.evidence_demand > 0:
            warnings.append("Analyzer selected operator while lexical audit detected non-zero evidence demand.")
        if decision.primary_regime == Stage.BUILDER and features.recurrence_potential == 0:
            warnings.append("Analyzer selected builder while lexical audit detected zero recurrence potential.")
        if decision.primary_regime in {Stage.OPERATOR, Stage.SYNTHESIS} and features.fragility_pressure > 0:
            warnings.append("Analyzer selected non-adversarial primary while lexical audit detected fragility pressure.")
        if not warnings:
            return
        decision.policy_warnings.extend(warnings)
        current_level = decision.confidence.level
        softened_level = current_level
        if current_level == Severity.HIGH.value:
            softened_level = Severity.MEDIUM.value
        elif current_level == Severity.MEDIUM.value:
            softened_level = Severity.LOW.value
        if softened_level == current_level:
            return
        decision.confidence = RegimeConfidenceResult(
            level=softened_level,
            rationale=f"{decision.confidence.rationale} Softened by lexical audit.",
            top_stage_score=decision.confidence.top_stage_score,
            runner_up_score=decision.confidence.runner_up_score,
            score_gap=decision.confidence.score_gap,
            nontrivial_stage_count=decision.confidence.nontrivial_stage_count,
            weak_lexical_dependence=decision.confidence.weak_lexical_dependence,
            structural_feature_state=decision.confidence.structural_feature_state,
        )
