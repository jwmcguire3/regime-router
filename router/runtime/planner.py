from __future__ import annotations

from typing import List, Optional, Set, Tuple

from ..analyzer import TaskAnalyzer
from ..classifier import TaskClassification, TaskClassifier
from ..control import EscalationPolicy
from ..models import Regime, RoutingDecision, TaskAnalyzerOutput
from ..routing import RegimeComposer, Router, extract_routing_features, infer_risk_profile
from ..state import Handoff, RouterState
from .state_updater import build_router_state, handoff_from_state


class RuntimePlanner:
    def __init__(
        self,
        *,
        router: Router,
        composer: RegimeComposer,
        escalation_policy: EscalationPolicy,
        task_classifier: TaskClassifier,
    ) -> None:
        self.router = router
        self.composer = composer
        self.escalation_policy = escalation_policy
        self.task_classifier = task_classifier

    def plan(
        self,
        bottleneck: str,
        *,
        router_state: Optional[RouterState],
        use_task_analyzer: bool,
        task_analyzer: Optional[TaskAnalyzer],
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = True,
        task_signals: Optional[List[str]] = None,
        risks_inferred: bool = False,
    ) -> Tuple[RoutingDecision, Regime, Handoff, RouterState, TaskClassification]:
        classification = self.task_classifier.classify(bottleneck)
        if classification.route_type == "direct" and classification.classification_source == "pattern":
            decision, regime, handoff, state = self.plan_direct(
                bottleneck,
                handoff_expected=handoff_expected,
                classification=classification,
            )
            return decision, regime, handoff, state, classification

        features = extract_routing_features(bottleneck)
        escalation = self.escalation_policy.evaluate(
            state=router_state,
            routing_features=features,
            task_text=bottleneck,
            current_regime=router_state.current_regime if router_state else None,
            regime_confidence=router_state.regime_confidence if router_state else None,
            misrouting_result=None,
        )
        signals = task_signals if task_signals is not None else features.structural_signals
        risks = set(risk_profile or set()) if risks_inferred else infer_risk_profile(bottleneck, risk_profile)
        deterministic_decision = self.router.route(
            bottleneck,
            task_signals=signals,
            risk_profile=risks,
            routing_features=features,
            escalation_policy_result=escalation,
        )
        analysis: Optional[TaskAnalyzerOutput] = None
        analyzer_attempted = False
        if (
            use_task_analyzer
            and task_analyzer
            and self.router.should_use_analyzer(deterministic_decision.confidence, score_gap_threshold=1)
        ):
            analyzer_attempted = True
            analysis = task_analyzer.analyze(
                bottleneck,
                routing_features=features,
                task_signals=signals,
                risk_profile=risks,
            )

        decision = self.router.route(
            bottleneck,
            task_signals=signals,
            risk_profile=risks,
            routing_features=features,
            escalation_policy_result=escalation,
            deterministic_stage_scores=deterministic_decision.deterministic_stage_scores,
            deterministic_confidence=deterministic_decision.confidence,
            analyzer_enabled=use_task_analyzer,
            analyzer_result=analysis,
            analyzer_gap_threshold=1,
        )
        if analyzer_attempted and analysis is None and decision.analyzer_summary is None:
            analyzer_error = (
                task_analyzer.last_error_summary if task_analyzer and task_analyzer.last_error_summary else "Analyzer returned invalid/non-JSON output."
            )
            decision.analyzer_summary = f"{analyzer_error} Deterministic routing retained."

        regime = self.composer.compose(decision.primary_regime, risk_profile=risks, handoff_expected=handoff_expected)
        state = build_router_state(
            bottleneck=bottleneck,
            decision=decision,
            regime=regime,
            signals=signals,
            risks=risks,
            features=features,
            composer=self.composer,
        )
        state.escalation_debug = {
            "direction": escalation.escalation_direction,
            "justification": escalation.justification,
            "biases": {stage.value: v for stage, v in escalation.preferred_regime_biases.items()},
            "switch_pressure_adjustment": escalation.switch_pressure_adjustment,
            "signals": escalation.debug_signals,
        }
        state.task_classification = {
            "route_type": classification.route_type,
            "confidence": classification.confidence,
            "reason": classification.reason,
        }
        handoff = handoff_from_state(state)
        return decision, regime, handoff, state, classification

    def plan_direct(
        self,
        task: str,
        *,
        handoff_expected: bool,
        classification: TaskClassification,
    ) -> Tuple[RoutingDecision, Regime, Handoff, RouterState]:
        decision = RoutingDecision(
            bottleneck=task,
            primary_regime=None,
            runner_up_regime=None,
            why_primary_wins_now="Direct execution — no reasoning bottleneck detected.",
            switch_trigger="Execute immediately; no regime switching needed.",
        )
        regime = self.composer.compose(None, risk_profile=set(), handoff_expected=handoff_expected)
        regime.name = "Direct Passthrough"
        regime.likely_failure_if_overused = "May skip deeper reasoning when hidden ambiguity exists."
        state = build_router_state(
            bottleneck=task,
            decision=decision,
            regime=regime,
            signals=[],
            risks=set(),
            features=extract_routing_features(task),
            composer=self.composer,
        )
        state.task_classification = {
            "route_type": classification.route_type,
            "confidence": classification.confidence,
            "reason": classification.reason,
        }
        handoff = handoff_from_state(state)
        return decision, regime, handoff, state
