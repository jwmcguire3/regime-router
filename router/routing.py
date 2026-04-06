from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .models import (
    RegimeConfidenceResult,
    RoutingDecision,
    RoutingFeatures,
    Severity,
    Stage,
    TaskAnalyzerOutput,
)
from .state import RouterState

if TYPE_CHECKING:
    from .control import EscalationPolicyResult


def _load_routing_support_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, Path(__file__).with_name("routing").joinpath(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from router/routing/{filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_feature_extraction_module = _load_routing_support_module("router.routing.feature_extraction", "feature_extraction.py")
_risk_inference_module = _load_routing_support_module("router.routing.risk_inference", "risk_inference.py")
_grammar_rules_module = _load_routing_support_module("router.routing.grammar_rules", "grammar_rules.py")

deduplicate_lines = _grammar_rules_module.deduplicate_lines
has_hard_conflict = _grammar_rules_module.has_hard_conflict
has_soft_conflict = _grammar_rules_module.has_soft_conflict
resolve_conflict = _grammar_rules_module.resolve_conflict
validate_regime_grammar = _grammar_rules_module.validate_regime_grammar


def _contains_any(text: str, phrases: Tuple[str, ...]) -> List[str]:
    return _feature_extraction_module.contains_any(text, phrases)


def _has_phrase(text: str, phrase: str) -> bool:
    return _feature_extraction_module.has_phrase(text, phrase)


def _score_from_matches(*matches: List[str]) -> int:
    return _feature_extraction_module.score_from_matches(*matches)


def extract_routing_features(task: str) -> RoutingFeatures:
    return _feature_extraction_module.extract_routing_features(task)


def extract_structural_signals(task: str) -> List[str]:
    return _feature_extraction_module.extract_structural_signals(task)


def explain_feature_matches(features: RoutingFeatures) -> Dict[str, List[str]]:
    return _feature_extraction_module.explain_feature_matches(features)


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    return _risk_inference_module.infer_risk_profile(task, risk_profile)


# ============================================================
# Core enums


class Router:
    def __init__(self) -> None:
        pass

    def _score_stages(self, features: RoutingFeatures) -> Dict[Stage, int]:
        scores: Dict[Stage, int] = {
            Stage.EXPLORATION: features.possibility_space_need,
            Stage.SYNTHESIS: max(0, 5 - features.possibility_space_need),
            Stage.EPISTEMIC: features.evidence_demand,
            Stage.ADVERSARIAL: features.fragility_pressure,
            Stage.OPERATOR: features.decision_pressure,
            Stage.BUILDER: features.recurrence_potential,
        }

        has_decision_markers = bool(features.detected_markers.get("decision_tradeoff_commitment"))
        if has_decision_markers and scores[Stage.OPERATOR] > 0 and scores[Stage.SYNTHESIS] > 0:
            scores[Stage.SYNTHESIS] = max(0, scores[Stage.SYNTHESIS] - 2)
        if scores[Stage.OPERATOR] > 0 and not has_decision_markers:
            scores[Stage.OPERATOR] = max(0, scores[Stage.OPERATOR] - 2)

        has_recurrence_markers = bool(features.detected_markers.get("recurrence_systemization"))
        if scores[Stage.BUILDER] > 0 and not has_recurrence_markers:
            scores[Stage.BUILDER] = max(0, scores[Stage.BUILDER] - 2)

        has_fragility_markers = bool(features.detected_markers.get("fragility_launch_trust"))
        if scores[Stage.ADVERSARIAL] > 0 and not has_fragility_markers:
            scores[Stage.ADVERSARIAL] = max(0, scores[Stage.ADVERSARIAL] - 2)

        return scores

    def _score_stages_from_state(self, state: RouterState) -> Dict[Stage, int]:
        scores: Dict[Stage, int] = {
            Stage.EXPLORATION: int(state.possibility_space_need),
            Stage.SYNTHESIS: int(state.synthesis_pressure),
            Stage.EPISTEMIC: int(state.evidence_demand),
            Stage.ADVERSARIAL: int(state.fragility_pressure),
            Stage.OPERATOR: int(state.decision_pressure),
            Stage.BUILDER: int(state.recurrence_potential),
        }

        if state.executed_regime_stages:
            last_executed = state.executed_regime_stages[-1]
            if scores[last_executed] > 0:
                scores[last_executed] -= 1

        return scores

    def route_switch(self, state: RouterState) -> RoutingDecision:
        scores = self._score_stages_from_state(state)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], list(Stage).index(item[0])))
        primary = ranked[0][0] if ranked else Stage.EXPLORATION
        runner_up = next((stage for stage, _ in ranked if stage != primary), Stage.SYNTHESIS)
        top_score = ranked[0][1] if ranked else 0
        runner_up_score = next((score for stage, score in ranked if stage == runner_up), 0)
        nontrivial_stage_count = sum(1 for score in scores.values() if score > 0)
        score_gap = max(0, top_score - runner_up_score)

        if top_score == 0:
            confidence = RegimeConfidenceResult(
                level=Severity.LOW.value,
                rationale="No positive stage signal detected from router state; using conservative fallback.",
                top_stage_score=top_score,
                runner_up_score=runner_up_score,
                score_gap=score_gap,
                nontrivial_stage_count=nontrivial_stage_count,
                weak_lexical_dependence=False,
                structural_feature_state="rich" if state.structural_signals else "sparse",
            )
            return RoutingDecision(
                bottleneck=state.current_bottleneck,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="State scores are flat; exploration is the safest switching fallback.",
                switch_trigger="Switch when one stage accumulates concrete state pressure above exploration fallback.",
                likely_endpoint_regime=Stage.OPERATOR.value,
                endpoint_confidence=0.3,
                confidence=confidence,
                deterministic_stage_scores=scores,
                deterministic_score_summary=f"state-led fallback: {', '.join(f'{stage.value}={score}' for stage, score in ranked)}",
                analyzer_enabled=False,
                analyzer_used=False,
                analyzer_summary="fallback: state scores were all zero",
                inference_quality="state_led",
            )

        if top_score >= 7 or score_gap >= 3:
            confidence_level = Severity.HIGH.value
        elif top_score >= 4:
            confidence_level = Severity.MEDIUM.value
        else:
            confidence_level = Severity.LOW.value

        endpoint_stage = Stage.OPERATOR
        endpoint_confidence = 0.55
        if primary == Stage.BUILDER and state.recurrence_potential > 0:
            endpoint_stage = Stage.BUILDER
            endpoint_confidence = 0.6
        elif primary in (Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL):
            endpoint_confidence = 0.5

        confidence = RegimeConfidenceResult(
            level=confidence_level,
            rationale=f"State-led routing from deterministic pressures (gap={score_gap}).",
            top_stage_score=top_score,
            runner_up_score=runner_up_score,
            score_gap=score_gap,
            nontrivial_stage_count=nontrivial_stage_count,
            weak_lexical_dependence=False,
            structural_feature_state="rich" if state.structural_signals else "sparse",
        )
        summary = ", ".join(f"{stage.value}={score}" for stage, score in ranked)

        return RoutingDecision(
            bottleneck=state.current_bottleneck,
            primary_regime=primary,
            runner_up_regime=runner_up,
            why_primary_wins_now=f"{primary.value} has the highest state pressure score ({top_score}).",
            switch_trigger=f"Switch when {runner_up.value} pressure overtakes {primary.value} by at least 1 point.",
            likely_endpoint_regime=endpoint_stage.value,
            endpoint_confidence=endpoint_confidence,
            confidence=confidence,
            deterministic_stage_scores=scores,
            deterministic_score_summary=f"state-led stage scores: {summary}",
            analyzer_enabled=False,
            analyzer_used=False,
            analyzer_summary="state_led: deterministic routing from router state",
            inference_quality="state_led",
        )

    def route(
        self,
        bottleneck: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        routing_features: Optional[RoutingFeatures] = None,
        escalation_policy_result: Optional["EscalationPolicyResult"] = None,
        analyzer_result: Optional[TaskAnalyzerOutput] = None,
        analyzer_enabled: bool = False,
    ) -> RoutingDecision:
        _ = (task_signals, risk_profile, escalation_policy_result, analyzer_result)
        if routing_features is None:
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="Insufficient feature evidence; defaulting to exploration fallback.",
                switch_trigger="Switch when routing features become concrete enough to score competing stages.",
                endpoint_confidence=0.3,
                confidence=RegimeConfidenceResult.low_default(),
                analyzer_enabled=analyzer_enabled,
                analyzer_used=False,
                analyzer_summary="fallback: routing features unavailable",
                inference_quality="fallback",
            )

        scores = self._score_stages(routing_features)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], list(Stage).index(item[0])))
        primary = ranked[0][0] if ranked else Stage.EXPLORATION
        runner_up = next((stage for stage, _ in ranked if stage != primary), Stage.SYNTHESIS)
        top_score = ranked[0][1] if ranked else 0
        runner_up_score = next((score for stage, score in ranked if stage == runner_up), 0)
        nontrivial_stage_count = sum(1 for score in scores.values() if score > 0)
        score_gap = max(0, top_score - runner_up_score)

        if top_score == 0:
            confidence = RegimeConfidenceResult(
                level=Severity.LOW.value,
                rationale="No positive stage signal detected; using conservative fallback.",
                top_stage_score=top_score,
                runner_up_score=runner_up_score,
                score_gap=score_gap,
                nontrivial_stage_count=nontrivial_stage_count,
                weak_lexical_dependence=True,
                structural_feature_state="rich" if routing_features.structural_signals else "sparse",
            )
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="Feature scores are flat; exploration is the safest starting regime.",
                switch_trigger="Switch when one stage accumulates concrete pressure above exploration fallback.",
                likely_endpoint_regime=Stage.OPERATOR.value,
                endpoint_confidence=0.3,
                confidence=confidence,
                deterministic_stage_scores=scores,
                deterministic_score_summary=f"feature-led fallback: {', '.join(f'{stage.value}={score}' for stage, score in ranked)}",
                analyzer_enabled=analyzer_enabled,
                analyzer_used=False,
                analyzer_summary="fallback: feature scores were all zero",
                inference_quality="fallback",
            )

        if top_score >= 7 or score_gap >= 3:
            confidence_level = Severity.HIGH.value
        elif top_score >= 4:
            confidence_level = Severity.MEDIUM.value
        else:
            confidence_level = Severity.LOW.value

        endpoint_stage = Stage.OPERATOR
        endpoint_confidence = 0.55
        if primary == Stage.BUILDER and routing_features.recurrence_potential > 0:
            endpoint_stage = Stage.BUILDER
            endpoint_confidence = 0.6
        elif primary in (Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL):
            endpoint_confidence = 0.5

        confidence = RegimeConfidenceResult(
            level=confidence_level,
            rationale=f"Feature-led routing from deterministic pressures (gap={score_gap}).",
            top_stage_score=top_score,
            runner_up_score=runner_up_score,
            score_gap=score_gap,
            nontrivial_stage_count=nontrivial_stage_count,
            weak_lexical_dependence=False,
            structural_feature_state="rich" if routing_features.structural_signals else "sparse",
        )
        summary = ", ".join(f"{stage.value}={score}" for stage, score in ranked)

        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=primary,
            runner_up_regime=runner_up,
            why_primary_wins_now=f"{primary.value} has the highest feature pressure score ({top_score}).",
            switch_trigger=f"Switch when {runner_up.value} pressure overtakes {primary.value} by at least 1 point.",
            likely_endpoint_regime=endpoint_stage.value,
            endpoint_confidence=endpoint_confidence,
            confidence=confidence,
            deterministic_stage_scores=scores,
            deterministic_score_summary=f"feature-led stage scores: {summary}",
            analyzer_enabled=analyzer_enabled,
            analyzer_used=False,
            analyzer_summary="feature_led: deterministic routing from extracted features",
            inference_quality="feature_led",
        )
# ============================================================
# Composer
# ============================================================

_composer_spec = spec_from_file_location(
    "router.routing.composer",
    Path(__file__).with_name("routing").joinpath("composer.py"),
)
if _composer_spec is None or _composer_spec.loader is None:
    raise ImportError("Unable to load RegimeComposer from router/routing/composer.py")
_composer_module = module_from_spec(_composer_spec)
_composer_spec.loader.exec_module(_composer_module)
RegimeComposer = _composer_module.RegimeComposer

_grammar_composer_spec = spec_from_file_location(
    "router.routing.grammar_composer",
    Path(__file__).with_name("routing").joinpath("grammar_composer.py"),
)
if _grammar_composer_spec is None or _grammar_composer_spec.loader is None:
    raise ImportError("Unable to load GrammarComposer from router/routing/grammar_composer.py")
_grammar_composer_module = module_from_spec(_grammar_composer_spec)
_grammar_composer_spec.loader.exec_module(_grammar_composer_module)
GrammarComposer = _grammar_composer_module.GrammarComposer


# ============================================================
# Ollama adapter
# ============================================================
