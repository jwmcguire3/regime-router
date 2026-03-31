from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .models import (
    Regime,
    RegimeConfidenceResult,
    RoutingDecision,
    RoutingFeatures,
    Stage,
    TaskAnalyzerOutput,
    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL,
    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED,
    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED,
)

if TYPE_CHECKING:
    from .control import EscalationPolicyResult


def _load_routing_support_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, Path(__file__).with_name("routing").joinpath(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from router/routing/{filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_lexical_tables_module = _load_routing_support_module("router.routing.lexical_tables", "lexical_tables.py")
_feature_extraction_module = _load_routing_support_module("router.routing.feature_extraction", "feature_extraction.py")
_risk_inference_module = _load_routing_support_module("router.routing.risk_inference", "risk_inference.py")
_confidence_module = _load_routing_support_module("router.routing.confidence", "confidence.py")
_analyzer_override_module = _load_routing_support_module("router.routing.analyzer_override", "analyzer_override.py")
_decision_builder_module = _load_routing_support_module("router.routing.decision_builder", "decision_builder.py")
_score_tracking_module = _load_routing_support_module("router.routing.score_tracking", "score_tracking.py")
_grammar_rules_module = _load_routing_support_module("router.routing.grammar_rules", "grammar_rules.py")

LEXICAL_PHRASE_TABLE = _lexical_tables_module.LEXICAL_PHRASE_TABLE
NEGATED_CLOSURE_PHRASES = _lexical_tables_module.NEGATED_CLOSURE_PHRASES
RegimeConfidenceCalculator = _confidence_module.RegimeConfidenceCalculator
StageScoreTracking = _score_tracking_module.StageScoreTracking
apply_deterministic_stage_score_overrides = _score_tracking_module.apply_deterministic_stage_score_overrides

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


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    return _risk_inference_module.infer_risk_profile(task, risk_profile)


# ============================================================
# Core enums


class Router:
    def __init__(self, embedding_router: Optional[object] = None) -> None:
        self.confidence_calculator = RegimeConfidenceCalculator()
        self.embedding_router = embedding_router
        self.precedence_order = [
            Stage.OPERATOR,
            Stage.EXPLORATION,
            Stage.EPISTEMIC,
            Stage.ADVERSARIAL,
            Stage.SYNTHESIS,
            Stage.BUILDER,
        ]

    @staticmethod
    def _format_stage_score_summary(stage_scores: Dict[Stage, int]) -> str:
        return _decision_builder_module.format_stage_score_summary(stage_scores)

    @staticmethod
    def _format_stage_contributions(contributions: Dict[Stage, List[str]]) -> str:
        return _decision_builder_module.format_stage_contributions(contributions)

    @staticmethod
    def _reason_for(stage: Stage) -> Tuple[str, str]:
        return _decision_builder_module.reason_for(stage)

    def should_use_analyzer(self, confidence: RegimeConfidenceResult, score_gap_threshold: int = 1) -> bool:
        return _analyzer_override_module.should_use_analyzer(confidence, score_gap_threshold)

    @staticmethod
    def _analyzer_scores_flat_or_nearly_flat(analyzer_ranked: List[Tuple[Stage, float]]) -> bool:
        return _analyzer_override_module.analyzer_scores_flat_or_nearly_flat(analyzer_ranked)

    @staticmethod
    def _rationale_too_short_or_generic(rationale: str) -> bool:
        return _analyzer_override_module.rationale_too_short_or_generic(rationale)

    def _accept_analyzer_override(
        self,
        *,
        analyzer_result: TaskAnalyzerOutput,
        analyzer_ranked: List[Tuple[Stage, float]],
        features: RoutingFeatures,
        zero_score_fallback: bool,
    ) -> Tuple[bool, List[str]]:
        return _analyzer_override_module.accept_analyzer_override(
            analyzer_result=analyzer_result,
            analyzer_ranked=analyzer_ranked,
            features=features,
            zero_score_fallback=zero_score_fallback,
        )

    def _apply_shortcut_routes(
        self,
        b: str,
        features: RoutingFeatures,
        analyzer_enabled: bool,
    ) -> Optional[RoutingDecision]:
        return _decision_builder_module.apply_shortcut_routes(
            bottleneck=b,
            features=features,
            analyzer_enabled=analyzer_enabled,
            has_phrase=_has_phrase,
            high_shortcut_rationale=RegimeConfidenceCalculator.high_shortcut_rationale,
        )

    def _apply_lexical_scores(
        self,
        b: str,
        features: RoutingFeatures,
        stage_scores: Dict[Stage, int],
        lexical_scores: Dict[Stage, int],
        stage_contributions: Dict[Stage, List[str]],
        add_score,
        suppress_score,
    ) -> None:
        interpretation_shortcut_markers = ["strongest interpretation", "strongest frame", "what this actually is"]
        epistemic_markers = ["evidence", "support", "verify", "unknown", "unknowns", "unclear", "unresolved"]

        for phrase, stage, weight, tag in LEXICAL_PHRASE_TABLE:
            if _has_phrase(b, phrase):
                add_score(stage, weight, "lexical", f"phrase='{tag}'")

        for phrase, weight in NEGATED_CLOSURE_PHRASES.items():
            if not _has_phrase(b, phrase):
                continue
            suppress_score(
                Stage.OPERATOR,
                weight,
                "lexical",
                f"negated_closure:phrase='{phrase}'",
            )
            add_score(
                Stage.EXPLORATION,
                2,
                "lexical",
                f"negated_closure:keep_open_phrase='{phrase}'",
            )

        before_deciding_epistemic_phrases = (
            "before deciding",
            "before we decide",
            "before deciding now",
        )
        if any(_has_phrase(b, phrase) for phrase in before_deciding_epistemic_phrases) and (
            features.evidence_demand > 0 or any(_has_phrase(b, marker) for marker in ("evidence", "proof", "verify", "confidence"))
        ):
            add_score(Stage.EPISTEMIC, 2, "structural", "before_deciding:verification_precedence")
            suppress_score(Stage.OPERATOR, 2, "structural", "before_deciding:suppress_premature_closure")

        if "pattern" in b:
            add_score(Stage.SYNTHESIS, 1, "lexical", "generic_pattern_signal")

        has_interpretation_shortcut_marker = any(_has_phrase(b, k) for k in interpretation_shortcut_markers)
        has_epistemic_marker = any(_has_phrase(b, k) for k in epistemic_markers)
        if has_interpretation_shortcut_marker:
            if not has_epistemic_marker:
                add_score(Stage.SYNTHESIS, 4, "lexical", "interpretation_shortcut_marker")
            else:
                add_score(Stage.EPISTEMIC, 2, "lexical", "epistemic_marker_with_interpretation_shortcut")

        # "options" alone is intentionally weak to avoid swallowing decision tasks.
        if "options" in b:
            add_score(Stage.EXPLORATION, 1, "lexical", "weak_options_token")

        if any(
            phrase in b
            for phrase in [
                "parts are legible",
                "whole organizing logic is missing",
                "fragments but no spine",
                "fragments are understood",
                "spine is still missing",
                "many signals but no center",
            ]
        ):
            add_score(Stage.SYNTHESIS, 4, "lexical", "parts_whole_mismatch_phrase_cluster")

    def _apply_structural_scores(
        self,
        b: str,
        features: RoutingFeatures,
        risks: Set[str],
        signals: Set[str],
        stage_scores: Dict[Stage, int],
        structural_scores: Dict[Stage, int],
        stage_contributions: Dict[Stage, List[str]],
        add_score,
        suppress_score,
        escalation_policy_result: Optional["EscalationPolicyResult"] = None,
    ) -> None:
        if STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in signals:
            add_score(Stage.SYNTHESIS, 5, "structural", STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED)
            add_score(Stage.EPISTEMIC, 1, "structural", "fragments_spine_gap:verification_followup")
        if STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL in signals:
            add_score(Stage.SYNTHESIS, 2, "structural", STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL)
        if STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED in signals:
            add_score(Stage.SYNTHESIS, 2, "structural", STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED)
        if "parts_whole_mismatch" in features.detected_markers:
            add_score(Stage.SYNTHESIS, 3, "structural", "detected_marker:parts_whole_mismatch")
        if "abstract_structural_task" in risks:
            add_score(Stage.SYNTHESIS, 2, "structural", "risk:abstract_structural_task")
        if "false_unification" in risks:
            add_score(Stage.SYNTHESIS, 2, "structural", "risk:false_unification")

        if features.evidence_demand > 0:
            add_score(
                Stage.EPISTEMIC,
                2 + min(3, features.evidence_demand // 2),
                "structural",
                f"feature:evidence_demand={features.evidence_demand}",
            )
            if "uncertainty_evidence_demand" in features.detected_markers:
                add_score(Stage.EXPLORATION, 1, "structural", "feature_support:uncertainty_can_require_exploration")
            if "uncertainty_characterization" in features.detected_markers:
                add_score(Stage.EPISTEMIC, 2, "structural", "feature:uncertainty_characterization")
                add_score(Stage.SYNTHESIS, 2, "structural", "feature_support:pattern_needs_structural_frame")
        if features.decision_pressure > 0:
            decision_markers = set(features.detected_markers.get("decision_tradeoff_commitment", []))
            has_explicit_decision_marker = bool(decision_markers & RegimeConfidenceCalculator.EXPLICIT_DECISION_MARKERS)
            add_score(
                Stage.OPERATOR,
                1 + min(3, features.decision_pressure // 2) + (1 if has_explicit_decision_marker else 0),
                "structural",
                f"feature:decision_pressure={features.decision_pressure}",
            )
        if features.fragility_pressure > 0:
            add_score(
                Stage.ADVERSARIAL,
                1 + min(3, features.fragility_pressure // 2),
                "structural",
                f"feature:fragility_pressure={features.fragility_pressure}",
            )
        if "recurrence_systemization_strong" in features.detected_markers:
            add_score(
                Stage.BUILDER,
                1 + min(3, features.recurrence_potential // 2),
                "structural",
                f"feature:recurrence_potential={features.recurrence_potential}",
            )
        if features.recurrence_potential > 0:
            add_score(
                Stage.SYNTHESIS,
                1 + min(2, features.recurrence_potential // 4),
                "structural",
                "feature_support:pattern_or_systemization",
            )
        if features.possibility_space_need > 0:
            add_score(
                Stage.EXPLORATION,
                1 + min(3, features.possibility_space_need // 2),
                "structural",
                f"feature:possibility_space_need={features.possibility_space_need}",
            )
        if "open_possibility_space" in features.detected_markers and "anti_convergence_preference" in features.detected_markers:
            add_score(Stage.EXPLORATION, 3, "structural", "anti_convergence:keep_space_open")
            suppress_score(Stage.OPERATOR, 3, "structural", "anti_convergence:suppress_forced_closure")
        if "negated_closure_preference" in features.detected_markers:
            add_score(Stage.EXPLORATION, 2, "structural", "negated_closure:delay_commitment")
            suppress_score(Stage.OPERATOR, 2, "structural", "negated_closure:suppress_closure_cues")
        explicit_open_space_markers = set(features.detected_markers.get("open_possibility_space", []))
        open_space_advantage_markers = {
            "multiple frames",
            "multiple possible frames",
            "multiple perspectives",
            "multiple interpretations",
            "perspectives",
            "interpretations",
            "map the space",
            "keep it open",
            "rather than converging",
            "instead of converging",
            "delay convergence",
            "delaying convergence",
            "before narrowing",
        }
        if explicit_open_space_markers & open_space_advantage_markers:
            add_score(
                Stage.EXPLORATION,
                2,
                "structural",
                "explicit_open_space_request:frames_perspectives_keep_open",
            )
        if (
            "negated_closure_preference" in features.detected_markers
            and not (explicit_open_space_markers & open_space_advantage_markers)
        ):
            add_score(
                Stage.OPERATOR,
                1,
                "structural",
                "negated_closure:operator_retained_as_secondary_context",
            )
        if features.decision_pressure > 0 and features.possibility_space_need > 0:
            explicit_closure_markers = (
                set(features.detected_markers.get("decision_tradeoff_commitment", []))
                & RegimeConfidenceCalculator.EXPLICIT_DECISION_MARKERS
            )
            has_explicit_anti_convergence = "anti_convergence_preference" in features.detected_markers
            if explicit_closure_markers:
                add_score(
                    Stage.OPERATOR,
                    2,
                    "structural",
                    "mixed_prompt:explicit_decision_now_precedence",
                )
                add_score(
                    Stage.EXPLORATION,
                    2,
                    "structural",
                    "mixed_prompt:exploration_retained_as_runner_up",
                )
            if has_explicit_anti_convergence:
                add_score(
                    Stage.EXPLORATION,
                    1,
                    "structural",
                    "mixed_prompt:explicit_keep_open_precedence",
                )
            if explicit_closure_markers:
                suppress_score(
                    Stage.EXPLORATION,
                    2,
                    "structural",
                    "mixed_prompt:hard_closure_overrides_open_space_bonus",
                )

        if escalation_policy_result and escalation_policy_result.escalation_direction != "none":
            for stage, bias in escalation_policy_result.preferred_regime_biases.items():
                if bias > 0:
                    add_score(stage, bias, "structural", f"escalation_policy:{escalation_policy_result.escalation_direction}")

    def _apply_embedding_scores(
        self,
        bottleneck: str,
        stage_scores: Dict[Stage, int],
        stage_contributions: Dict[Stage, List[str]],
        add_score,
    ) -> None:
        if self.embedding_router is not None:
            embedding_score = self.embedding_router.score(bottleneck)
            if not embedding_score.below_threshold:
                ranked_embedding = sorted(
                    embedding_score.stage_scores.items(),
                    key=lambda x: (-x[1], self.precedence_order.index(x[0])),
                )
                if ranked_embedding:
                    best_stage, best_value = ranked_embedding[0]
                    if best_value > 0.35:
                        add_score(best_stage, 3, "embedding", f"best_similarity={best_value:.3f}")
                if len(ranked_embedding) > 1:
                    second_stage, second_value = ranked_embedding[1]
                    if second_value > 0.35:
                        add_score(second_stage, 1, "embedding", f"second_similarity={second_value:.3f}")

    def _build_final_decision(
        self,
        *,
        bottleneck: str,
        features: RoutingFeatures,
        stage_scores: Dict[Stage, int],
        lexical_scores: Dict[Stage, int],
        structural_scores: Dict[Stage, int],
        stage_contributions: Dict[Stage, List[str]],
        deterministic_confidence: Optional[RegimeConfidenceResult],
        analyzer_enabled: bool,
        analyzer_gap_threshold: int,
        analyzer_result: Optional[TaskAnalyzerOutput],
    ) -> RoutingDecision:
        return _decision_builder_module.build_final_decision(
            bottleneck=bottleneck,
            features=features,
            stage_scores=stage_scores,
            lexical_scores=lexical_scores,
            structural_scores=structural_scores,
            stage_contributions=stage_contributions,
            deterministic_confidence=deterministic_confidence,
            analyzer_enabled=analyzer_enabled,
            analyzer_gap_threshold=analyzer_gap_threshold,
            analyzer_result=analyzer_result,
            precedence_order=self.precedence_order,
            confidence_calculator=self.confidence_calculator,
            should_use_analyzer_fn=self.should_use_analyzer,
            accept_analyzer_override_fn=self._accept_analyzer_override,
        )


    def route(
        self,
        bottleneck: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        routing_features: Optional[RoutingFeatures] = None,
        escalation_policy_result: Optional["EscalationPolicyResult"] = None,
        deterministic_stage_scores: Optional[Dict[Stage, int]] = None,
        deterministic_confidence: Optional[RegimeConfidenceResult] = None,
        analyzer_result: Optional[TaskAnalyzerOutput] = None,
        analyzer_enabled: bool = False,
        analyzer_gap_threshold: int = 1,
    ) -> RoutingDecision:
        b = bottleneck.lower().replace("’", "'").strip()
        features = routing_features or extract_routing_features(bottleneck)
        signals = set(task_signals or features.structural_signals)
        risks = set(risk_profile or set())
        if escalation_policy_result is None:
            from .control import EscalationPolicy

            escalation_policy_result = EscalationPolicy().evaluate(
                state=None,
                routing_features=features,
                task_text=bottleneck,
                current_regime=None,
                regime_confidence=deterministic_confidence,
                misrouting_result=None,
            )
        shortcut_decision = self._apply_shortcut_routes(bottleneck, features, analyzer_enabled)
        if shortcut_decision is not None:
            return shortcut_decision

        tracking = StageScoreTracking()

        self._apply_lexical_scores(
            b,
            features,
            tracking.stage_scores,
            tracking.lexical_scores,
            tracking.stage_contributions,
            tracking.add_score,
            tracking.suppress_score,
        )
        self._apply_structural_scores(
            b,
            features,
            risks,
            signals,
            tracking.stage_scores,
            tracking.structural_scores,
            tracking.stage_contributions,
            tracking.add_score,
            tracking.suppress_score,
            escalation_policy_result=escalation_policy_result,
        )
        self._apply_embedding_scores(
            bottleneck,
            tracking.stage_scores,
            tracking.stage_contributions,
            tracking.add_score,
        )

        if deterministic_stage_scores:
            apply_deterministic_stage_score_overrides(
                tracking=tracking,
                deterministic_stage_scores=deterministic_stage_scores,
            )

        return self._build_final_decision(
            bottleneck=bottleneck,
            features=features,
            stage_scores=tracking.stage_scores,
            lexical_scores=tracking.lexical_scores,
            structural_scores=tracking.structural_scores,
            stage_contributions=tracking.stage_contributions,
            deterministic_confidence=deterministic_confidence,
            analyzer_enabled=analyzer_enabled,
            analyzer_gap_threshold=analyzer_gap_threshold,
            analyzer_result=analyzer_result,
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



# ============================================================
# Ollama adapter
# ============================================================
