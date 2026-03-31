from __future__ import annotations

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .models import (
    Regime,
    RegimeConfidenceResult,
    RoutingDecision,
    RoutingFeatures,
    Severity,
    Stage,
    TaskAnalyzerOutput,
    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL,
    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED,
    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED,
)

if TYPE_CHECKING:
    from .control import EscalationPolicyResult

def _contains_any(text: str, phrases: Tuple[str, ...]) -> List[str]:
    matches: List[str] = []
    for phrase in phrases:
        if " " in phrase:
            if phrase in text:
                matches.append(phrase)
            continue

        if re.search(rf"\b{re.escape(phrase)}\b", text):
            matches.append(phrase)

    return matches


def _has_phrase(text: str, phrase: str) -> bool:
    if " " in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def _score_from_matches(*matches: List[str]) -> int:
    return min(10, sum(len(group) for group in matches))


LEXICAL_PHRASE_TABLE: tuple[tuple[str, Stage, int, str], ...] = (
    # Operator
    ("decide", Stage.OPERATOR, 4, "decide"),
    ("deciding", Stage.OPERATOR, 4, "deciding"),
    ("decision", Stage.OPERATOR, 4, "decision"),
    ("choose", Stage.OPERATOR, 4, "choose"),
    ("choosing", Stage.OPERATOR, 4, "choosing"),
    ("recommend", Stage.OPERATOR, 4, "recommend"),
    ("recommendation", Stage.OPERATOR, 4, "recommendation"),
    ("make a call", Stage.OPERATOR, 5, "make a call"),
    ("what should we do", Stage.OPERATOR, 5, "what should we do"),
    ("now", Stage.OPERATOR, 2, "now"),
    ("this week", Stage.OPERATOR, 2, "this week"),
    ("immediate", Stage.OPERATOR, 3, "immediate"),
    ("commit", Stage.OPERATOR, 3, "commit"),
    ("next move", Stage.OPERATOR, 4, "next move"),
    ("tradeoff", Stage.OPERATOR, 4, "tradeoff"),
    ("best option now", Stage.OPERATOR, 6, "best option now"),
    ("select", Stage.OPERATOR, 4, "select"),
    ("selecting", Stage.OPERATOR, 4, "selecting"),
    ("select between options", Stage.OPERATOR, 5, "select between options"),
    ("choose between", Stage.OPERATOR, 5, "choose between"),
    ("time pressure", Stage.OPERATOR, 3, "time pressure"),
    # Epistemic
    ("unknown", Stage.EPISTEMIC, 4, "unknown"),
    ("unknowns", Stage.EPISTEMIC, 4, "unknowns"),
    ("unclear", Stage.EPISTEMIC, 4, "unclear"),
    ("unresolved", Stage.EPISTEMIC, 4, "unresolved"),
    ("what is missing", Stage.EPISTEMIC, 5, "what is missing"),
    ("what do we not know", Stage.EPISTEMIC, 5, "what do we not know"),
    ("support", Stage.EPISTEMIC, 3, "support"),
    ("evidence", Stage.EPISTEMIC, 4, "evidence"),
    ("verify", Stage.EPISTEMIC, 4, "verify"),
    ("rigor", Stage.EPISTEMIC, 3, "rigor"),
    ("proof", Stage.EPISTEMIC, 3, "proof"),
    ("confidence", Stage.EPISTEMIC, 2, "confidence"),
    ("are you sure", Stage.EPISTEMIC, 4, "are you sure"),
    ("can't tell", Stage.EPISTEMIC, 4, "can't tell"),
    ("can't tell what kind", Stage.EPISTEMIC, 5, "can't tell what kind"),
    ("don't know what kind", Stage.EPISTEMIC, 5, "don't know what kind"),
    ("hard to characterize", Stage.EPISTEMIC, 4, "hard to characterize"),
    ("can't characterize", Stage.EPISTEMIC, 4, "can't characterize"),
    ("can't identify", Stage.EPISTEMIC, 4, "can't identify"),
    ("can't name it yet", Stage.EPISTEMIC, 4, "can't name it yet"),
    # Synthesis
    ("strongest interpretation", Stage.SYNTHESIS, 10, "strongest interpretation"),
    ("strongest frame", Stage.SYNTHESIS, 10, "strongest frame"),
    ("what this actually is", Stage.SYNTHESIS, 10, "what this actually is"),
    ("many signals", Stage.SYNTHESIS, 4, "many signals"),
    ("no center", Stage.SYNTHESIS, 4, "no center"),
    ("parts are legible", Stage.SYNTHESIS, 5, "parts are legible"),
    ("whole organizing logic is missing", Stage.SYNTHESIS, 6, "whole organizing logic is missing"),
    ("fragments but no spine", Stage.SYNTHESIS, 6, "fragments but no spine"),
    ("fragments are understood", Stage.SYNTHESIS, 5, "fragments are understood"),
    ("spine is still missing", Stage.SYNTHESIS, 6, "spine is still missing"),
    ("hidden spine", Stage.SYNTHESIS, 4, "hidden spine"),
    ("what this really is", Stage.SYNTHESIS, 4, "what this really is"),
    ("unify", Stage.SYNTHESIS, 4, "unify"),
    ("coherent picture", Stage.SYNTHESIS, 5, "coherent picture"),
    ("coherent", Stage.SYNTHESIS, 2, "coherent"),
    # Adversarial
    ("weakest points", Stage.ADVERSARIAL, 5, "weakest points"),
    ("weak spots", Stage.ADVERSARIAL, 5, "weak spots"),
    ("strongest objections", Stage.ADVERSARIAL, 6, "strongest objections"),
    ("vulnerabilities", Stage.ADVERSARIAL, 6, "vulnerabilities"),
    ("failure modes", Stage.ADVERSARIAL, 6, "failure modes"),
    ("where this breaks", Stage.ADVERSARIAL, 5, "where this breaks"),
    ("break under pressure", Stage.ADVERSARIAL, 6, "break under pressure"),
    ("how this could fail", Stage.ADVERSARIAL, 6, "how this could fail"),
    ("attack this frame", Stage.ADVERSARIAL, 7, "attack this frame"),
    ("stress points", Stage.ADVERSARIAL, 5, "stress points"),
    ("what would break this frame", Stage.ADVERSARIAL, 6, "what would break this frame"),
    ("what would break it", Stage.ADVERSARIAL, 5, "what would break it"),
    # Exploration
    ("possibility", Stage.EXPLORATION, 3, "possibility"),
    ("brainstorm", Stage.EXPLORATION, 4, "brainstorm"),
    ("explore", Stage.EXPLORATION, 2, "explore"),
    ("alternatives", Stage.EXPLORATION, 3, "alternatives"),
    ("option space", Stage.EXPLORATION, 3, "option space"),
    ("open possibilities", Stage.EXPLORATION, 4, "open possibilities"),
)

NEGATED_CLOSURE_PHRASES: Dict[str, int] = {
    "do not decide": 4,
    "don't decide": 4,
    "not decide yet": 4,
    "do not recommend": 4,
    "don't recommend": 4,
    "do not choose": 4,
    "don't choose": 4,
    "do not make a call": 5,
    "not ready to decide": 4,
}


def extract_routing_features(task: str) -> RoutingFeatures:
    text = task.lower().replace("’", "'")

    # Grouped deterministic pattern families, optimized for task-shape markers.
    expansion_words = ("expand", "expands", "expansion", "broadens", "gets bigger", "widens", "balloons")
    define_words = ("define", "defined", "definition", "specify", "specified", "scope", "frame")
    concrete_words = ("concrete", "specific", "instance", "version", "example", "implementation")
    too_small_words = ("too small", "small", "narrow", "shrinks", "feels tiny", "cramped", "thin slice")
    parts_words = ("fragment", "fragments", "pieces", "parts", "components")
    whole_words = ("whole", "spine", "core", "throughline", "center", "backbone", "organizing logic")
    missing_words = ("missed", "missing", "lost", "not seen", "not grasped", "not holding")
    understood_words = ("understood", "clear", "comprehensible", "makes sense", "legible")

    evidence_words = (
        "evidence",
        "support",
        "verify",
        "unknown",
        "unknowns",
        "unclear",
        "unresolved",
        "proof",
        "confidence",
    )
    uncertainty_words = ("uncertain", "ambigu", "not sure", "missing information", "what is missing")
    uncertainty_characterization_words = (
        "can't tell",
        "can't tell what kind",
        "don't know what kind",
        "hard to characterize",
        "can't characterize",
        "can't identify",
        "can't name it yet",
    )

    decision_words = (
        "decide",
        "deciding",
        "decision",
        "choose",
        "choosing",
        "commit",
        "recommend",
        "recommendation",
        "make a call",
        "what should we do",
        "next move",
        "time pressure",
        "ship now",
        "best option now",
        "now",
        "this week",
        "immediate",
        "select",
        "selecting",
    )
    tradeoff_words = ("tradeoff", "trade-off", "between options", "opportunity cost")

    fragility_words = (
        "fragile",
        "break",
        "stress test",
        "failure mode",
        "failure modes",
        "weakest points",
        "weak spots",
        "strongest objections",
        "vulnerabilities",
        "where this breaks",
        "break under pressure",
        "how this could fail",
        "attack this frame",
        "stress points",
        "risk",
        "destabil",
        "brittle",
    )
    launch_words = ("launch", "production", "deploy", "deployment", "go-live", "trust", "customer-facing")

    recurrence_words_strong = (
        "repeatable",
        "reusable",
        "template",
        "playbook",
        "systematize",
        "systematized",
        "systematizing",
        "standardize",
        "standardized",
    )
    recurrence_words_generic = ("pattern",)
    builder_words = ("productize", "modules", "interfaces", "workflow", "automation")

    possibility_words = (
        "possibility",
        "explore",
        "exploration",
        "brainstorm",
        "alternatives",
        "option space",
        "open",
        "multiple frames",
        "multiple possible frames",
        "multiple perspectives",
        "multiple interpretations",
        "perspectives",
        "interpretations",
        "map the space",
    )
    convergence_words = ("too early", "premature", "locked in", "single frame", "compresses", "narrowing")
    anti_convergence_words = (
        "keep it open",
        "rather than converging",
        "instead of converging",
        "delay convergence",
        "delaying convergence",
        "before narrowing",
    )
    negated_closure_words = (
        "do not decide",
        "don't decide",
        "not decide yet",
        "do not recommend",
        "don't recommend",
        "do not choose",
        "don't choose",
        "do not make a call",
        "not ready to decide",
    )

    matches: Dict[str, List[str]] = {}

    expansion_hits = _contains_any(text, expansion_words)
    define_hits = _contains_any(text, define_words)
    concrete_hits = _contains_any(text, concrete_words)
    too_small_hits = _contains_any(text, too_small_words)
    parts_hits = _contains_any(text, parts_words)
    whole_hits = _contains_any(text, whole_words)
    missing_hits = _contains_any(text, missing_words)
    understood_hits = _contains_any(text, understood_words)
    evidence_hits = _contains_any(text, evidence_words)
    uncertainty_hits = _contains_any(text, uncertainty_words)
    uncertainty_characterization_hits = _contains_any(text, uncertainty_characterization_words)
    decision_hits = _contains_any(text, decision_words)
    tradeoff_hits = _contains_any(text, tradeoff_words)
    fragility_hits = _contains_any(text, fragility_words)
    launch_hits = _contains_any(text, launch_words)
    recurrence_hits_strong = _contains_any(text, recurrence_words_strong)
    recurrence_hits_generic = _contains_any(text, recurrence_words_generic)
    builder_hits = _contains_any(text, builder_words)
    possibility_hits = _contains_any(text, possibility_words)
    convergence_hits = _contains_any(text, convergence_words)
    anti_convergence_hits = _contains_any(text, anti_convergence_words)
    negated_closure_hits = _contains_any(text, negated_closure_words)

    if negated_closure_hits:
        negated_tokens = {"decide", "recommend", "choose", "make a call"}
        decision_hits = [hit for hit in decision_hits if hit not in negated_tokens]
        anti_convergence_hits = sorted(set(anti_convergence_hits + negated_closure_hits))

    structural_signals: List[str] = []

    # expansion-when-defined
    if expansion_hits and define_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED)
        matches["expansion_when_defined"] = sorted(set(expansion_hits + define_hits))

    # concrete-form-too-small / abstraction overflow
    if concrete_hits and too_small_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL)
        matches["concrete_form_too_small"] = sorted(set(concrete_hits + too_small_hits))

    # parts/whole mismatch (legacy-compatible signal name retained)
    if parts_hits and whole_hits and missing_hits and understood_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED)
        matches["parts_whole_mismatch"] = sorted(set(parts_hits + whole_hits + missing_hits + understood_hits))

    if parts_hits and whole_hits and missing_hits:
        matches.setdefault("parts_whole_mismatch", sorted(set(parts_hits + whole_hits + missing_hits)))

    if evidence_hits or uncertainty_hits or uncertainty_characterization_hits:
        matches["uncertainty_evidence_demand"] = sorted(
            set(evidence_hits + uncertainty_hits + uncertainty_characterization_hits)
        )
    if uncertainty_characterization_hits:
        matches["uncertainty_characterization"] = sorted(set(uncertainty_characterization_hits))
    if decision_hits or tradeoff_hits:
        matches["decision_tradeoff_commitment"] = sorted(set(decision_hits + tradeoff_hits))
    if fragility_hits or launch_hits:
        matches["fragility_launch_trust"] = sorted(set(fragility_hits + launch_hits))
    if recurrence_hits_strong or builder_hits:
        matches["recurrence_systemization_strong"] = sorted(set(recurrence_hits_strong + builder_hits))
    if recurrence_hits_generic:
        matches["recurrence_pattern_generic"] = sorted(set(recurrence_hits_generic))
    if recurrence_hits_strong or recurrence_hits_generic or builder_hits:
        matches["recurrence_systemization"] = sorted(
            set(recurrence_hits_strong + recurrence_hits_generic + builder_hits)
        )
    if possibility_hits or convergence_hits or anti_convergence_hits:
        matches["open_possibility_space"] = sorted(set(possibility_hits + convergence_hits + anti_convergence_hits))
    if anti_convergence_hits:
        matches["anti_convergence_preference"] = sorted(set(anti_convergence_hits))
    if negated_closure_hits:
        matches["negated_closure_preference"] = sorted(set(negated_closure_hits))

    return RoutingFeatures(
        structural_signals=structural_signals,
        decision_pressure=_score_from_matches(decision_hits, tradeoff_hits),
        evidence_demand=_score_from_matches(evidence_hits, uncertainty_hits, uncertainty_characterization_hits),
        fragility_pressure=_score_from_matches(fragility_hits, launch_hits),
        recurrence_potential=min(10, (2 * len(recurrence_hits_strong)) + (2 * len(builder_hits))),
        possibility_space_need=_score_from_matches(possibility_hits, convergence_hits, anti_convergence_hits),
        detected_markers=matches,
    )


def extract_structural_signals(task: str) -> List[str]:
    return extract_routing_features(task).structural_signals


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    inferred = set(risk_profile or set())
    text = task.lower()
    features = extract_routing_features(task)
    signals = set(features.structural_signals)

    if signals:
        inferred.add("abstract_structural_task")
    if (
        STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in signals
        and any(
        k in text for k in ("single frame", "one frame", "unif", "compress", "organizing idea")
        )
    ):
        inferred.add("false_unification")
    if features.fragility_pressure >= 2:
        inferred.add("fragility_pressure")
    if features.evidence_demand >= 2:
        inferred.add("evidence_gap")
    if features.decision_pressure >= 2:
        inferred.add("decision_urgency")

    return inferred


# ============================================================
# Core enums
class RegimeConfidenceCalculator:
    NONTRIVIAL_SCORE_FLOOR = 2
    EXPLICIT_DECISION_MARKERS = frozenset(
        {
            "decide",
            "deciding",
            "decision",
            "choose",
            "choosing",
            "choose now",
            "recommend",
            "recommendation",
            "make a call",
            "what should we do",
            "now",
            "this week",
            "immediate",
            "commit",
            "next move",
            "best option now",
            "select",
            "selecting",
            "select now",
            "select one",
            "tradeoff",
            "trade-off",
            "between options",
            "selection",
            "opportunity cost",
        }
    )

    def calculate(
        self,
        *,
        top_stage_score: int,
        runner_up_score: int,
        lexical_scores: Dict[Stage, int],
        structural_scores: Dict[Stage, int],
        features: RoutingFeatures,
    ) -> RegimeConfidenceResult:
        score_gap = max(0, top_stage_score - runner_up_score)
        nontrivial_stage_count = sum(
            1
            for stage in Stage
            if (lexical_scores.get(stage, 0) + structural_scores.get(stage, 0)) >= self.NONTRIVIAL_SCORE_FLOOR
        )

        total_lexical = sum(lexical_scores.values())
        total_structural = sum(structural_scores.values())
        weak_lexical_dependence = total_lexical > 0 and total_structural <= 1 and total_lexical >= total_structural * 3

        structural_sparse = len(features.structural_signals) == 0 and "parts_whole_mismatch" not in features.detected_markers
        decision_tradeoff_markers = set(features.detected_markers.get("decision_tradeoff_commitment", []))
        has_explicit_decision_marker = bool(decision_tradeoff_markers & self.EXPLICIT_DECISION_MARKERS)
        mixed_decision_and_possibility = features.decision_pressure >= 2 and features.possibility_space_need >= 2
        explicit_decision_priority = (
            mixed_decision_and_possibility
            and has_explicit_decision_marker
            and features.decision_pressure >= features.possibility_space_need
        )
        structural_conflicting = (
            (mixed_decision_and_possibility and not explicit_decision_priority)
            or (features.fragility_pressure >= 2 and features.possibility_space_need >= 2)
        )
        structural_state = "conflicting" if structural_conflicting else ("sparse" if structural_sparse else "coherent")
        pure_open_space_signal = (
            features.possibility_space_need >= 1
            and features.decision_pressure == 0
            and features.evidence_demand == 0
            and features.fragility_pressure == 0
        )
        clean_single_regime_lexical = (
            pure_open_space_signal
            and nontrivial_stage_count == 1
            and top_stage_score >= 5
            and score_gap >= 4
            and runner_up_score <= 1
        )

        if (
            top_stage_score >= 8
            and score_gap >= 4
            and nontrivial_stage_count <= 2
            and not weak_lexical_dependence
            and not structural_conflicting
        ):
            return RegimeConfidenceResult(
                level=Severity.HIGH.value,
                rationale="Primary regime wins by a clear margin with concentrated, coherent signals.",
                top_stage_score=top_stage_score,
                runner_up_score=runner_up_score,
                score_gap=score_gap,
                nontrivial_stage_count=nontrivial_stage_count,
                weak_lexical_dependence=weak_lexical_dependence,
                structural_feature_state=structural_state,
            )

        if clean_single_regime_lexical and not structural_conflicting:
            return RegimeConfidenceResult(
                level=Severity.HIGH.value,
                rationale=(
                    "Single-regime intent is explicit and dominant; sparse structure is acceptable here because "
                    "there are no competing stage pressures."
                ),
                top_stage_score=top_stage_score,
                runner_up_score=runner_up_score,
                score_gap=score_gap,
                nontrivial_stage_count=nontrivial_stage_count,
                weak_lexical_dependence=weak_lexical_dependence,
                structural_feature_state=structural_state,
            )

        if (
            top_stage_score >= 4
            and score_gap >= 1
            and not structural_conflicting
        ):
            medium_rationale = "Two plausible regimes are relatively close; sequencing is likely."
            if score_gap >= 3 and nontrivial_stage_count <= 2:
                medium_rationale = (
                    "Primary regime leads, but structural support is limited or secondary signals remain plausible."
                )
            return RegimeConfidenceResult(
                level=Severity.MEDIUM.value,
                rationale=medium_rationale,
                top_stage_score=top_stage_score,
                runner_up_score=runner_up_score,
                score_gap=score_gap,
                nontrivial_stage_count=nontrivial_stage_count,
                weak_lexical_dependence=weak_lexical_dependence,
                structural_feature_state=structural_state,
            )

        low_reason = "Evidence is weak or misframed for reliable single-regime routing."
        if structural_conflicting:
            low_reason = "Signals conflict across stages, suggesting likely misframing."
        elif weak_lexical_dependence and structural_sparse and nontrivial_stage_count <= 1:
            low_reason = "Routing is mostly weak lexical cue matching with sparse structural evidence."
        return RegimeConfidenceResult(
            level=Severity.LOW.value,
            rationale=low_reason,
            top_stage_score=top_stage_score,
            runner_up_score=runner_up_score,
            score_gap=score_gap,
            nontrivial_stage_count=nontrivial_stage_count,
            weak_lexical_dependence=weak_lexical_dependence,
            structural_feature_state=structural_state,
        )

    @staticmethod
    def high_shortcut_rationale(reason: str) -> RegimeConfidenceResult:
        return RegimeConfidenceResult(
            level=Severity.HIGH.value,
            rationale=reason,
            top_stage_score=10,
            runner_up_score=3,
            score_gap=7,
            nontrivial_stage_count=2,
            weak_lexical_dependence=False,
            structural_feature_state="coherent",
        )


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
        return ", ".join(f"{stage.value}:{stage_scores.get(stage, 0)}" for stage in Stage)

    @staticmethod
    def _format_stage_contributions(contributions: Dict[Stage, List[str]]) -> str:
        parts: List[str] = []
        for stage in Stage:
            items = contributions.get(stage, [])
            if items:
                parts.append(f"{stage.value}=>[{'; '.join(items)}]")
        return " | ".join(parts) if parts else "none"

    @staticmethod
    def _reason_for(stage: Stage) -> Tuple[str, str]:
        reasons = {
            Stage.OPERATOR: (
                "Decision-intent language dominates; the immediate need is commitment and explicit tradeoffs.",
                "Switch when the decision, tradeoff, and fallback trigger are explicit.",
            ),
            Stage.EPISTEMIC: (
                "Uncertainty/evidence language dominates; the current risk is claims outrunning support.",
                "Switch when supported vs unsupported claims are separated and the next decision becomes clear.",
            ),
            Stage.SYNTHESIS: (
                "Structural-compression signals dominate; the work needs an organizing spine before action.",
                "Switch when a dominant frame emerges that can guide exclusion or action.",
            ),
            Stage.EXPLORATION: (
                "The bottleneck indicates missing possibility space more than proof or commitment.",
                "Switch when 2-5 distinct frames exist and one begins to dominate.",
            ),
            Stage.BUILDER: (
                "The pattern should be turned into durable structure.",
                "Switch when modules, recurrence, and implementation order are clear.",
            ),
            Stage.ADVERSARIAL: (
                "The bottleneck is hidden fragility, not idea generation.",
                "Switch when the top destabilizer is identified and revisions are clear.",
            ),
        }
        return reasons[stage]

    def should_use_analyzer(self, confidence: RegimeConfidenceResult, score_gap_threshold: int = 1) -> bool:
        if confidence.level == Severity.HIGH.value:
            return False
        if confidence.level == Severity.MEDIUM.value:
            return confidence.score_gap <= score_gap_threshold
        return True

    @staticmethod
    def _analyzer_scores_flat_or_nearly_flat(analyzer_ranked: List[Tuple[Stage, float]]) -> bool:
        if len(analyzer_ranked) < 2:
            return True
        values = [score for _, score in analyzer_ranked]
        spread = max(values) - min(values)
        top_gap = values[0] - values[1]
        return spread <= 0.18 or top_gap <= 0.08

    @staticmethod
    def _rationale_too_short_or_generic(rationale: str) -> bool:
        cleaned = (rationale or "").strip().lower()
        if not cleaned:
            return True
        if len(cleaned) < 32 or len(cleaned.split()) < 5:
            return True
        generic_markers = (
            "based on the prompt",
            "best fit",
            "most suitable",
            "general",
            "broad",
            "overall",
            "appears to",
            "seems to",
            "likely",
        )
        generic_hits = sum(1 for marker in generic_markers if marker in cleaned)
        return generic_hits >= 2

    def _accept_analyzer_override(
        self,
        *,
        analyzer_result: TaskAnalyzerOutput,
        analyzer_ranked: List[Tuple[Stage, float]],
        features: RoutingFeatures,
        zero_score_fallback: bool,
    ) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        candidate_count = len(analyzer_result.candidate_regimes or [])
        if candidate_count > 3:
            reasons.append(f"candidate_regimes too broad ({candidate_count} > 3)")

        if self._analyzer_scores_flat_or_nearly_flat(analyzer_ranked):
            reasons.append("analyzer stage scores are flat/nearly flat")

        top_score = analyzer_ranked[0][1] if analyzer_ranked else 0.0
        runner_score = analyzer_ranked[1][1] if len(analyzer_ranked) > 1 else 0.0
        if top_score - runner_score < 0.15:
            reasons.append(
                f"top analyzer score is not meaningfully above runner-up ({top_score:.2f} vs {runner_score:.2f})"
            )

        if self._rationale_too_short_or_generic(analyzer_result.rationale):
            reasons.append("analyzer rationale too short/generic")

        proposed_primary = analyzer_ranked[0][0] if analyzer_ranked else Stage.EXPLORATION
        recurrence_evidence = (
            features.recurrence_potential > 0
            or "recurrence_systemization_strong" in features.detected_markers
            or len(analyzer_result.structural_signals) > 0
        )
        decision_evidence = (
            features.decision_pressure > 0
            or "decision_tradeoff_commitment" in features.detected_markers
            or analyzer_result.decision_pressure > 0
        )
        fragility_evidence = (
            features.fragility_pressure > 0
            or "fragility_launch_trust" in features.detected_markers
        )

        if proposed_primary == Stage.BUILDER and not recurrence_evidence:
            reasons.append("builder proposed without recurrence/systemization evidence")
        if proposed_primary == Stage.OPERATOR and not decision_evidence:
            reasons.append("operator proposed without decision evidence")
        if proposed_primary == Stage.ADVERSARIAL and not fragility_evidence:
            reasons.append("adversarial proposed without fragility/failure evidence")

        if zero_score_fallback and len(analyzer_result.candidate_regimes or []) == len(Stage):
            reasons.append("zero-score fallback: analyzer proposed all regimes")

        return (len(reasons) == 0, reasons)

    def _apply_shortcut_routes(
        self,
        b: str,
        features: RoutingFeatures,
        analyzer_enabled: bool,
    ) -> Optional[RoutingDecision]:
        normalized_b = b.lower().replace("’", "'").strip()
        interpretation_shortcut_markers = ["strongest interpretation", "strongest frame", "what this actually is"]
        epistemic_markers = ["evidence", "support", "verify", "unknown", "unknowns", "unclear", "unresolved"]

        if any(_has_phrase(normalized_b, k) for k in interpretation_shortcut_markers) and not any(
            _has_phrase(normalized_b, k) for k in epistemic_markers
        ):
            return RoutingDecision(
                bottleneck=b,
                primary_regime=Stage.SYNTHESIS,
                runner_up_regime=Stage.ADVERSARIAL,
                why_primary_wins_now="The task asks for interpretation-level compression first, then pressure-testing against break conditions.",
                switch_trigger="Switch when the strongest frame is identified and the next bottleneck becomes exposing how it fails under stress.",
                confidence=RegimeConfidenceCalculator.high_shortcut_rationale(
                    "Explicit interpretation-first shortcut with deterministic routing precedence."
                ),
                deterministic_stage_scores={Stage.SYNTHESIS: 10, Stage.ADVERSARIAL: 3},
                deterministic_score_summary="shortcut: interpretation-first precedence",
                deterministic_score_contributions={Stage.SYNTHESIS: ["shortcut:interpretation_first"], Stage.ADVERSARIAL: ["shortcut:stress_test_followup"]},
                analyzer_enabled=analyzer_enabled,
            )

        adversarial_shortcut_markers = [
            "stress test this frame",
            "stress test",
            "break it",
            "too clean",
            "fragile",
            "launch",
            "weakest points",
            "weak spots",
            "strongest objections",
            "vulnerabilities",
            "failure modes",
            "where this breaks",
            "break under pressure",
            "how this could fail",
            "attack this frame",
            "stress points",
            "what would break this frame",
        ]

        if any(_has_phrase(normalized_b, k) for k in adversarial_shortcut_markers):
            return RoutingDecision(
                bottleneck=b,
                primary_regime=Stage.ADVERSARIAL,
                runner_up_regime=Stage.EPISTEMIC,
                why_primary_wins_now="The bottleneck is hidden fragility, not idea generation.",
                switch_trigger="Switch when the top destabilizer is identified and revisions are clear.",
                confidence=RegimeConfidenceCalculator.high_shortcut_rationale(
                    "Explicit stress-test/failure language makes the adversarial bottleneck clear."
                ),
                deterministic_stage_scores={Stage.ADVERSARIAL: 10, Stage.EPISTEMIC: 3},
                deterministic_score_summary="shortcut: stress-test precedence",
                deterministic_score_contributions={Stage.ADVERSARIAL: ["shortcut:stress_test_precedence"], Stage.EPISTEMIC: ["shortcut:evidence_followup"]},
                analyzer_enabled=analyzer_enabled,
            )

        if any(
            _has_phrase(normalized_b, k)
            for k in [
                "repeatable",
                "reusable",
                "template",
                "playbook",
                "systematize",
                "standardize",
                "modules",
                "workflow",
                "automation",
                "productize",
            ]
        ):
            return RoutingDecision(
                bottleneck=b,
                primary_regime=Stage.BUILDER,
                runner_up_regime=Stage.OPERATOR,
                why_primary_wins_now="The pattern should be turned into durable structure.",
                switch_trigger="Switch when modules, recurrence, and implementation order are clear.",
                confidence=RegimeConfidenceCalculator.high_shortcut_rationale(
                    "Explicit repeatability/productization language deterministically points to builder mode."
                ),
                deterministic_stage_scores={Stage.BUILDER: 10, Stage.OPERATOR: 3},
                deterministic_score_summary="shortcut: repeatability precedence",
                deterministic_score_contributions={Stage.BUILDER: ["shortcut:repeatability_precedence"], Stage.OPERATOR: ["shortcut:execution_followup"]},
                analyzer_enabled=analyzer_enabled,
            )

        return None

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
        ranked = sorted(stage_scores.items(), key=lambda x: (-x[1], self.precedence_order.index(x[0])))
        top_stage, top_score = ranked[0]

        if top_score <= 0:
            decision = RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="No specific regime has enough signal; exploration is the safest fallback.",
                switch_trigger="Switch when one frame becomes more decision-relevant than the others.",
                confidence=RegimeConfidenceResult(
                    level=Severity.LOW.value,
                    rationale="No stage has nontrivial score; routing is intentionally conservative.",
                    top_stage_score=top_score,
                    runner_up_score=0,
                    score_gap=0,
                    nontrivial_stage_count=0,
                    weak_lexical_dependence=True,
                    structural_feature_state="sparse",
                ),
                deterministic_stage_scores=stage_scores,
                deterministic_score_summary=self._format_stage_score_summary(stage_scores),
                deterministic_score_contributions=stage_contributions,
                analyzer_enabled=analyzer_enabled,
            )
            if not (analyzer_enabled and analyzer_result and self.should_use_analyzer(decision.confidence, score_gap_threshold=analyzer_gap_threshold)):
                return decision
            if analyzer_result.confidence < 0.6:
                return replace(
                    decision,
                    analyzer_used=True,
                    analyzer_summary=(
                        f"Analyzer output ignored due to low analyzer confidence ({analyzer_result.confidence:.2f})."
                    ),
                )
            analyzer_ranked = sorted(
                analyzer_result.stage_scores.items(),
                key=lambda x: (-x[1], self.precedence_order.index(x[0])),
            )
            accepted, rejection_reasons = self._accept_analyzer_override(
                analyzer_result=analyzer_result,
                analyzer_ranked=analyzer_ranked,
                features=features,
                zero_score_fallback=True,
            )
            if not accepted:
                return replace(
                    decision,
                    analyzer_used=True,
                    analyzer_summary=(
                        "Analyzer output ignored in zero-score fallback: " + "; ".join(rejection_reasons)
                    ),
                )
            analyzer_primary = analyzer_ranked[0][0] if analyzer_ranked else decision.primary_regime
            candidate_set = set(analyzer_result.candidate_regimes or [])
            if candidate_set and analyzer_primary not in candidate_set:
                analyzer_primary = decision.primary_regime
            new_primary = analyzer_primary if analyzer_primary != Stage.EXPLORATION else decision.primary_regime
            new_runner = next((stage for stage, _ in analyzer_ranked if stage != new_primary), decision.runner_up_regime)
            why, switch = self._reason_for(new_primary)
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=new_primary,
                runner_up_regime=new_runner,
                why_primary_wins_now=why,
                switch_trigger=switch,
                confidence=decision.confidence,
                deterministic_stage_scores=stage_scores,
                deterministic_score_summary=self._format_stage_score_summary(stage_scores),
                deterministic_score_contributions=stage_contributions,
                analyzer_enabled=analyzer_enabled,
                analyzer_used=True,
                analyzer_changed_primary=new_primary != decision.primary_regime,
                analyzer_changed_runner_up=new_runner != decision.runner_up_regime,
                analyzer_summary=(
                    f"Analyzer confidence={analyzer_result.confidence:.2f}; rationale={analyzer_result.rationale}; "
                    f"candidates={[stage.value for stage in analyzer_result.candidate_regimes]}"
                ),
            )

        runner_up = next((stage for stage, score in ranked[1:] if score > 0), Stage.EXPLORATION if top_stage != Stage.EXPLORATION else Stage.SYNTHESIS)
        runner_up_score = next((score for stage, score in ranked[1:] if stage == runner_up), 0)
        why_primary_wins_now, switch_trigger = self._reason_for(top_stage)
        confidence = deterministic_confidence or self.confidence_calculator.calculate(
            top_stage_score=top_score,
            runner_up_score=runner_up_score,
            lexical_scores=lexical_scores,
            structural_scores=structural_scores,
            features=features,
        )
        decision = RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=top_stage,
            runner_up_regime=runner_up,
            why_primary_wins_now=why_primary_wins_now,
            switch_trigger=switch_trigger,
            confidence=confidence,
            deterministic_stage_scores=stage_scores,
            deterministic_score_summary=self._format_stage_score_summary(stage_scores),
            deterministic_score_contributions=stage_contributions,
            analyzer_enabled=analyzer_enabled,
        )
        if not (analyzer_enabled and analyzer_result):
            return decision
        if not self.should_use_analyzer(confidence, score_gap_threshold=analyzer_gap_threshold):
            return decision
        if analyzer_result.confidence < 0.6:
            return replace(
                decision,
                analyzer_used=True,
                analyzer_summary=(
                    f"Analyzer output ignored due to low analyzer confidence ({analyzer_result.confidence:.2f})."
                ),
            )

        analyzer_ranked = sorted(
            analyzer_result.stage_scores.items(),
            key=lambda x: (-x[1], self.precedence_order.index(x[0])),
        )
        accepted, rejection_reasons = self._accept_analyzer_override(
            analyzer_result=analyzer_result,
            analyzer_ranked=analyzer_ranked,
            features=features,
            zero_score_fallback=False,
        )
        if not accepted:
            return replace(
                decision,
                analyzer_used=True,
                analyzer_summary="Analyzer output ignored: " + "; ".join(rejection_reasons),
            )

        analyzer_primary = analyzer_ranked[0][0] if analyzer_ranked else decision.primary_regime
        candidate_set = set(analyzer_result.candidate_regimes or [])
        if candidate_set and analyzer_primary not in candidate_set:
            analyzer_primary = decision.primary_regime

        allowed_primary_candidates = {decision.primary_regime, decision.runner_up_regime}
        should_shift_primary = (
            confidence.level == Severity.LOW.value
            and confidence.score_gap <= analyzer_gap_threshold
            and analyzer_primary in allowed_primary_candidates
            and analyzer_primary != decision.primary_regime
        )
        new_primary = analyzer_primary if should_shift_primary else decision.primary_regime
        default_runner = decision.runner_up_regime or (
            Stage.EXPLORATION if new_primary != Stage.EXPLORATION else Stage.SYNTHESIS
        )
        analyzer_runner = next((stage for stage, _ in analyzer_ranked if stage != new_primary), default_runner)
        if candidate_set and analyzer_runner not in candidate_set and default_runner:
            analyzer_runner = default_runner
        new_runner = analyzer_runner if analyzer_runner else default_runner
        if new_runner == new_primary:
            new_runner = default_runner
        why, switch = self._reason_for(new_primary)
        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=new_primary,
            runner_up_regime=new_runner,
            why_primary_wins_now=why,
            switch_trigger=switch,
            confidence=confidence,
            deterministic_stage_scores=stage_scores,
            deterministic_score_summary=self._format_stage_score_summary(stage_scores),
            deterministic_score_contributions=stage_contributions,
            analyzer_enabled=analyzer_enabled,
            analyzer_used=True,
            analyzer_changed_primary=new_primary != decision.primary_regime,
            analyzer_changed_runner_up=new_runner != decision.runner_up_regime,
            analyzer_summary=(
                f"Analyzer confidence={analyzer_result.confidence:.2f}; rationale={analyzer_result.rationale}; "
                f"candidates={[stage.value for stage in analyzer_result.candidate_regimes]}"
            ),
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

        stage_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        lexical_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        structural_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        stage_contributions: Dict[Stage, List[str]] = {stage: [] for stage in Stage}

        def add_score(stage: Stage, amount: int, bucket: str, reason: str) -> None:
            if amount <= 0:
                return
            stage_scores[stage] += amount
            if bucket == "lexical":
                lexical_scores[stage] += amount
            else:
                structural_scores[stage] += amount
            stage_contributions[stage].append(f"+{amount} {bucket}:{reason}")

        def suppress_score(stage: Stage, amount: int, bucket: str, reason: str) -> None:
            if amount <= 0:
                return
            stage_scores[stage] = max(0, stage_scores[stage] - amount)
            if bucket == "lexical":
                lexical_scores[stage] = max(0, lexical_scores[stage] - amount)
            else:
                structural_scores[stage] = max(0, structural_scores[stage] - amount)
            stage_contributions[stage].append(f"-{amount} {bucket}:{reason}")

        self._apply_lexical_scores(
            b,
            features,
            stage_scores,
            lexical_scores,
            stage_contributions,
            add_score,
            suppress_score,
        )
        self._apply_structural_scores(
            b,
            features,
            risks,
            signals,
            stage_scores,
            structural_scores,
            stage_contributions,
            add_score,
            suppress_score,
            escalation_policy_result=escalation_policy_result,
        )
        self._apply_embedding_scores(
            bottleneck,
            stage_scores,
            stage_contributions,
            add_score,
        )

        if deterministic_stage_scores:
            for stage in Stage:
                if stage in deterministic_stage_scores:
                    override_value = int(deterministic_stage_scores[stage])
                    if override_value != stage_scores[stage]:
                        stage_contributions[stage].append(f"override:external_deterministic_score={override_value}")
                    stage_scores[stage] = override_value

        return self._build_final_decision(
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
