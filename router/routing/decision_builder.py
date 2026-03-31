from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

from router.models import RegimeConfidenceResult, RoutingDecision, RoutingFeatures, Severity, Stage, TaskAnalyzerOutput


def format_stage_score_summary(stage_scores: Dict[Stage, int]) -> str:
    return ", ".join(f"{stage.value}:{stage_scores.get(stage, 0)}" for stage in Stage)


def format_stage_contributions(contributions: Dict[Stage, List[str]]) -> str:
    parts: List[str] = []
    for stage in Stage:
        items = contributions.get(stage, [])
        if items:
            parts.append(f"{stage.value}=>[{'; '.join(items)}]")
    return " | ".join(parts) if parts else "none"


def reason_for(stage: Stage) -> Tuple[str, str]:
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


def apply_shortcut_routes(
    *,
    bottleneck: str,
    features: RoutingFeatures,
    analyzer_enabled: bool,
    has_phrase: Callable[[str, str], bool],
    high_shortcut_rationale: Callable[[str], RegimeConfidenceResult],
) -> Optional[RoutingDecision]:
    normalized_b = bottleneck.lower().replace("’", "'").strip()
    interpretation_shortcut_markers = ["strongest interpretation", "strongest frame", "what this actually is"]
    epistemic_markers = ["evidence", "support", "verify", "unknown", "unknowns", "unclear", "unresolved"]

    if any(has_phrase(normalized_b, k) for k in interpretation_shortcut_markers) and not any(
        has_phrase(normalized_b, k) for k in epistemic_markers
    ):
        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=Stage.SYNTHESIS,
            runner_up_regime=Stage.ADVERSARIAL,
            why_primary_wins_now="The task asks for interpretation-level compression first, then pressure-testing against break conditions.",
            switch_trigger="Switch when the strongest frame is identified and the next bottleneck becomes exposing how it fails under stress.",
            confidence=high_shortcut_rationale(
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

    if any(has_phrase(normalized_b, k) for k in adversarial_shortcut_markers):
        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=Stage.ADVERSARIAL,
            runner_up_regime=Stage.EPISTEMIC,
            why_primary_wins_now="The bottleneck is hidden fragility, not idea generation.",
            switch_trigger="Switch when the top destabilizer is identified and revisions are clear.",
            confidence=high_shortcut_rationale(
                "Explicit stress-test/failure language makes the adversarial bottleneck clear."
            ),
            deterministic_stage_scores={Stage.ADVERSARIAL: 10, Stage.EPISTEMIC: 3},
            deterministic_score_summary="shortcut: stress-test precedence",
            deterministic_score_contributions={Stage.ADVERSARIAL: ["shortcut:stress_test_precedence"], Stage.EPISTEMIC: ["shortcut:evidence_followup"]},
            analyzer_enabled=analyzer_enabled,
        )

    if any(
        has_phrase(normalized_b, k)
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
            bottleneck=bottleneck,
            primary_regime=Stage.BUILDER,
            runner_up_regime=Stage.OPERATOR,
            why_primary_wins_now="The pattern should be turned into durable structure.",
            switch_trigger="Switch when modules, recurrence, and implementation order are clear.",
            confidence=high_shortcut_rationale(
                "Explicit repeatability/productization language deterministically points to builder mode."
            ),
            deterministic_stage_scores={Stage.BUILDER: 10, Stage.OPERATOR: 3},
            deterministic_score_summary="shortcut: repeatability precedence",
            deterministic_score_contributions={Stage.BUILDER: ["shortcut:repeatability_precedence"], Stage.OPERATOR: ["shortcut:execution_followup"]},
            analyzer_enabled=analyzer_enabled,
        )

    return None


def build_final_decision(
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
    precedence_order: List[Stage],
    confidence_calculator,
    should_use_analyzer_fn: Callable[[RegimeConfidenceResult, int], bool],
    accept_analyzer_override_fn: Callable[..., Tuple[bool, List[str]]],
) -> RoutingDecision:
    ranked = sorted(stage_scores.items(), key=lambda x: (-x[1], precedence_order.index(x[0])))
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
            deterministic_score_summary=format_stage_score_summary(stage_scores),
            deterministic_score_contributions=stage_contributions,
            analyzer_enabled=analyzer_enabled,
        )
        if not (analyzer_enabled and analyzer_result and should_use_analyzer_fn(decision.confidence, analyzer_gap_threshold)):
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
            key=lambda x: (-x[1], precedence_order.index(x[0])),
        )
        accepted, rejection_reasons = accept_analyzer_override_fn(
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
        why, switch = reason_for(new_primary)
        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=new_primary,
            runner_up_regime=new_runner,
            why_primary_wins_now=why,
            switch_trigger=switch,
            confidence=decision.confidence,
            deterministic_stage_scores=stage_scores,
            deterministic_score_summary=format_stage_score_summary(stage_scores),
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
    why_primary_wins_now, switch_trigger = reason_for(top_stage)
    confidence = deterministic_confidence or confidence_calculator.calculate(
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
        deterministic_score_summary=format_stage_score_summary(stage_scores),
        deterministic_score_contributions=stage_contributions,
        analyzer_enabled=analyzer_enabled,
    )
    if not (analyzer_enabled and analyzer_result):
        return decision
    if not should_use_analyzer_fn(confidence, analyzer_gap_threshold):
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
        key=lambda x: (-x[1], precedence_order.index(x[0])),
    )
    accepted, rejection_reasons = accept_analyzer_override_fn(
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
    why, switch = reason_for(new_primary)
    return RoutingDecision(
        bottleneck=bottleneck,
        primary_regime=new_primary,
        runner_up_regime=new_runner,
        why_primary_wins_now=why,
        switch_trigger=switch,
        confidence=confidence,
        deterministic_stage_scores=stage_scores,
        deterministic_score_summary=format_stage_score_summary(stage_scores),
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
