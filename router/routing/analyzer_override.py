from __future__ import annotations

from typing import List, Tuple

from router.models import RegimeConfidenceResult, RoutingFeatures, Severity, Stage, TaskAnalyzerOutput


def should_use_analyzer(confidence: RegimeConfidenceResult, score_gap_threshold: int = 1) -> bool:
    if confidence.level == Severity.HIGH.value:
        return False
    if confidence.level == Severity.MEDIUM.value:
        return confidence.score_gap <= score_gap_threshold
    return True


def analyzer_scores_flat_or_nearly_flat(analyzer_ranked: List[Tuple[Stage, float]]) -> bool:
    if len(analyzer_ranked) < 2:
        return True
    values = [score for _, score in analyzer_ranked]
    spread = max(values) - min(values)
    top_gap = values[0] - values[1]
    return spread <= 0.18 or top_gap <= 0.08


def rationale_too_short_or_generic(rationale: str) -> bool:
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


def accept_analyzer_override(
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

    if analyzer_scores_flat_or_nearly_flat(analyzer_ranked):
        reasons.append("analyzer stage scores are flat/nearly flat")

    top_score = analyzer_ranked[0][1] if analyzer_ranked else 0.0
    runner_score = analyzer_ranked[1][1] if len(analyzer_ranked) > 1 else 0.0
    if top_score - runner_score < 0.15:
        reasons.append(
            f"top analyzer score is not meaningfully above runner-up ({top_score:.2f} vs {runner_score:.2f})"
        )

    if rationale_too_short_or_generic(analyzer_result.rationale):
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
