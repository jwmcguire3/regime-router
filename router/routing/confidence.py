from __future__ import annotations

from typing import Dict

from router.models import RegimeConfidenceResult, RoutingFeatures, Severity, Stage


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
