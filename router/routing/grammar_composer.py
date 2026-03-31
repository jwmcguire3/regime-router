from __future__ import annotations

from typing import List, Optional, Set, Tuple
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from router.models import (
    CANONICAL_FAILURE_IF_OVERUSED,
    FAILURE_SUPPRESSOR_MAP,
    LIBRARY,
    FunctionType,
    LinePrimitive,
    Regime,
    Stage,
)

_FAILURE_SELECTION_SPEC = spec_from_file_location(
    "router.routing.failure_selection",
    Path(__file__).with_name("failure_selection.py"),
)
if _FAILURE_SELECTION_SPEC is None or _FAILURE_SELECTION_SPEC.loader is None:
    raise ImportError("Unable to load failure_selection.py")
_failure_selection = module_from_spec(_FAILURE_SELECTION_SPEC)
_FAILURE_SELECTION_SPEC.loader.exec_module(_failure_selection)

_GRAMMAR_RULES_SPEC = spec_from_file_location(
    "router.routing.grammar_rules",
    Path(__file__).with_name("grammar_rules.py"),
)
if _GRAMMAR_RULES_SPEC is None or _GRAMMAR_RULES_SPEC.loader is None:
    raise ImportError("Unable to load grammar_rules.py")
_grammar_rules = module_from_spec(_GRAMMAR_RULES_SPEC)
_GRAMMAR_RULES_SPEC.loader.exec_module(_grammar_rules)

rank_failures_by_cost = _failure_selection.rank_failures_by_cost
select_dominant = _failure_selection.select_dominant
select_shapes = _failure_selection.select_shapes
select_suppressions = _failure_selection.select_suppressions
select_tail = _failure_selection.select_tail

deduplicate_lines = _grammar_rules.deduplicate_lines
has_hard_conflict = _grammar_rules.has_hard_conflict
validate_regime_grammar = _grammar_rules.validate_regime_grammar


class GrammarComposer:
    def compose(
        self,
        stage: Stage,
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = False,
    ) -> Regime:
        effective_stage = stage or Stage.EXPLORATION
        risks = set(risk_profile or set())

        dominant = select_dominant(effective_stage, risks)
        ranked_failures = rank_failures_by_cost(dominant, risks)
        suppressions = select_suppressions(dominant, ranked_failures, risks)
        suppressions = self._apply_synthesis_break_condition_pressure(
            stage=effective_stage,
            dominant=dominant,
            suppressions=suppressions,
            risk_profile=risks,
        )
        shapes = select_shapes(effective_stage, dominant, suppressions, risks)
        tail = select_tail(effective_stage, handoff_expected, risks)

        candidate_lines = [dominant] + suppressions + shapes + ([tail] if tail else [])

        filtered_lines, rejected_lines, rejection_reasons = self._remove_hard_conflicts(candidate_lines)
        deduped_lines, dedupe_rejected, dedupe_reasons = self._dedupe_with_reasons(filtered_lines)
        deduped_lines, tail_rejected, tail_reasons = self._ensure_tail_slot(deduped_lines, tail)
        rejected_lines.extend(dedupe_rejected)
        rejected_lines.extend(tail_rejected)
        rejection_reasons.extend(dedupe_reasons)
        rejection_reasons.extend(tail_reasons)

        is_valid, violations = validate_regime_grammar(deduped_lines)
        if not is_valid:
            rejection_reasons.extend(f"Grammar validation failed: {violation}" for violation in violations)
            deduped_lines, fallback_rejected, fallback_reasons = self._fallback_minimal_regime(
                dominant=dominant,
                ranked_failures=ranked_failures,
                selected_suppressions=suppressions,
                prior_lines=deduped_lines,
            )
            rejected_lines.extend(fallback_rejected)
            rejection_reasons.extend(fallback_reasons)

        return self._build_regime(
            stage=effective_stage,
            lines=deduped_lines,
            rejected_lines=rejected_lines,
            rejection_reasons=rejection_reasons,
        )

    def _apply_synthesis_break_condition_pressure(
        self,
        *,
        stage: Stage,
        dominant: LinePrimitive,
        suppressions: List[LinePrimitive],
        risk_profile: Set[str],
    ) -> List[LinePrimitive]:
        if stage != Stage.SYNTHESIS or not self._requires_synthesis_break_condition_pressure(risk_profile):
            return suppressions

        selected = list(suppressions)
        selected_ids = {line.id for line in selected}
        candidate = LIBRARY["SYN-P2"]
        if candidate.id in selected_ids:
            return selected
        if has_hard_conflict(dominant, candidate):
            return selected
        if any(has_hard_conflict(existing, candidate) for existing in selected):
            return selected
        selected.append(candidate)
        return selected[:2]

    @staticmethod
    def _requires_synthesis_break_condition_pressure(risk_profile: Set[str]) -> bool:
        high_risk_synthesis_conditions = {
            "coherence_over_truth",
            "false_unification",
            "high_stakes",
            "abstract_structural_task",
        }
        return bool(high_risk_synthesis_conditions & risk_profile)

    def _remove_hard_conflicts(self, lines: List[LinePrimitive]) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        kept: List[LinePrimitive] = []
        rejected: List[str] = []
        reasons: List[str] = []

        for candidate in lines:
            conflicting = next((existing for existing in kept if has_hard_conflict(existing, candidate)), None)
            if conflicting is not None:
                rejected.append(candidate.id)
                reasons.append(f"Rejected {candidate.id}: hard conflict with {conflicting.id}.")
                continue
            kept.append(candidate)

        return kept, rejected, reasons

    def _dedupe_with_reasons(self, lines: List[LinePrimitive]) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        deduped = deduplicate_lines(lines)
        deduped_ids = {line.id for line in deduped}

        rejected: List[str] = []
        reasons: List[str] = []

        seen_ids: Set[str] = set()
        for line in lines:
            if line.id in seen_ids:
                rejected.append(line.id)
                reasons.append(f"Rejected {line.id}: duplicate line id.")
                continue
            seen_ids.add(line.id)
            if line.id not in deduped_ids:
                rejected.append(line.id)
                reasons.append(f"Rejected {line.id}: removed during dedupe/line cap enforcement.")

        return deduped, rejected, reasons

    def _ensure_tail_slot(
        self,
        lines: List[LinePrimitive],
        tail: Optional[LinePrimitive],
    ) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        if tail is None or any(line.id == tail.id for line in lines):
            return lines, [], []
        if any(has_hard_conflict(line, tail) for line in lines):
            return lines, [tail.id], [f"Rejected {tail.id}: hard conflict prevented tail insertion."]

        updated = list(lines)
        rejected: List[str] = []
        reasons: List[str] = []

        if len(updated) >= 5:
            shape_index = next((i for i in range(len(updated) - 1, -1, -1) if updated[i].function == FunctionType.SHAPE), None)
            if shape_index is not None:
                removed = updated.pop(shape_index)
                rejected.append(removed.id)
                reasons.append(f"Rejected {removed.id}: removed to preserve requested tail line {tail.id}.")
            else:
                return lines, [tail.id], [f"Rejected {tail.id}: no removable shape slot available under 5-line cap."]

        updated.append(tail)
        return deduplicate_lines(updated), rejected, reasons

    def _fallback_minimal_regime(
        self,
        *,
        dominant: LinePrimitive,
        ranked_failures: List[str],
        selected_suppressions: List[LinePrimitive],
        prior_lines: List[LinePrimitive],
    ) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        fallback_suppression = self._first_compatible_suppression(dominant, selected_suppressions, ranked_failures)
        fallback_lines = [dominant] + ([fallback_suppression] if fallback_suppression else [])

        rejected: List[str] = []
        reasons: List[str] = []
        fallback_ids = {line.id for line in fallback_lines}

        for line in prior_lines:
            if line.id not in fallback_ids:
                rejected.append(line.id)
                reasons.append(f"Rejected {line.id}: dropped by minimal fallback regime.")

        return fallback_lines, rejected, reasons

    def _first_compatible_suppression(
        self,
        dominant: LinePrimitive,
        selected_suppressions: List[LinePrimitive],
        ranked_failures: List[str],
    ) -> Optional[LinePrimitive]:
        for suppression in selected_suppressions:
            if not has_hard_conflict(dominant, suppression):
                return suppression

        for failure in ranked_failures:
            for suppressor_id in FAILURE_SUPPRESSOR_MAP.get(failure, []):
                suppressor = LIBRARY[suppressor_id]
                if not has_hard_conflict(dominant, suppressor):
                    return suppressor

        return None

    def _build_regime(
        self,
        *,
        stage: Stage,
        lines: List[LinePrimitive],
        rejected_lines: List[str],
        rejection_reasons: List[str],
    ) -> Regime:
        dominance = [line for line in lines if line.function == FunctionType.DOMINANCE]
        suppressions = [line for line in lines if line.function == FunctionType.SUPPRESSION]
        shapes = [line for line in lines if line.function == FunctionType.SHAPE]
        tail = next((line for line in lines if line.function in (FunctionType.GATE, FunctionType.TRANSFER)), None)

        if not dominance:
            dominance = [select_dominant(stage, set())]

        return Regime(
            name=f"{stage.value.title()} Core",
            stage=stage,
            dominant_line=dominance[0],
            suppression_lines=suppressions,
            shape_lines=shapes,
            tail_line=tail,
            rejected_lines=rejected_lines,
            rejection_reasons=rejection_reasons,
            likely_failure_if_overused=CANONICAL_FAILURE_IF_OVERUSED[stage],
        )
