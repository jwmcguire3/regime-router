from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple

from router.models import DOMINANT_FAILURE_MAP, FAILURE_SUPPRESSOR_MAP, FunctionType, LinePrimitive

_FUNCTION_PRIORITY: Dict[FunctionType, int] = {
    FunctionType.DOMINANCE: 0,
    FunctionType.SUPPRESSION: 1,
    FunctionType.SHAPE: 2,
    FunctionType.GATE: 3,
    FunctionType.TRANSFER: 3,
}

_FAILURE_COST: Dict[str, int] = {
    "forced_closure": 10,
    "false_unification": 9,
    "coherence_over_truth": 9,
    "premature_lock": 8,
    "strawman_attack": 8,
    "overbuild": 7,
    "tunnel_vision": 7,
    "decision_drag": 6,
    "sprawl": 6,
}

_DECISION_DIRECTNESS_KEYWORDS = (
    "decision",
    "choose",
    "closure",
    "action",
    "tradeoff",
    "commit",
)


def has_hard_conflict(line_a: LinePrimitive, line_b: LinePrimitive) -> bool:
    if line_a.id in line_b.incompatible_with or line_b.id in line_a.incompatible_with:
        return True

    if (
        line_a.function == FunctionType.DOMINANCE
        and line_b.function == FunctionType.DOMINANCE
        and line_a.stage != line_b.stage
    ):
        return True

    if line_a.function == FunctionType.DOMINANCE and line_b.function == FunctionType.SUPPRESSION:
        return line_a.attractor in line_b.suppresses
    if line_b.function == FunctionType.DOMINANCE and line_a.function == FunctionType.SUPPRESSION:
        return line_b.attractor in line_a.suppresses
    return False


def has_soft_conflict(line_a: LinePrimitive, line_b: LinePrimitive) -> bool:
    if line_a.function == FunctionType.DOMINANCE and line_b.function == FunctionType.DOMINANCE:
        return True

    if line_a.function == FunctionType.SHAPE and line_b.function == FunctionType.DOMINANCE:
        return line_a.stage != line_b.stage
    if line_b.function == FunctionType.SHAPE and line_a.function == FunctionType.DOMINANCE:
        return line_b.stage != line_a.stage

    # Context-based shape/suppression ratio checks are handled by callers.
    return False


def resolve_conflict(keep: LinePrimitive, candidate: LinePrimitive) -> Tuple[bool, str]:
    keep_priority = _FUNCTION_PRIORITY[keep.function]
    candidate_priority = _FUNCTION_PRIORITY[candidate.function]
    if candidate_priority < keep_priority:
        return True, f"{candidate.id} kept: higher basin priority ({candidate.function.value})."
    if candidate_priority > keep_priority:
        return False, f"{candidate.id} rejected: lower basin priority than {keep.id}."

    keep_failure_score = _line_failure_coverage_score(keep)
    candidate_failure_score = _line_failure_coverage_score(candidate)
    if candidate_failure_score > keep_failure_score:
        return True, f"{candidate.id} kept: covers higher-cost failure modes."
    if candidate_failure_score < keep_failure_score:
        return False, f"{candidate.id} rejected: {keep.id} covers higher-cost failure modes."

    keep_directness = _decision_directness_score(keep)
    candidate_directness = _decision_directness_score(candidate)
    if candidate_directness > keep_directness:
        return True, f"{candidate.id} kept: more decision-direct guidance."
    if candidate_directness < keep_directness:
        return False, f"{candidate.id} rejected: less decision-direct than {keep.id}."

    if len(candidate.risks) < len(keep.risks):
        return True, f"{candidate.id} kept: fewer listed risks."
    return False, f"{candidate.id} rejected: tie defaults to existing line {keep.id}."


def validate_regime_grammar(lines: List[LinePrimitive]) -> Tuple[bool, List[str]]:
    violations: List[str] = []

    dominance = [line for line in lines if line.function == FunctionType.DOMINANCE]
    suppressions = [line for line in lines if line.function == FunctionType.SUPPRESSION]
    shapes = [line for line in lines if line.function == FunctionType.SHAPE]
    tails = [line for line in lines if line.function in {FunctionType.GATE, FunctionType.TRANSFER}]

    if len(dominance) not in {1, 2}:
        violations.append("Regime must contain exactly 1 dominance line, or exactly 2 compatible dominance lines.")
    elif len(dominance) == 2:
        a, b = dominance
        if a.stage != b.stage:
            violations.append("Two dominance lines are only allowed when they share the same stage.")
        if has_hard_conflict(a, b):
            violations.append("Dominance lines are incompatible.")

    if not 1 <= len(suppressions) <= 2:
        violations.append("Regime must contain 1-2 suppression lines.")

    if len(shapes) > 2:
        violations.append("Regime must contain 0-2 shape lines.")

    if len(tails) > 1:
        violations.append("Regime must contain at most one gate/transfer line.")

    if len(lines) > 5:
        violations.append("Regime must contain at most 5 lines.")

    for line_a, line_b in combinations(lines, 2):
        if has_hard_conflict(line_a, line_b):
            violations.append(f"Hard conflict: {line_a.id} vs {line_b.id}.")

    if dominance:
        dominant = dominance[0]
        dominant_failures = DOMINANT_FAILURE_MAP.get(dominant.id, [])
        allowed_suppressors = {
            suppressor_id
            for failure in dominant_failures
            for suppressor_id in FAILURE_SUPPRESSOR_MAP.get(failure, [])
        }
        for suppression in suppressions:
            if suppression.id not in allowed_suppressors:
                violations.append(
                    f"Suppression line {suppression.id} does not target failures for dominant {dominant.id}."
                )

    return len(violations) == 0, violations


def deduplicate_lines(lines: List[LinePrimitive], max_lines: int = 5) -> List[LinePrimitive]:
    selected_by_function: Dict[FunctionType, List[LinePrimitive]] = {
        FunctionType.DOMINANCE: [],
        FunctionType.SUPPRESSION: [],
        FunctionType.SHAPE: [],
        FunctionType.GATE: [],
        FunctionType.TRANSFER: [],
    }
    seen_ids = set()
    for line in lines:
        if line.id in seen_ids:
            continue
        selected_by_function[line.function].append(line)
        seen_ids.add(line.id)

    # Keep one best candidate for gate/transfer in the shared slot.
    tail_candidates = selected_by_function[FunctionType.GATE] + selected_by_function[FunctionType.TRANSFER]
    tail_candidates = _sort_within_function(tail_candidates)

    deduped: List[LinePrimitive] = []
    deduped.extend(_sort_within_function(selected_by_function[FunctionType.DOMINANCE]))
    deduped.extend(_sort_within_function(selected_by_function[FunctionType.SUPPRESSION]))
    deduped.extend(_sort_within_function(selected_by_function[FunctionType.SHAPE]))
    if tail_candidates:
        deduped.append(tail_candidates[0])

    deduped.sort(key=lambda line: (_FUNCTION_PRIORITY[line.function], len(line.risks), line.id))
    return deduped[:max_lines]


def _sort_within_function(lines: List[LinePrimitive]) -> List[LinePrimitive]:
    return sorted(lines, key=lambda line: (len(line.risks), line.id))


def _line_failure_coverage_score(line: LinePrimitive) -> int:
    covered = set(line.suppresses)
    if line.function == FunctionType.DOMINANCE:
        covered.update(DOMINANT_FAILURE_MAP.get(line.id, []))
    return sum(_FAILURE_COST.get(failure, 1) for failure in covered)


def _decision_directness_score(line: LinePrimitive) -> int:
    haystack = " ".join([line.text.lower(), line.attractor.lower(), " ".join(line.suppresses).lower()])
    return sum(1 for keyword in _DECISION_DIRECTNESS_KEYWORDS if keyword in haystack)
