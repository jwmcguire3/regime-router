from __future__ import annotations

from typing import List, Optional, Set

from router.models import (
    CANONICAL_DOMINANTS,
    DOMINANT_FAILURE_MAP,
    DOMINANT_SELECTION_RULES,
    FAILURE_SUPPRESSOR_MAP,
    FunctionType,
    LIBRARY,
    LinePrimitive,
    SHAPES_BY_STAGE,
    Stage,
    TAILS_BY_STAGE,
)


def select_dominant(stage: Stage, risk_profile: Set[str]) -> LinePrimitive:
    rules = DOMINANT_SELECTION_RULES.get(stage)
    if rules:
        for condition, dominant_id in rules.items():
            if condition != "default" and condition in risk_profile:
                return LIBRARY[dominant_id]
        if "default" in rules:
            return LIBRARY[rules["default"]]

    return LIBRARY[CANONICAL_DOMINANTS[stage][0]]


def rank_failures_by_cost(dominant: LinePrimitive, risk_profile: Set[str]) -> List[str]:
    failures = DOMINANT_FAILURE_MAP.get(dominant.id, [])

    def score(failure: str) -> int:
        value = 0
        if failure in risk_profile:
            value += 2
        if failure in dominant.risks:
            value += 1
        return value

    return sorted(failures, key=lambda f: (-score(f), failures.index(f)))


def _is_compatible_with_dominant(candidate: LinePrimitive, dominant: LinePrimitive) -> bool:
    if candidate.id in dominant.incompatible_with:
        return False
    if dominant.id in candidate.incompatible_with:
        return False
    return True


def select_suppressions(
    dominant: LinePrimitive,
    ranked_failures: List[str],
    risk_profile: Set[str],
) -> List[LinePrimitive]:
    del risk_profile  # currently covered by ranked_failures ordering

    selected: List[LinePrimitive] = []
    selected_ids: Set[str] = set()

    for failure in ranked_failures:
        suppressor_ids = FAILURE_SUPPRESSOR_MAP.get(failure, [])
        for suppressor_id in suppressor_ids:
            if suppressor_id in selected_ids:
                continue
            suppressor = LIBRARY[suppressor_id]
            if _is_compatible_with_dominant(suppressor, dominant):
                selected.append(suppressor)
                selected_ids.add(suppressor_id)
                break

        if selected:
            break

    if len(selected) == 1:
        first_id = selected[0].id
        for failure in ranked_failures[1:]:
            suppressor_ids = FAILURE_SUPPRESSOR_MAP.get(failure, [])
            if first_id in suppressor_ids:
                continue
            for suppressor_id in suppressor_ids:
                if suppressor_id in selected_ids:
                    continue
                suppressor = LIBRARY[suppressor_id]
                if _is_compatible_with_dominant(suppressor, dominant):
                    selected.append(suppressor)
                    selected_ids.add(suppressor_id)
                    break
            if len(selected) == 2:
                break

    return selected[:2]


def select_shapes(
    stage: Stage,
    dominant: LinePrimitive,
    suppressions: List[LinePrimitive],
    risk_profile: Set[str],
) -> List[LinePrimitive]:
    del suppressions

    if stage == Stage.SYNTHESIS and "high_stakes" in risk_profile:
        return []

    selected: List[LinePrimitive] = []
    for shape_id in SHAPES_BY_STAGE.get(stage, []):
        shape = LIBRARY[shape_id]
        if shape.stage != dominant.stage:
            continue
        if not _is_compatible_with_dominant(shape, dominant):
            continue
        selected.append(shape)
        if len(selected) == 2:
            break

    return selected


def select_tail(stage: Stage, handoff_expected: bool, risk_profile: Set[str]) -> Optional[LinePrimitive]:
    tails = [LIBRARY[line_id] for line_id in TAILS_BY_STAGE.get(stage, [])]

    if handoff_expected:
        transfer = next((line for line in tails if line.function == FunctionType.TRANSFER), None)
        if transfer is not None:
            return transfer

    if stage == Stage.OPERATOR:
        return next((line for line in tails if line.function == FunctionType.GATE), None)

    if stage == Stage.EPISTEMIC and "strict" in risk_profile:
        return next((line for line in tails if line.function == FunctionType.GATE), None)

    return None
