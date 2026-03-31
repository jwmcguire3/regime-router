from __future__ import annotations

from typing import Dict

from ..models import Stage
from ..state import RouterState


def failure_signal_active(stage: Stage, state: RouterState, artifact: Dict[str, object]) -> bool:
    dispatch = {
        Stage.EXPLORATION: failure_signal_exploration,
        Stage.SYNTHESIS: failure_signal_synthesis,
        Stage.EPISTEMIC: failure_signal_epistemic,
        Stage.ADVERSARIAL: failure_signal_adversarial,
        Stage.OPERATOR: failure_signal_operator,
        Stage.BUILDER: failure_signal_builder,
    }
    handler = dispatch.get(stage)
    return handler(state, artifact) if handler else False


def completion_signal_active(stage: Stage, state: RouterState, artifact: Dict[str, object]) -> bool:
    dispatch = {
        Stage.EXPLORATION: completion_signal_exploration,
        Stage.SYNTHESIS: completion_signal_synthesis,
        Stage.EPISTEMIC: completion_signal_epistemic,
    }
    handler = dispatch.get(stage)
    if handler:
        return handler(state, artifact)
    return not failure_signal_active(stage, state, artifact)


def failure_signal_exploration(state: RouterState, artifact: Dict[str, object]) -> bool:
    del state
    candidate_frames = item_count(artifact.get("candidate_frames"))
    has_differentiation = exploration_has_differentiation(artifact)
    return candidate_frames >= 5 and not has_differentiation


def failure_signal_synthesis(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_central_claim = present(artifact.get("central_claim"))
    has_organizing_idea = present(artifact.get("organizing_idea"))
    has_support = present(artifact.get("supporting_structure"))
    contradictions_live = len(state.contradictions) > 0
    has_pressure_points = present(artifact.get("pressure_points"))
    unsupported_unification = has_central_claim and has_organizing_idea and not has_support and not has_pressure_points
    stress_without_structure = has_central_claim and has_organizing_idea and has_pressure_points and not has_support
    contradictions_flattened = contradictions_live and not has_pressure_points and not has_support
    return unsupported_unification or stress_without_structure or contradictions_flattened


def failure_signal_epistemic(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_support_separation = epistemic_has_support_separation(artifact)
    has_uncertainty_handling = epistemic_has_uncertainty_handling(state, artifact)
    return not has_support_separation and not has_uncertainty_handling


def failure_signal_adversarial(state: RouterState, artifact: Dict[str, object]) -> bool:
    del state
    destabilizers = artifact.get("top_destabilizers")
    residual_risks = artifact.get("residual_risks")
    same_objections = normalized(destabilizers) == normalized(residual_risks) and present(destabilizers)
    no_revision_movement = not present(artifact.get("survivable_revisions"))
    return same_objections or (present(destabilizers) and no_revision_movement)


def failure_signal_operator(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_decision = present(artifact.get("decision"))
    missing_decision = not has_decision
    missing_tradeoff = not present(artifact.get("tradeoff_accepted"))
    missing_rationale = not present(artifact.get("rationale"))
    missing_next_actions = not present(artifact.get("next_actions"))
    missing_fallback = not present(artifact.get("fallback_trigger"))
    missing_review_point = not present(artifact.get("review_point"))
    assumptions_hidden = has_live_assumptions(state) and missing_rationale and missing_fallback
    return (
        missing_decision
        or missing_tradeoff
        or missing_rationale
        or missing_next_actions
        or missing_fallback
        or missing_review_point
        or assumptions_hidden
    )


def failure_signal_builder(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_modules_or_interfaces = present(artifact.get("modules")) or present(artifact.get("interfaces"))
    return has_modules_or_interfaces and not recurrence_established(state)


def completion_signal_exploration(state: RouterState, artifact: Dict[str, object]) -> bool:
    del state
    candidate_frames = item_count(artifact.get("candidate_frames"))
    has_differentiation = exploration_has_differentiation(artifact)
    return candidate_frames >= 3 and has_differentiation


def completion_signal_synthesis(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_central_pattern = present(artifact.get("central_claim")) or present(artifact.get("organizing_idea"))
    has_connective_structure = present(artifact.get("supporting_structure")) or present(artifact.get("pressure_points"))
    if len(state.contradictions) > 0:
        has_connective_structure = has_connective_structure or present(artifact.get("contradictions"))
    return has_central_pattern and has_connective_structure


def completion_signal_epistemic(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_support_separation = epistemic_has_support_separation(artifact)
    has_uncertainty_handling = epistemic_has_uncertainty_handling(state, artifact)
    return has_support_separation and has_uncertainty_handling


def exploration_has_differentiation(artifact: Dict[str, object]) -> bool:
    has_selection_criteria = present(artifact.get("selection_criteria"))
    has_unresolved_axes = present(artifact.get("unresolved_axes"))
    candidate_frames = artifact.get("candidate_frames")
    rich_frames = 0
    if isinstance(candidate_frames, list):
        for frame in candidate_frames:
            if isinstance(frame, str) and len(frame.strip().split()) >= 2:
                rich_frames += 1
    frame_text = normalized(candidate_frames)
    unique_tokens = {token for token in frame_text.split() if len(token) > 2}
    has_frame_diversity = len(unique_tokens) >= 6 and rich_frames >= 2
    return has_selection_criteria or has_unresolved_axes or has_frame_diversity


def epistemic_has_support_separation(artifact: Dict[str, object]) -> bool:
    has_supported = present(artifact.get("supported_claims"))
    has_unproven = present(artifact.get("plausible_but_unproven")) or present(artifact.get("omitted_due_to_insufficient_support"))
    return has_supported and has_unproven


def epistemic_has_uncertainty_handling(state: RouterState, artifact: Dict[str, object]) -> bool:
    has_uncertainty_markers = (
        present(artifact.get("contradictions"))
        or present(artifact.get("omitted_due_to_insufficient_support"))
        or present(artifact.get("hidden_assumptions"))
    )
    return has_uncertainty_markers or len(state.assumptions) > 0 or len(state.contradictions) > 0


def item_count(value: object) -> int:
    if isinstance(value, list):
        return sum(1 for item in value if present(item))
    return 0


def present(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(present(item) for item in value)
    if isinstance(value, dict):
        return any(present(v) for v in value.values())
    return value is not None


def normalized(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(normalized(item) for item in value if present(item)).strip()
    if isinstance(value, dict):
        return " | ".join(f"{k}:{normalized(v)}" for k, v in sorted(value.items()) if present(v)).strip()
    if isinstance(value, str):
        return " ".join(value.lower().split())
    if value is None:
        return ""
    return str(value).strip().lower()


def assumption_collapse_detected(state: RouterState, artifact: Dict[str, object]) -> bool:
    if not has_live_assumptions(state):
        return False
    return present(artifact.get("hidden_assumptions")) or present(artifact.get("contradictions"))


def adversarial_needed(artifact: Dict[str, object]) -> bool:
    return present(artifact.get("pressure_points"))


def operator_evidence_gap(artifact: Dict[str, object]) -> bool:
    has_decision = present(artifact.get("decision"))
    has_rationale = present(artifact.get("rationale"))
    return (not has_decision) or (has_decision and not has_rationale)


def recurrence_established(state: RouterState) -> bool:
    return state.recurrence_potential >= 2.0


def has_live_assumptions(state: RouterState) -> bool:
    return len(state.assumptions) > 0
