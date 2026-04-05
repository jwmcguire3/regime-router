from __future__ import annotations

from typing import Optional

from ..models import ReentryJustification, Stage
from ..state import RouterState
from .canonical_status import CanonicalStatus
from .escalation_policy import EscalationPolicyResult
from .misrouting_detector import MisroutingDetectionResult
from .output_contract import RegimeOutputContract

DEFAULT_FORWARD_PATHWAYS = {
    Stage.EXPLORATION: {Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.OPERATOR},
    Stage.SYNTHESIS: {Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR},
    Stage.EPISTEMIC: {Stage.ADVERSARIAL, Stage.OPERATOR},
    Stage.ADVERSARIAL: {Stage.OPERATOR},
    Stage.OPERATOR: set(),
    Stage.BUILDER: set(),
}

CONDITIONAL_REENTRY_PATHWAYS = {
    Stage.EXPLORATION: {Stage.EXPLORATION, Stage.ADVERSARIAL, Stage.BUILDER},
    Stage.SYNTHESIS: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.BUILDER},
    Stage.EPISTEMIC: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.BUILDER},
    Stage.ADVERSARIAL: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.BUILDER},
    Stage.OPERATOR: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR, Stage.BUILDER},
    Stage.BUILDER: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR, Stage.BUILDER},
}


def signal_from_output(output: RegimeOutputContract, *, key: str) -> str:
    parsed = output.validation.get("parsed", {})
    if isinstance(parsed, dict):
        value = parsed.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def assumption_or_frame_collapse(state: RouterState, failure_signal: str) -> bool:
    normalized = " ".join(failure_signal.lower().split())
    assumption_collapse_signaled = "assumption" in normalized and "collapse" in normalized
    frame_collapse_signaled = "frame" in normalized and "collapse" in normalized and "pressure" not in normalized
    if not (assumption_collapse_signaled or frame_collapse_signaled):
        return False
    return bool(state.assumptions) and (bool(state.contradictions) or assumption_collapse_signaled)


def operator_semantic_failure(output: RegimeOutputContract) -> bool:
    if output.stage != Stage.OPERATOR:
        return False
    validation = output.validation
    semantic_failures = validation.get("semantic_failures", [])
    if not semantic_failures:
        return False
    return (
        bool(validation.get("valid_json", False))
        and bool(validation.get("required_keys_present", False))
        and bool(validation.get("artifact_fields_present", False))
        and bool(validation.get("artifact_type_matches", False))
        and bool(validation.get("contract_controls_valid", False))
        and not bool(validation.get("semantic_valid", True))
    )


def control_failure_regime_mismatch(output: RegimeOutputContract) -> bool:
    validation = output.validation
    if not validation.get("valid_json", False):
        return False
    control_failures = validation.get("control_failures", [])
    return any("regime field mismatch" in str(f).lower() for f in control_failures)


def defect_class_from_context(
    *,
    state: RouterState,
    current_stage: Stage,
    next_stage: Optional[Stage],
    completion_signal: str,
    failure_signal: str,
    detection: MisroutingDetectionResult,
    output: RegimeOutputContract,
) -> Optional[str]:
    if next_stage is None:
        return None
    normalized_failure = failure_signal.lower()
    validation = output.validation
    semantic_failures = validation.get("semantic_failures", [])
    if semantic_failures or control_failure_regime_mismatch(output):
        return "contract_invalidated"
    if "collapse" in normalized_failure:
        return "frame_failure"
    if "evidence" in normalized_failure or "insufficient_support" in normalized_failure:
        return "evidence_failure"
    if "break" in normalized_failure or "destabiliz" in normalized_failure:
        return "break_condition_discovery"
    if "decision_not_actionable" in normalized_failure or "not_actionable" in normalized_failure:
        return "decision_non_actionable"
    if current_stage == Stage.BUILDER and ("too_abstract" in normalized_failure or "over-abstract" in normalized_failure):
        return "abstraction_overshot"
    if "constraint" in normalized_failure or "new_constraint" in normalized_failure:
        return "new_constraint"
    if detection.misrouting_detected and detection.recommended_next_stage is not None:
        return "new_constraint"
    if completion_signal and not failure_signal and next_stage in CONDITIONAL_REENTRY_PATHWAYS.get(current_stage, set()):
        return "break_condition_discovery"
    return None


def repair_target_for_stage(target_stage: Stage, defect_class: str) -> str:
    mapping = {
        "frame_failure": "rebuild_frame_and_assumptions",
        "evidence_failure": "strengthen_evidence_quality",
        "break_condition_discovery": "incorporate_break_conditions",
        "decision_non_actionable": "produce_actionable_decision",
        "abstraction_overshot": "reduce_abstraction_to_executable_shape",
        "contract_invalidated": "repair_output_contract_integrity",
        "new_constraint": "integrate_new_constraint_into_plan",
    }
    base = mapping.get(defect_class, "repair_upstream_defect")
    return f"{target_stage.value}:{base}"


def build_reentry_justification(
    *,
    state: RouterState,
    current_stage: Stage,
    next_stage: Optional[Stage],
    canonical: CanonicalStatus,
    detection: MisroutingDetectionResult,
    output: RegimeOutputContract,
) -> Optional[ReentryJustification]:
    if next_stage is None:
        return None
    if not transition_requires_justification(current_stage, next_stage, state):
        return None
    defect_class = defect_class_from_context(
        state=state,
        current_stage=current_stage,
        next_stage=next_stage,
        completion_signal=canonical.completion_signal,
        failure_signal=canonical.failure_signal,
        detection=detection,
        output=output,
    )
    if defect_class is None:
        return None
    contract_delta = (state.last_contract_delta or "").strip()
    state_delta = (state.last_state_delta or "").strip()
    if not contract_delta:
        contract_delta = "contract_invalidated" if defect_class == "contract_invalidated" else "contract_repair_required"
    if not state_delta:
        state_delta = "upstream_defect_exposed"
    return ReentryJustification(
        defect_class=defect_class,
        repair_target=repair_target_for_stage(next_stage, defect_class),
        contract_delta=contract_delta,
        state_delta=state_delta,
    )


def transition_requires_justification(current_stage: Stage, next_stage: Stage, state: RouterState) -> bool:
    if next_stage in DEFAULT_FORWARD_PATHWAYS.get(current_stage, set()):
        return False
    if next_stage == current_stage:
        return True
    if next_stage in CONDITIONAL_REENTRY_PATHWAYS.get(current_stage, set()):
        return True
    return next_stage in state.executed_regime_stages


def _looks_like_reusable_structure(state: RouterState, output: RegimeOutputContract, detection: MisroutingDetectionResult) -> bool:
    parsed = output.validation.get("parsed", {})
    artifact = parsed.get("artifact", {}) if isinstance(parsed, dict) else {}
    if not isinstance(artifact, dict):
        artifact = {}
    structural_markers = (
        artifact.get("reusable_pattern"),
        artifact.get("modules"),
        artifact.get("implementation_sequence"),
    )
    if any(bool(marker) for marker in structural_markers):
        return True
    rationale = f"{artifact.get('rationale', '')} {state.current_bottleneck}".lower()
    if "repeat" in rationale or "recurr" in rationale or "template" in rationale:
        return True
    if detection.recommended_next_stage == Stage.BUILDER and state.recurrence_potential >= 2.0:
        return True
    return False


def next_stage(
    state: RouterState,
    detection: MisroutingDetectionResult,
    escalation: Optional[EscalationPolicyResult],
    output: RegimeOutputContract,
    *,
    canonical: CanonicalStatus,
    semantic_operator_failure: bool = False,
) -> Optional[Stage]:
    current_stage = state.current_regime.stage
    recommended = detection.recommended_next_stage
    completion_signal = canonical.completion_signal
    failure_signal = canonical.failure_signal

    if detection.misrouting_detected and recommended is not None:
        if escalation and escalation.escalation_direction == "looser" and escalation.switch_pressure_adjustment <= -2:
            return None
        if recommended in DEFAULT_FORWARD_PATHWAYS.get(current_stage, set()):
            return recommended
        if recommended in CONDITIONAL_REENTRY_PATHWAYS.get(current_stage, set()):
            defect_class = defect_class_from_context(
                state=state,
                current_stage=current_stage,
                next_stage=recommended,
                completion_signal=completion_signal,
                failure_signal=failure_signal,
                detection=detection,
                output=output,
            )
            return recommended if defect_class else None

    if current_stage == Stage.EXPLORATION and canonical.terminal_signal == "completion":
        return Stage.SYNTHESIS
    if current_stage == Stage.SYNTHESIS and canonical.terminal_signal in ("failure", "contradictory"):
        if recommended == Stage.ADVERSARIAL:
            return Stage.ADVERSARIAL
        return Stage.EPISTEMIC
    if current_stage == Stage.OPERATOR and (
        semantic_operator_failure
        or control_failure_regime_mismatch(output)
        or canonical.terminal_signal in ("failure", "contradictory")
    ):
        return Stage.EPISTEMIC
    if current_stage in {Stage.EPISTEMIC, Stage.ADVERSARIAL} and canonical.terminal_signal == "completion":
        return Stage.OPERATOR
    if current_stage == Stage.OPERATOR and canonical.terminal_signal == "completion":
        suggested_builder = recommended == Stage.BUILDER or (
            state.recommended_next_regime is not None and state.recommended_next_regime.stage == Stage.BUILDER
        )
        if suggested_builder and _looks_like_reusable_structure(state, output, detection):
            return Stage.BUILDER
        return None
    if (
        escalation
        and escalation.escalation_direction == "stricter"
        and escalation.switch_pressure_adjustment >= 2
        and detection.recommended_next_stage is not None
    ):
        target = detection.recommended_next_stage
        if target in DEFAULT_FORWARD_PATHWAYS.get(current_stage, set()):
            return target
        if target in CONDITIONAL_REENTRY_PATHWAYS.get(current_stage, set()):
            defect_class = defect_class_from_context(
                state=state,
                current_stage=current_stage,
                next_stage=target,
                completion_signal=completion_signal,
                failure_signal=failure_signal,
                detection=detection,
                output=output,
            )
            if defect_class:
                return target
    return None
