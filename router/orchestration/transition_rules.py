from __future__ import annotations

from typing import Optional

from ..models import Stage
from ..state import RouterState
from .escalation_policy import EscalationPolicyResult
from .misrouting_detector import MisroutingDetectionResult
from .output_contract import RegimeOutputContract

ALLOWED_PATHWAYS = {
    Stage.EXPLORATION: {Stage.SYNTHESIS},
    Stage.SYNTHESIS: {Stage.EPISTEMIC, Stage.ADVERSARIAL},
    Stage.EPISTEMIC: {Stage.OPERATOR},
    Stage.ADVERSARIAL: {Stage.OPERATOR},
    Stage.OPERATOR: {Stage.EPISTEMIC, Stage.BUILDER},
    Stage.BUILDER: set(),
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


def next_stage(
    state: RouterState,
    completion_signal: str,
    failure_signal: str,
    detection: MisroutingDetectionResult,
    escalation: Optional[EscalationPolicyResult],
    *,
    semantic_operator_failure: bool = False,
) -> Optional[Stage]:
    current_stage = state.current_regime.stage
    allowed = ALLOWED_PATHWAYS.get(current_stage, set())
    recommended = detection.recommended_next_stage
    if detection.misrouting_detected and recommended in allowed:
        if escalation and escalation.escalation_direction == "looser" and escalation.switch_pressure_adjustment <= -2:
            # Light-touch damping: do not override explicit completion/failure pathways, only soft misrouting nudges.
            return None
        return recommended

    if current_stage == Stage.EXPLORATION and completion_signal:
        return Stage.SYNTHESIS
    if current_stage == Stage.SYNTHESIS and failure_signal:
        if recommended == Stage.ADVERSARIAL and Stage.ADVERSARIAL in allowed:
            return Stage.ADVERSARIAL
        return Stage.EPISTEMIC if Stage.EPISTEMIC in allowed else None
    if current_stage == Stage.OPERATOR and (semantic_operator_failure or (failure_signal and detection.misrouting_detected)):
        if recommended == Stage.EPISTEMIC and Stage.EPISTEMIC in allowed:
            return Stage.EPISTEMIC
        return Stage.EPISTEMIC if Stage.EPISTEMIC in allowed else None
    if current_stage in {Stage.EPISTEMIC, Stage.ADVERSARIAL} and completion_signal:
        return Stage.OPERATOR if Stage.OPERATOR in allowed else None
    if current_stage == Stage.OPERATOR and completion_signal and state.recurrence_potential >= 2.0:
        return Stage.BUILDER if Stage.BUILDER in allowed else None
    if (
        escalation
        and escalation.escalation_direction == "stricter"
        and escalation.switch_pressure_adjustment >= 2
        and detection.recommended_next_stage in allowed
    ):
        return detection.recommended_next_stage
    return None
