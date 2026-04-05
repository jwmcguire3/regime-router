from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Mapping, Optional

from ..models import Stage
from ..state import RouterState
from .misrouting_rules import failure_signal_active


@dataclass(frozen=True)
class CanonicalStatus:
    terminal_signal: Literal["completion", "failure", "contradictory", "neither"]
    artifact_status: Literal["valid_complete", "valid_blocked", "invalid", "repairable"]
    switch_posture: Literal["stay", "repair", "stop", "switch"]
    completion_signal: str
    failure_signal: str
    is_valid: bool
    structurally_valid: bool
    semantic_valid: bool
    control_conflict: bool
    recommended_next_stage: Optional[Stage]


def canonical_status_from_validation(
    *,
    current_stage: Stage,
    state: RouterState,
    validation_result: Mapping[str, object],
    artifact: Mapping[str, object],
) -> CanonicalStatus:
    parsed = validation_result.get("parsed", {})
    if not isinstance(parsed, dict):
        parsed = {}

    is_valid = bool(validation_result.get("is_valid", False))
    structurally_valid = bool(
        validation_result.get("valid_json", False)
        and validation_result.get("required_keys_present", False)
        and validation_result.get("artifact_fields_present", False)
        and validation_result.get("artifact_type_matches", False)
        and validation_result.get("contract_controls_valid", False)
    )
    semantic_valid = bool(validation_result.get("semantic_valid", False))

    completion_signal = str(parsed.get("completion_signal", "")).strip()
    failure_signal = str(parsed.get("failure_signal", "")).strip()
    recommended_next_stage = _parse_recommended_next_stage(parsed.get("recommended_next_regime"))

    artifact_payload: Dict[str, object] = dict(artifact) if isinstance(artifact, Mapping) else {}
    failure_is_active = failure_signal_active(current_stage, state, artifact_payload)

    if not is_valid:
        terminal_signal: Literal["completion", "failure", "contradictory", "neither"] = "failure"
    elif not completion_signal and not failure_is_active:
        terminal_signal = "neither"
    elif completion_signal and not failure_is_active:
        terminal_signal = "completion"
    elif not completion_signal and failure_is_active:
        terminal_signal = "failure"
    elif completion_signal and failure_is_active:
        terminal_signal = "contradictory"
    else:
        terminal_signal = "neither"

    if not is_valid and not structurally_valid:
        artifact_status: Literal["valid_complete", "valid_blocked", "invalid", "repairable"] = "invalid"
    elif not is_valid and structurally_valid:
        artifact_status = "repairable"
    elif is_valid and terminal_signal == "completion":
        artifact_status = "valid_complete"
    else:
        artifact_status = "valid_blocked"

    if terminal_signal == "completion":
        switch_posture: Literal["stay", "repair", "stop", "switch"] = "stay"
    elif terminal_signal == "failure":
        switch_posture = "switch"
    elif terminal_signal == "contradictory":
        switch_posture = "repair"
    else:
        switch_posture = "stay"

    return CanonicalStatus(
        terminal_signal=terminal_signal,
        artifact_status=artifact_status,
        switch_posture=switch_posture,
        completion_signal=completion_signal,
        failure_signal=failure_signal,
        is_valid=is_valid,
        structurally_valid=structurally_valid,
        semantic_valid=semantic_valid,
        control_conflict=terminal_signal == "contradictory",
        recommended_next_stage=recommended_next_stage,
    )


def _parse_recommended_next_stage(raw_value: object) -> Optional[Stage]:
    if isinstance(raw_value, Stage):
        return raw_value
    if not isinstance(raw_value, str):
        return None

    normalized = raw_value.strip().lower()
    if not normalized:
        return None
    if normalized in Stage._value2member_map_:
        return Stage(normalized)

    for stage in Stage:
        if stage.value in normalized:
            return stage
    return None
