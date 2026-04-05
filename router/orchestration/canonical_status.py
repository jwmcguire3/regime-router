from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional
from typing import Literal

from ..models import Stage

TerminalSignal = Literal["completion", "failure", "contradictory", "neither"]
ArtifactStatus = Literal["valid_complete", "valid_blocked", "invalid", "repairable"]
SwitchPosture = Literal["stay", "repair", "stop", "switch"]


@dataclass(frozen=True)
class CanonicalStatus:
    """Shared status snapshot derived from a regime output contract."""

    terminal_signal: TerminalSignal
    artifact_status: ArtifactStatus
    switch_posture: SwitchPosture
    completion_signal: str
    failure_signal: str
    is_valid: bool
    structurally_valid: bool
    semantic_valid: bool
    control_conflict: bool
    recommended_next_stage: Optional[Stage]


def canonical_status_from_validation(
    validation_result: Mapping[str, object],
    *,
    current_stage: Optional[Stage] = None,
    should_stop: bool = False,
) -> CanonicalStatus:
    parsed = validation_result.get("parsed", {})
    completion_signal = ""
    failure_signal = ""
    recommended_next_stage: Optional[Stage] = None
    if isinstance(parsed, Mapping):
        completion_signal = str(parsed.get("completion_signal", "")).strip()
        failure_signal = str(parsed.get("failure_signal", "")).strip()
        candidate = parsed.get("recommended_next_regime")
        if isinstance(candidate, str):
            normalized = candidate.strip().lower()
            if normalized in Stage._value2member_map_:
                recommended_next_stage = Stage(normalized)

    is_valid = bool(validation_result.get("is_valid", False))
    structurally_valid = bool(
        validation_result.get("valid_json", False)
        and validation_result.get("required_keys_present", False)
        and validation_result.get("artifact_fields_present", False)
        and validation_result.get("artifact_type_matches", False)
        and validation_result.get("contract_controls_valid", False)
    )
    semantic_valid = bool(validation_result.get("semantic_valid", True))

    control_conflict = bool(completion_signal and failure_signal and completion_signal != failure_signal)
    if control_conflict:
        terminal_signal: TerminalSignal = "contradictory"
    elif completion_signal and is_valid:
        terminal_signal = "completion"
    elif failure_signal or not is_valid:
        terminal_signal = "failure"
    else:
        terminal_signal = "neither"

    if not structurally_valid:
        artifact_status: ArtifactStatus = "invalid"
    elif not semantic_valid:
        artifact_status = "repairable"
    elif terminal_signal == "completion":
        artifact_status = "valid_complete"
    else:
        artifact_status = "valid_blocked"

    if should_stop:
        switch_posture: SwitchPosture = "stop"
    elif artifact_status in {"invalid", "repairable"} or terminal_signal in {"failure", "contradictory"}:
        switch_posture = "repair"
    elif recommended_next_stage is not None and current_stage is not None and recommended_next_stage != current_stage:
        switch_posture = "switch"
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
        control_conflict=control_conflict,
        recommended_next_stage=recommended_next_stage,
    )
