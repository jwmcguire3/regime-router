from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import Regime, Stage
from ..routing import RegimeComposer
from ..state import RouterState
from .collapse_detector import CollapseDetector
from .escalation_policy import EscalationPolicyResult
from .misrouting_detector import MisroutingDetectionResult
from .output_contract import RegimeOutputContract
from .canonical_status import canonical_status_from_validation
from .transition_rules import (
    build_reentry_justification,
    next_stage,
    operator_semantic_failure,
    signal_from_output,
)

BAD_OUTPUT_CAUSE = "invalid_output_unrecoverable"


@dataclass(frozen=True)
class SwitchOrchestrationResult:
    next_regime: Optional[Regime]
    switch_recommended_now: bool
    reason_for_switch: str
    updated_state: RouterState


class SwitchOrchestrator:
    def __init__(self, composer: RegimeComposer, collapse_detector: Optional[CollapseDetector] = None) -> None:
        self._composer = composer
        self._collapse_detector = collapse_detector or CollapseDetector()

    def orchestrate(
        self,
        state: RouterState,
        output: RegimeOutputContract,
        detection: MisroutingDetectionResult,
        *,
        switches_used: int,
        max_switches: int = 2,
        escalation: Optional[EscalationPolicyResult] = None,
    ) -> SwitchOrchestrationResult:
        completion_signal = signal_from_output(output, key="completion_signal")
        failure_signal = signal_from_output(output, key="failure_signal")
        semantic_failure = operator_semantic_failure(output)
        bounded = switches_used >= max_switches
        bad_output_cause = _normalized_bad_output_cause(output)

        if bounded:
            state.observed_switch_cause = "switch_bound_reached"
            state.switch_trigger = state.observed_switch_cause
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Switch limit reached; execution remains on current regime.",
                updated_state=state,
            )

        if bad_output_cause:
            if state.current_regime.stage == Stage.EXPLORATION:
                state.observed_switch_cause = bad_output_cause
                state.switch_trigger = bad_output_cause
                return SwitchOrchestrationResult(
                    next_regime=None,
                    switch_recommended_now=False,
                    reason_for_switch="Exploration produced unrecoverable invalid output; stopping to avoid churn.",
                    updated_state=state,
                )
            next_regime = self._resolve_stage(state, Stage.EXPLORATION)
            state.recommended_next_regime = next_regime
            state.observed_switch_cause = bad_output_cause
            state.switch_trigger = bad_output_cause
            return SwitchOrchestrationResult(
                next_regime=next_regime,
                switch_recommended_now=True,
                reason_for_switch="Unrecoverable invalid output detected; fallback to exploration.",
                updated_state=state,
            )

        parsed = output.validation.get("parsed", {})
        parsed_mapping = parsed if isinstance(parsed, dict) else {}
        artifact = parsed_mapping.get("artifact", {})
        artifact_mapping = artifact if isinstance(artifact, dict) else {}
        canonical = canonical_status_from_validation(
            current_stage=output.stage,
            state=state,
            validation_result=output.validation,
            artifact=artifact_mapping,
        )
        collapse_detection = self._collapse_detector.detect(state, output.validation, artifact_mapping, failure_signal)
        if collapse_detection.collapse_detected:
            next_regime = self._resolve_stage(state, Stage.EXPLORATION)
            state.recommended_next_regime = next_regime
            state.observed_switch_cause = "assumption_or_frame_collapse"
            state.switch_trigger = state.observed_switch_cause
            return SwitchOrchestrationResult(
                next_regime=next_regime,
                switch_recommended_now=True,
                reason_for_switch=f"Collapse detected in active frame ({collapse_detection.reason}); fallback to exploration.",
                updated_state=state,
            )

        if canonical.terminal_signal == "neither" and not detection.misrouting_detected:
            state.last_reentry_justification = None
            state.observed_switch_cause = None
            state.switch_trigger = None
            parsed = output.validation.get("parsed", {})
            parsed_mapping = parsed if isinstance(parsed, dict) else {}
            recommended_raw = parsed_mapping.get("recommended_next_regime")
            sparse_next_stage = Stage(str(recommended_raw).strip().lower()) if isinstance(recommended_raw, str) and str(recommended_raw).strip().lower() in Stage._value2member_map_ else None
            sparse_justification = build_reentry_justification(
                state=state,
                current_stage=state.current_regime.stage,
                next_stage=sparse_next_stage,
                canonical=canonical,
                detection=detection,
                output=output,
            )
            if sparse_justification is None:
                return SwitchOrchestrationResult(
                    next_regime=None,
                    switch_recommended_now=False,
                    reason_for_switch="No structured switching signal is active.",
                    updated_state=state,
                )
            state.last_reentry_justification = sparse_justification
            next_regime = self._resolve_stage(state, sparse_next_stage) if sparse_next_stage is not None else None
            if next_regime is None:
                return SwitchOrchestrationResult(
                    next_regime=None,
                    switch_recommended_now=False,
                    reason_for_switch="No structured switching signal is active.",
                    updated_state=state,
                )
            state.recommended_next_regime = next_regime
            state.observed_switch_cause = sparse_justification.defect_class
            state.switch_trigger = state.observed_switch_cause
            return SwitchOrchestrationResult(
                next_regime=next_regime,
                switch_recommended_now=True,
                reason_for_switch=f"Defect evidence supports reentry toward {sparse_next_stage.value}.",
                updated_state=state,
            )

        current_stage = state.current_regime.stage
        resolved_next_stage = next_stage(
            state,
            detection,
            escalation,
            output,
            canonical=canonical,
            semantic_operator_failure=semantic_failure,
        )
        if resolved_next_stage is None:
            state.last_reentry_justification = None
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Current regime should continue; no allowed transition selected.",
                updated_state=state,
            )

        reentry_justification = build_reentry_justification(
            state=state,
            current_stage=current_stage,
            next_stage=resolved_next_stage,
            canonical=canonical,
            detection=detection,
            output=output,
        )
        state.last_reentry_justification = reentry_justification
        next_regime = self._resolve_stage(state, resolved_next_stage)
        state.recommended_next_regime = next_regime
        state.observed_switch_cause = (
            reentry_justification.defect_class
            if reentry_justification is not None
            else "semantic_validation_failed_in_operator"
            if semantic_failure
            else failure_signal or completion_signal or "misrouting_detected"
        )
        state.switch_trigger = state.observed_switch_cause
        return SwitchOrchestrationResult(
            next_regime=next_regime,
            switch_recommended_now=True,
            reason_for_switch=f"Switching from {state.current_regime.stage.value} to {resolved_next_stage.value}.",
            updated_state=state,
        )

    def _resolve_stage(self, state: RouterState, stage: Stage) -> Regime:
        return state.resolve_regime(stage, self._composer.compose)


def _normalized_bad_output_cause(output: RegimeOutputContract) -> str:
    validation = output.validation
    if "is_valid" not in validation:
        return ""
    if validation.get("is_valid", False):
        return ""
    raw_response = str(output.raw_response or "").strip()
    if not raw_response:
        return BAD_OUTPUT_CAUSE
    if validation.get("repair_attempted", False) and not validation.get("repair_succeeded", False):
        return BAD_OUTPUT_CAUSE
    if not validation.get("valid_json", False):
        return BAD_OUTPUT_CAUSE
    return ""
