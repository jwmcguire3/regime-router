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
from .transition_rules import (
    next_stage,
    operator_semantic_failure,
    signal_from_output,
)


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

        if bounded:
            state.observed_switch_cause = "switch_bound_reached"
            state.switch_trigger = state.observed_switch_cause
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Switch limit reached; execution remains on current regime.",
                updated_state=state,
            )

        parsed = output.validation.get("parsed", {})
        parsed_mapping = parsed if isinstance(parsed, dict) else {}
        artifact = parsed_mapping.get("artifact", {})
        artifact_mapping = artifact if isinstance(artifact, dict) else {}
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

        if not completion_signal and not failure_signal and not detection.misrouting_detected:
            state.observed_switch_cause = None
            state.switch_trigger = None
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="No structured switching signal is active.",
                updated_state=state,
            )

        resolved_next_stage = next_stage(
            state,
            completion_signal,
            failure_signal,
            detection,
            escalation,
            output,
            semantic_operator_failure=semantic_failure,
        )
        if resolved_next_stage is None:
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Current regime should continue; no allowed transition selected.",
                updated_state=state,
            )

        next_regime = self._resolve_stage(state, resolved_next_stage)
        state.recommended_next_regime = next_regime
        state.observed_switch_cause = (
            "semantic_validation_failed_in_operator"
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
