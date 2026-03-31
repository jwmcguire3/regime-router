from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..models import Regime, Stage
from ..routing import RegimeComposer
from ..state import RouterState
from .escalation_policy import EscalationPolicyResult
from .misrouting_detector import MisroutingDetectionResult
from .output_contract import RegimeOutputContract
from .transition_rules import (
    assumption_or_frame_collapse,
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
    def __init__(self) -> None:
        self._composer = RegimeComposer()

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
            state.switch_trigger = "switch_bound_reached"
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Switch limit reached; execution remains on current regime.",
                updated_state=state,
            )

        if assumption_or_frame_collapse(state, failure_signal):
            next_regime = self._resolve_stage(state, Stage.EXPLORATION)
            state.recommended_next_regime = next_regime
            state.switch_trigger = "assumption_or_frame_collapse"
            return SwitchOrchestrationResult(
                next_regime=next_regime,
                switch_recommended_now=True,
                reason_for_switch="Assumptions or frame collapsed; fallback to exploration.",
                updated_state=state,
            )

        if not completion_signal and not failure_signal and not detection.misrouting_detected:
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
        state.switch_trigger = (
            "semantic_validation_failed_in_operator"
            if semantic_failure
            else failure_signal or completion_signal or "misrouting_detected"
        )
        return SwitchOrchestrationResult(
            next_regime=next_regime,
            switch_recommended_now=True,
            reason_for_switch=f"Switching from {state.current_regime.stage.value} to {resolved_next_stage.value}.",
            updated_state=state,
        )

    def _resolve_stage(self, state: RouterState, stage: Stage) -> Regime:
        return state.resolve_regime(stage, self._composer.compose)
