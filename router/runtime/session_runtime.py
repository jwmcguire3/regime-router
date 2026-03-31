from __future__ import annotations

from typing import List, Set

from ..control import EscalationPolicy, MisroutingDetector, RegimeOutputContract, SwitchOrchestrator
from ..models import RegimeExecutionResult, RoutingFeatures
from ..state import RouterState


class SessionRuntime:
    def __init__(
        self,
        *,
        misrouting_detector: MisroutingDetector,
        escalation_policy: EscalationPolicy,
        switch_orchestrator: SwitchOrchestrator,
    ) -> None:
        self.misrouting_detector = misrouting_detector
        self.escalation_policy = escalation_policy
        self.switch_orchestrator = switch_orchestrator

    def run_orchestration_loop(
        self,
        *,
        state: RouterState,
        task: str,
        model: str,
        initial_result: RegimeExecutionResult,
        task_signals: List[str],
        risk_profile: Set[str],
        routing_features: RoutingFeatures,
        max_switches: int,
        execute_regime_once,
        update_state_from_execution,
    ) -> RegimeExecutionResult:
        current_result = initial_result
        while True:
            state.switches_attempted += 1
            switch_index = state.switches_attempted
            if state.switches_executed >= max_switches:
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=None,
                    switch_recommended=False,
                    switch_executed=False,
                    reason=f"Switch limit reached; max_switches={max_switches}.",
                    switch_trigger=state.switch_trigger,
                )
                state.orchestration_stop_reason = "switch_limit_reached"
                break
            output_contract = RegimeOutputContract(
                stage=current_result.stage,
                raw_response=current_result.raw_response,
                validation=current_result.validation,
            )
            detection = self.misrouting_detector.detect(state, output_contract)
            escalation = self.escalation_policy.evaluate(
                state=state,
                routing_features=routing_features,
                task_text=task,
                current_regime=state.current_regime,
                regime_confidence=state.regime_confidence,
                misrouting_result=detection,
            )
            state.escalation_debug = {
                "direction": escalation.escalation_direction,
                "justification": escalation.justification,
                "biases": {stage.value: v for stage, v in escalation.preferred_regime_biases.items()},
                "switch_pressure_adjustment": escalation.switch_pressure_adjustment,
                "signals": escalation.debug_signals,
            }
            prior_recommended_next = state.recommended_next_regime
            orchestrated = self.switch_orchestrator.orchestrate(
                state,
                output_contract,
                detection,
                switches_used=state.switches_executed,
                max_switches=max_switches,
                escalation=escalation,
            )
            state = orchestrated.updated_state
            if not orchestrated.switch_recommended_now or orchestrated.next_regime is None:
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=None,
                    switch_recommended=False,
                    switch_executed=False,
                    reason=orchestrated.reason_for_switch,
                    switch_trigger=state.switch_trigger,
                )
                state.orchestration_stop_reason = "switch_not_recommended"
                break
            if orchestrated.next_regime.stage == state.current_regime.stage:
                state.recommended_next_regime = prior_recommended_next
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage,
                    switch_recommended=True,
                    switch_executed=False,
                    reason="Switch denied to avoid same-stage loop.",
                    switch_trigger=state.switch_trigger,
                )
                state.orchestration_stop_reason = "loop_prevented_same_stage"
                break
            if orchestrated.next_regime.stage in state.executed_regime_stages:
                state.recommended_next_regime = prior_recommended_next
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage,
                    switch_recommended=True,
                    switch_executed=False,
                    reason="Switch denied to avoid re-entering a previously executed stage.",
                    switch_trigger=state.switch_trigger,
                )
                state.orchestration_stop_reason = "loop_prevented_prior_stage"
                break
            state.record_switch_decision(
                switch_index=switch_index,
                from_stage=state.current_regime.stage,
                to_stage=orchestrated.next_regime.stage,
                switch_recommended=True,
                switch_executed=True,
                reason=orchestrated.reason_for_switch,
                switch_trigger=state.switch_trigger,
            )
            state.switches_executed += 1
            state.current_regime = orchestrated.next_regime
            current_result = execute_regime_once(
                task=task,
                model=model,
                regime=orchestrated.next_regime,
                task_signals=task_signals,
                risk_profile=risk_profile,
            )
            update_state_from_execution(state, current_result, reason_entered=orchestrated.reason_for_switch)
        return current_result
