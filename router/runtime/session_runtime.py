from __future__ import annotations

from typing import List, Optional, Set

from ..control import EscalationPolicy, MisroutingDetector, RegimeOutputContract, SwitchOrchestrationResult, SwitchOrchestrator
from ..models import (
    ControlAuthority,
    PolicyEvent,
    ReentryDecision,
    ReentryJustification,
    RegimeExecutionResult,
    RoutingDecision,
    RoutingFeatures,
    Stage,
)
from ..routing import RegimeComposer
from ..orchestration.stop_policy import StopPolicy
from ..state import Handoff, RouterState

BAD_OUTPUT_CAUSE = "invalid_output_unrecoverable"


class SessionRuntime:
    def __init__(
        self,
        *,
        misrouting_detector: MisroutingDetector,
        escalation_policy: EscalationPolicy,
        switch_orchestrator: SwitchOrchestrator,
        stop_policy: StopPolicy,
    ) -> None:
        self.misrouting_detector = misrouting_detector
        self.escalation_policy = escalation_policy
        self.switch_orchestrator = switch_orchestrator
        self.stop_policy = stop_policy
        self._composer = RegimeComposer()

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
        routing_decision: Optional[RoutingDecision],
        execute_regime_once,
        update_state_from_execution,
        handoff_from_state,
        compute_forward_handoff,
    ) -> RegimeExecutionResult:
        current_result = initial_result
        prior_handoff: Handoff = handoff_from_state(state)
        while True:
            stop_decision = self.stop_policy.should_stop(
                router_state=state,
                validation_result=current_result.validation,
                routing_decision=routing_decision,
                current_stage=state.current_regime.stage,
            )
            if stop_decision.should_stop:
                state.orchestration_stop_reason = stop_decision.reason
                break
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
                    planned_switch_condition=state.planned_switch_condition,
                    observed_switch_cause=state.observed_switch_cause or state.switch_trigger,
                    defect_class=None,
                    repair_target=None,
                    contract_delta=state.last_contract_delta,
                    state_delta=state.last_state_delta,
                    reentry_allowed=False,
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
            builder_gate_decision = self.stop_policy.should_stop(
                router_state=state,
                validation_result=current_result.validation,
                routing_decision=routing_decision,
                current_stage=state.current_regime.stage,
            )
            if builder_gate_decision.should_stop and builder_gate_decision.reason.startswith("Builder blocked:"):
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage if orchestrated.next_regime else None,
                    switch_recommended=orchestrated.switch_recommended_now,
                    switch_executed=False,
                    reason=builder_gate_decision.reason,
                    planned_switch_condition=state.planned_switch_condition,
                    observed_switch_cause=state.observed_switch_cause or state.switch_trigger,
                    defect_class=(state.last_reentry_justification.defect_class if state.last_reentry_justification else None),
                    repair_target=(state.last_reentry_justification.repair_target if state.last_reentry_justification else None),
                    contract_delta=state.last_contract_delta,
                    state_delta=state.last_state_delta,
                    reentry_allowed=False,
                )
                state.orchestration_stop_reason = builder_gate_decision.reason
                break
            if not orchestrated.switch_recommended_now or orchestrated.next_regime is None:
                if self._is_unrecoverable_invalid_output(current_result):
                    if state.current_regime.stage == Stage.EXPLORATION:
                        state.record_switch_decision(
                            switch_index=switch_index,
                            from_stage=state.current_regime.stage,
                            to_stage=None,
                            switch_recommended=False,
                            switch_executed=False,
                            reason="Exploration produced unrecoverable invalid output; stopping to avoid churn.",
                            planned_switch_condition=state.planned_switch_condition,
                            observed_switch_cause=BAD_OUTPUT_CAUSE,
                            defect_class=None,
                            repair_target=None,
                            contract_delta=state.last_contract_delta,
                            state_delta=state.last_state_delta,
                            reentry_allowed=False,
                        )
                        state.orchestration_stop_reason = "invalid_output_unrecoverable"
                        break
                    fallback_regime = state.resolve_regime(Stage.EXPLORATION, self._composer.compose)
                    state.recommended_next_regime = fallback_regime
                    orchestrated = SwitchOrchestrationResult(
                        next_regime=fallback_regime,
                        switch_recommended_now=True,
                        reason_for_switch="invalid_output_recovery",
                        updated_state=state,
                    )
                    state.observed_switch_cause = BAD_OUTPUT_CAUSE
                    state.switch_trigger = BAD_OUTPUT_CAUSE
                else:
                    state.record_switch_decision(
                        switch_index=switch_index,
                        from_stage=state.current_regime.stage,
                        to_stage=None,
                        switch_recommended=False,
                        switch_executed=False,
                        reason=orchestrated.reason_for_switch,
                        planned_switch_condition=state.planned_switch_condition,
                        observed_switch_cause=state.observed_switch_cause or state.switch_trigger,
                        defect_class=(state.last_reentry_justification.defect_class if state.last_reentry_justification else None),
                        repair_target=(state.last_reentry_justification.repair_target if state.last_reentry_justification else None),
                        contract_delta=state.last_contract_delta,
                        state_delta=state.last_state_delta,
                        reentry_allowed=False,
                    )
                    state.orchestration_stop_reason = "switch_not_recommended"
                    break
            reentry_decision = self._evaluate_reentry(
                state=state,
                next_stage=orchestrated.next_regime.stage,
                reason_for_switch=orchestrated.reason_for_switch,
            )
            if not reentry_decision.allowed:
                state.recommended_next_regime = prior_recommended_next
                state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage,
                    switch_recommended=True,
                    switch_executed=False,
                    reason=reentry_decision.reason,
                    planned_switch_condition=state.planned_switch_condition,
                    observed_switch_cause=state.observed_switch_cause or state.switch_trigger,
                    defect_class=(reentry_decision.justification.defect_class if reentry_decision.justification else None),
                    repair_target=(reentry_decision.justification.repair_target if reentry_decision.justification else None),
                    contract_delta=(reentry_decision.justification.contract_delta if reentry_decision.justification else state.last_contract_delta),
                    state_delta=(reentry_decision.justification.state_delta if reentry_decision.justification else state.last_state_delta),
                    reentry_allowed=False,
                )
                state.orchestration_stop_reason = "loop_prevented_reentry"
                break
            state.record_switch_decision(
                switch_index=switch_index,
                from_stage=state.current_regime.stage,
                to_stage=orchestrated.next_regime.stage,
                switch_recommended=True,
                switch_executed=True,
                reason=orchestrated.reason_for_switch,
                planned_switch_condition=state.planned_switch_condition,
                observed_switch_cause=state.observed_switch_cause or state.switch_trigger,
                defect_class=(reentry_decision.justification.defect_class if reentry_decision.justification else None),
                repair_target=(reentry_decision.justification.repair_target if reentry_decision.justification else None),
                contract_delta=(reentry_decision.justification.contract_delta if reentry_decision.justification else state.last_contract_delta),
                state_delta=(reentry_decision.justification.state_delta if reentry_decision.justification else state.last_state_delta),
                reentry_allowed=True,
            )
            state.switches_executed += 1
            state.current_regime = orchestrated.next_regime
            current_result = execute_regime_once(
                task=task,
                model=model,
                regime=orchestrated.next_regime,
                task_signals=task_signals,
                risk_profile=risk_profile,
                prior_handoff=prior_handoff,
            )
            update_state_from_execution(state, current_result, reason_entered=orchestrated.reason_for_switch)
            prior_handoff = compute_forward_handoff(current_result, state, orchestrated.next_regime)
            state.latest_forward_handoff = prior_handoff
        return current_result

    def _evaluate_reentry(
        self,
        *,
        state: RouterState,
        next_stage: Stage,
        reason_for_switch: str,
    ) -> ReentryDecision:
        current_stage = state.current_regime.stage
        same_stage = next_stage == current_stage
        is_prior_stage = next_stage in state.executed_regime_stages
        justification = state.last_reentry_justification
        cause = (state.observed_switch_cause or state.switch_trigger or "").strip()
        last_delta = (state.last_state_delta or "").strip()
        trivial_delta = last_delta in {"", "no_material_state_delta"}
        if same_stage and state.switch_history:
            last = state.switch_history[-1]
            if last.to_stage == next_stage and (last.observed_switch_cause or "") == cause:
                state.record_policy_event(
                    PolicyEvent(
                        rule_name="reentry_denied_same_stage_unchanged_cause",
                        authority=ControlAuthority.HARD_VETO.value,
                        consumed_features=["next_stage", "switch_history.last.to_stage", "observed_switch_cause"],
                        action="block_reentry",
                        detail=f"Same-stage reentry denied for {next_stage.value}: previous transition had unchanged cause '{cause}'.",
                    )
                )
                return ReentryDecision(allowed=False, reason="Switch denied: same-stage retry has unchanged brief/cause.")
        if is_prior_stage and trivial_delta:
            state.record_policy_event(
                PolicyEvent(
                    rule_name="reentry_denied_trivial_delta",
                    authority=ControlAuthority.HARD_VETO.value,
                    consumed_features=["next_stage", "executed_regime_stages", "last_state_delta"],
                    action="block_reentry",
                    detail=f"Prior stage {next_stage.value} denied: no material state delta ({last_delta or 'empty'}).",
                )
            )
            return ReentryDecision(allowed=False, reason="Switch denied: previously visited stage without material state delta.")
        if self._is_ping_pong(state, next_stage):
            state.record_policy_event(
                PolicyEvent(
                    rule_name="reentry_denied_ping_pong",
                    authority=ControlAuthority.HARD_VETO.value,
                    consumed_features=["next_stage", "switch_history", "observed_switch_cause", "last_state_delta"],
                    action="block_reentry",
                    detail=f"Reentry to {next_stage.value} denied due to ping-pong oscillation for cause '{cause}'.",
                )
            )
            return ReentryDecision(allowed=False, reason="Switch denied: ping-pong oscillation detected for same cause/target.")
        if same_stage or is_prior_stage:
            if not self._justification_complete(justification):
                state.record_policy_event(
                    PolicyEvent(
                        rule_name="reentry_denied_missing_justification",
                        authority=ControlAuthority.HARD_VETO.value,
                        consumed_features=[
                            "next_stage",
                            "same_stage_or_prior_stage",
                            "last_reentry_justification.defect_class",
                            "last_reentry_justification.repair_target",
                            "last_reentry_justification.contract_delta",
                            "last_reentry_justification.state_delta",
                        ],
                        action="block_reentry",
                        detail=f"Reentry to {next_stage.value} denied: required justification fields are incomplete.",
                    )
                )
                return ReentryDecision(allowed=False, reason="Switch denied: reentry justification is missing required fields.")
            state.record_policy_event(
                PolicyEvent(
                    rule_name="reentry_allowed_with_justification",
                    authority=ControlAuthority.SOFT_GUARDRAIL.value,
                    consumed_features=[
                        "next_stage",
                        "same_stage_or_prior_stage",
                        "last_reentry_justification.defect_class",
                        "last_reentry_justification.repair_target",
                        "last_reentry_justification.contract_delta",
                        "last_reentry_justification.state_delta",
                    ],
                    action="allow_reentry",
                    detail=f"Reentry to {next_stage.value} allowed with complete justification for '{reason_for_switch}'.",
                )
            )
            return ReentryDecision(allowed=True, reason=reason_for_switch, justification=justification)
        state.record_policy_event(
            PolicyEvent(
                rule_name="reentry_allowed_forward_progression",
                authority=ControlAuthority.ADVISORY_ONLY.value,
                consumed_features=["current_stage", "next_stage", "executed_regime_stages"],
                action="allow_reentry",
                detail=f"Forward progression from {current_stage.value} to {next_stage.value} allowed for '{reason_for_switch}'.",
            )
        )
        return ReentryDecision(allowed=True, reason=reason_for_switch, justification=justification)

    def _justification_complete(self, justification: Optional[ReentryJustification]) -> bool:
        if justification is None:
            return False
        return all(
            bool(str(value).strip())
            for value in (
                justification.defect_class,
                justification.repair_target,
                justification.contract_delta,
                justification.state_delta,
            )
        )

    def _is_ping_pong(self, state: RouterState, next_stage: Stage) -> bool:
        if len(state.switch_history) < 2:
            return False
        recent = state.switch_history[-2:]
        cause = (state.observed_switch_cause or state.switch_trigger or "").strip()
        last_delta = (state.last_state_delta or "").strip()
        if last_delta and last_delta != "no_material_state_delta":
            return False
        first, second = recent
        if not first.to_stage or not second.to_stage:
            return False
        oscillating_pair = first.from_stage == second.to_stage and first.to_stage == second.from_stage
        same_cause = (first.observed_switch_cause or "").strip() == (second.observed_switch_cause or "").strip() == cause
        return oscillating_pair and first.to_stage == next_stage and same_cause

    def _is_unrecoverable_invalid_output(self, result: RegimeExecutionResult) -> bool:
        validation = result.validation
        if "is_valid" not in validation:
            return False
        if validation.get("is_valid", False):
            return False
        if not str(result.raw_response or "").strip():
            return True
        if validation.get("repair_attempted", False) and not validation.get("repair_succeeded", False):
            return True
        return not bool(validation.get("valid_json", False))
