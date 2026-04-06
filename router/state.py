from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Mapping, Optional, Set

from .models import (
    FunctionType,
    LinePrimitive,
    PolicyEvent,
    ReentryJustification,
    Regime,
    RegimeConfidenceResult,
    RegimeExecutionResult,
    RoutingDecision,
    Stage,
)

RegimeConfidence = RegimeConfidenceResult


@dataclass
class RegimeStep:
    regime: Regime
    reason_entered: str
    completion_signal_seen: bool
    failure_signal_seen: bool
    outcome_summary: str


@dataclass
class SwitchDecisionRecord:
    switch_index: int
    from_stage: Stage
    to_stage: Optional[Stage]
    switch_recommended: bool
    switch_executed: bool
    reason: str
    planned_switch_condition: Optional[str]
    observed_switch_cause: Optional[str]
    switch_trigger: Optional[str]  # legacy alias for observed_switch_cause
    defect_class: Optional[str] = None
    repair_target: Optional[str] = None
    contract_delta: Optional[str] = None
    state_delta: Optional[str] = None
    reentry_allowed: Optional[bool] = None


@dataclass
class RouterState:
    task_id: str
    task_summary: str
    current_bottleneck: str
    current_regime: Regime
    runner_up_regime: Optional[Regime]
    regime_confidence: RegimeConfidence
    dominant_frame: Optional[str]
    knowns: List[str]
    uncertainties: List[str]
    contradictions: List[str]
    assumptions: List[str]
    risks: List[str]
    stage_goal: str
    substantive_assumptions: List[str] = field(default_factory=list)
    planned_switch_condition: Optional[str] = None
    observed_switch_cause: Optional[str] = None
    switch_trigger: Optional[str] = None  # legacy alias for observed_switch_cause
    recommended_next_regime: Optional[Regime] = None
    decision_pressure: float = 0.0
    fragility_pressure: float = 0.0
    possibility_space_need: float = 0.0
    synthesis_pressure: float = 0.0
    detected_markers: Dict[str, List[str]] = field(default_factory=dict)
    structural_signals: List[str] = field(default_factory=list)
    risk_tags: Set[str] = field(default_factory=set)
    evidence_demand: float = 0.0
    evidence_quality: float = 0.0
    recurrence_potential: float = 0.0
    prior_regimes: List[RegimeStep] = field(default_factory=list)
    orchestration_enabled: bool = False
    max_switches: int = 0
    switches_attempted: int = 0
    switches_executed: int = 0
    collapse_reentries: int = 0
    orchestration_stop_reason: Optional[str] = None
    executed_regime_stages: List[Stage] = field(default_factory=list)
    switch_history: List[SwitchDecisionRecord] = field(default_factory=list)
    policy_events: List[PolicyEvent] = field(default_factory=list)
    last_reentry_justification: Optional[ReentryJustification] = None
    last_state_delta: Optional[str] = None
    last_contract_delta: Optional[str] = None
    escalation_debug: Dict[str, object] = field(default_factory=dict)
    task_classification: Optional[Dict[str, object]] = None
    latest_forward_handoff: Optional["Handoff"] = None

    def record_regime_step(
        self,
        *,
        regime: Regime,
        reason_entered: str,
        completion_signal_seen: bool,
        failure_signal_seen: bool,
        outcome_summary: str,
    ) -> None:
        self.prior_regimes.append(
            RegimeStep(
                regime=regime,
                reason_entered=reason_entered,
                completion_signal_seen=completion_signal_seen,
                failure_signal_seen=failure_signal_seen,
                outcome_summary=outcome_summary,
            )
        )
        self.executed_regime_stages.append(regime.stage)

    def record_switch_decision(
        self,
        *,
        switch_index: int,
        from_stage: Stage,
        to_stage: Optional[Stage],
        switch_recommended: bool,
        switch_executed: bool,
        reason: str,
        planned_switch_condition: Optional[str],
        observed_switch_cause: Optional[str],
        defect_class: Optional[str] = None,
        repair_target: Optional[str] = None,
        contract_delta: Optional[str] = None,
        state_delta: Optional[str] = None,
        reentry_allowed: Optional[bool] = None,
    ) -> None:
        self.switch_history.append(
            SwitchDecisionRecord(
                switch_index=switch_index,
                from_stage=from_stage,
                to_stage=to_stage,
                switch_recommended=switch_recommended,
                switch_executed=switch_executed,
                reason=reason,
                planned_switch_condition=planned_switch_condition,
                observed_switch_cause=observed_switch_cause,
                switch_trigger=observed_switch_cause,
                defect_class=defect_class,
                repair_target=repair_target,
                contract_delta=contract_delta,
                state_delta=state_delta,
                reentry_allowed=reentry_allowed,
            )
        )

    def record_policy_event(self, event: PolicyEvent) -> None:
        self.policy_events.append(event)

    def apply_dominant_frame(self, dominant_frame: Optional[str]) -> None:
        self.dominant_frame = dominant_frame

    def update_inference_state(
        self,
        *,
        contradictions: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
        substantive_assumptions: Optional[List[str]] = None,
    ) -> None:
        if contradictions is not None:
            self.contradictions = contradictions
        if assumptions is not None:
            self.assumptions = assumptions
        if substantive_assumptions is not None:
            self.substantive_assumptions = substantive_assumptions
        if uncertainties is not None:
            self.uncertainties = uncertainties

    def resolve_regime(self, stage: Stage, compose_fn: Callable[[Stage], Regime]) -> Regime:
        if self.current_regime.stage == stage:
            return self.current_regime
        if self.runner_up_regime and self.runner_up_regime.stage == stage:
            return self.runner_up_regime
        if self.recommended_next_regime and self.recommended_next_regime.stage == stage:
            return self.recommended_next_regime
        return compose_fn(stage)


@dataclass
class Handoff:
    current_bottleneck: str
    dominant_frame: str
    what_is_known: List[str]
    what_remains_uncertain: List[str]
    active_contradictions: List[str]
    assumptions_in_play: List[str]
    main_risk_if_continue: str
    recommended_next_regime: Optional[Stage]
    minimum_useful_artifact: str
    prior_artifact_summary: str = ""
    recommended_next_regime_full: Optional[Regime] = None
    source_stage: Optional[Stage] = None
    source_regime_name: str = ""
    created_from: str = ""  # one of: "initial_run", "switch", "repair", "fallback"
    stable_elements: List[str] = field(default_factory=list)
    tentative_elements: List[str] = field(default_factory=list)
    broken_elements: List[str] = field(default_factory=list)
    do_not_relitigate: List[str] = field(default_factory=list)


@dataclass
class SessionRecord:
    timestamp_utc: str
    task: str
    risk_profile: List[str]
    model: str
    routing: Dict[str, object]
    regime: Dict[str, object]
    result: Dict[str, object]
    handoff: Dict[str, object]
    orchestration: Dict[str, object]
    router_state: Optional[Dict[str, object]] = None


def to_jsonable(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def make_record(
    task: str,
    risk_profile: Set[str],
    model: str,
    routing: RoutingDecision,
    regime: Regime,
    result: RegimeExecutionResult,
    handoff: Handoff,
    router_state: Optional[RouterState] = None,
    bounded_orchestration: bool = False,
    max_switches: int = 2,
) -> SessionRecord:
    orchestration: Dict[str, object] = {
        "bounded_orchestration": bounded_orchestration,
        "max_switches": max_switches,
        "switches_attempted": 0,
        "switches_executed": 0,
        "switch_history": [],
        "final_current_regime": to_jsonable(regime.stage),
        "execution_stages": [to_jsonable(regime.stage)],
        "stop_reason": "single_step_mode" if not bounded_orchestration else "unknown",
    }
    if router_state is not None:
        orchestration.update(
            {
                "bounded_orchestration": router_state.orchestration_enabled,
                "max_switches": router_state.max_switches,
                "switches_attempted": router_state.switches_attempted,
                "switches_executed": router_state.switches_executed,
                "switch_history": to_jsonable(router_state.switch_history),
                "final_current_regime": to_jsonable(router_state.current_regime.stage),
                "execution_stages": to_jsonable(router_state.executed_regime_stages),
                "stop_reason": router_state.orchestration_stop_reason,
            }
        )
    return SessionRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        task=task,
        risk_profile=sorted(risk_profile),
        model=model,
        routing=to_jsonable(routing),
        regime=to_jsonable(regime),
        result=to_jsonable(result),
        handoff=to_jsonable(handoff),
        orchestration=orchestration,
        router_state=to_jsonable(router_state) if router_state else None,
    )


def _stage_from_value(value: object) -> Optional[Stage]:
    if isinstance(value, Stage):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in Stage._value2member_map_:
            return Stage(normalized)
    return None


def _line_from_payload(payload: object) -> Optional[LinePrimitive]:
    if isinstance(payload, LinePrimitive):
        return payload
    if not isinstance(payload, Mapping):
        return None
    stage = _stage_from_value(payload.get("stage"))
    function_raw = payload.get("function")
    function = (
        function_raw
        if isinstance(function_raw, FunctionType)
        else FunctionType(str(function_raw).strip().lower())
        if isinstance(function_raw, str) and str(function_raw).strip().lower() in FunctionType._value2member_map_
        else None
    )
    line_id = payload.get("id")
    text = payload.get("text")
    attractor = payload.get("attractor")
    if not stage or not function or not isinstance(line_id, str) or not isinstance(text, str) or not isinstance(attractor, str):
        return None
    return LinePrimitive(
        id=line_id,
        text=text,
        stage=stage,
        function=function,
        attractor=attractor,
        suppresses=tuple(str(v) for v in payload.get("suppresses", []) if isinstance(v, str)),
        tension=str(payload.get("tension", "")),
        risks=tuple(str(v) for v in payload.get("risks", []) if isinstance(v, str)),
        compatible_with=tuple(str(v) for v in payload.get("compatible_with", []) if isinstance(v, str)),
        incompatible_with=tuple(str(v) for v in payload.get("incompatible_with", []) if isinstance(v, str)),
    )


def _regime_from_payload(payload: object, resolve_stage: Callable[[Stage], Regime]) -> Optional[Regime]:
    if isinstance(payload, Regime):
        return payload
    if isinstance(payload, str):
        stage = _stage_from_value(payload)
        return resolve_stage(stage) if stage else None
    if not isinstance(payload, Mapping):
        return None

    stage = _stage_from_value(payload.get("stage"))
    if not stage:
        return None

    dominant_line = _line_from_payload(payload.get("dominant_line"))
    if dominant_line is None:
        # Backward-compatibility adapter: some old records only persisted stage/name.
        return resolve_stage(stage)

    suppression_lines = [
        parsed
        for parsed in (_line_from_payload(item) for item in payload.get("suppression_lines", []))
        if parsed is not None
    ]
    shape_lines = [
        parsed
        for parsed in (_line_from_payload(item) for item in payload.get("shape_lines", []))
        if parsed is not None
    ]
    tail_line = _line_from_payload(payload.get("tail_line"))

    name = payload.get("name")
    return Regime(
        name=name if isinstance(name, str) and name.strip() else resolve_stage(stage).name,
        stage=stage,
        dominant_line=dominant_line,
        suppression_lines=suppression_lines,
        shape_lines=shape_lines,
        tail_line=tail_line,
        rejected_lines=[str(v) for v in payload.get("rejected_lines", []) if isinstance(v, str)],
        rejection_reasons=[str(v) for v in payload.get("rejection_reasons", []) if isinstance(v, str)],
        likely_failure_if_overused=str(payload.get("likely_failure_if_overused", "")),
    )


def _regime_confidence_from_payload(payload: object) -> RegimeConfidenceResult:
    if isinstance(payload, RegimeConfidenceResult):
        return payload
    base = RegimeConfidenceResult.low_default()
    if not isinstance(payload, Mapping):
        return base
    return RegimeConfidenceResult(
        level=str(payload.get("level", base.level)),
        rationale=str(payload.get("rationale", base.rationale)),
        top_stage_score=int(payload.get("top_stage_score", base.top_stage_score)),
        runner_up_score=int(payload.get("runner_up_score", base.runner_up_score)),
        score_gap=int(payload.get("score_gap", base.score_gap)),
        nontrivial_stage_count=int(payload.get("nontrivial_stage_count", base.nontrivial_stage_count)),
        weak_lexical_dependence=bool(payload.get("weak_lexical_dependence", base.weak_lexical_dependence)),
        structural_feature_state=str(payload.get("structural_feature_state", base.structural_feature_state)),
    )


def _reentry_justification_from_payload(payload: object) -> Optional[ReentryJustification]:
    if isinstance(payload, ReentryJustification):
        return payload
    if not isinstance(payload, Mapping):
        return None
    defect_class = payload.get("defect_class")
    repair_target = payload.get("repair_target")
    contract_delta = payload.get("contract_delta")
    state_delta = payload.get("state_delta")
    if not all(isinstance(v, str) for v in (defect_class, repair_target, contract_delta, state_delta)):
        return None
    return ReentryJustification(
        defect_class=defect_class,
        repair_target=repair_target,
        contract_delta=contract_delta,
        state_delta=state_delta,
    )


def router_state_from_jsonable(payload: object, resolve_stage: Callable[[Stage], Regime]) -> Optional[RouterState]:
    if payload is None:
        return None
    if isinstance(payload, RouterState):
        return payload
    if not isinstance(payload, Mapping):
        return None

    current_regime = _regime_from_payload(payload.get("current_regime"), resolve_stage)
    if current_regime is None:
        return None
    runner_up_regime = _regime_from_payload(payload.get("runner_up_regime"), resolve_stage)
    recommended_next_regime = _regime_from_payload(payload.get("recommended_next_regime"), resolve_stage)

    prior_regimes: List[RegimeStep] = []
    for item in payload.get("prior_regimes", []):
        if not isinstance(item, Mapping):
            continue
        prior_regime = _regime_from_payload(item.get("regime"), resolve_stage)
        if prior_regime is None:
            continue
        prior_regimes.append(
            RegimeStep(
                regime=prior_regime,
                reason_entered=str(item.get("reason_entered", "")),
                completion_signal_seen=bool(item.get("completion_signal_seen", False)),
                failure_signal_seen=bool(item.get("failure_signal_seen", False)),
                outcome_summary=str(item.get("outcome_summary", "")),
            )
        )

    executed_regime_stages = [
        stage
        for stage in (_stage_from_value(item) for item in payload.get("executed_regime_stages", []))
        if stage is not None
    ]

    switch_history: List[SwitchDecisionRecord] = []
    for item in payload.get("switch_history", []):
        if not isinstance(item, Mapping):
            continue
        from_stage = _stage_from_value(item.get("from_stage"))
        if from_stage is None:
            continue
        to_stage = _stage_from_value(item.get("to_stage"))
        switch_history.append(
            SwitchDecisionRecord(
                switch_index=int(item.get("switch_index", 0)),
                from_stage=from_stage,
                to_stage=to_stage,
                switch_recommended=bool(item.get("switch_recommended", False)),
                switch_executed=bool(item.get("switch_executed", False)),
                reason=str(item.get("reason", "")),
                planned_switch_condition=(
                    str(item.get("planned_switch_condition"))
                    if item.get("planned_switch_condition") is not None
                    else str(payload.get("planned_switch_condition"))
                    if payload.get("planned_switch_condition") is not None
                    else str(payload.get("switch_trigger"))
                    if payload.get("switch_trigger") is not None
                    else None
                ),
                observed_switch_cause=(
                    str(item.get("observed_switch_cause"))
                    if item.get("observed_switch_cause") is not None
                    else str(item.get("switch_trigger"))
                    if item.get("switch_trigger") is not None
                    else None
                ),
                switch_trigger=(
                    str(item.get("switch_trigger"))
                    if item.get("switch_trigger") is not None
                    else str(item.get("observed_switch_cause"))
                    if item.get("observed_switch_cause") is not None
                    else None
                ),
                defect_class=str(item.get("defect_class")) if item.get("defect_class") is not None else None,
                repair_target=str(item.get("repair_target")) if item.get("repair_target") is not None else None,
                contract_delta=str(item.get("contract_delta")) if item.get("contract_delta") is not None else None,
                state_delta=str(item.get("state_delta")) if item.get("state_delta") is not None else None,
                reentry_allowed=(
                    bool(item.get("reentry_allowed")) if item.get("reentry_allowed") is not None else None
                ),
            )
        )

    policy_events: List[PolicyEvent] = []
    for item in payload.get("policy_events", []):
        if not isinstance(item, Mapping):
            continue
        rule_name = item.get("rule_name")
        authority = item.get("authority")
        action = item.get("action")
        detail = item.get("detail")
        consumed_features = [str(v) for v in item.get("consumed_features", []) if isinstance(v, str)]
        if not all(isinstance(v, str) for v in (rule_name, authority, action, detail)):
            continue
        policy_events.append(
            PolicyEvent(
                rule_name=rule_name,
                authority=authority,
                consumed_features=consumed_features,
                action=action,
                detail=detail,
            )
        )

    last_reentry_justification = _reentry_justification_from_payload(payload.get("last_reentry_justification"))

    assumptions = [str(v) for v in payload.get("assumptions", []) if isinstance(v, str)]
    legacy_structural_signals: List[str] = []
    filtered_assumptions: List[str] = []
    legacy_prefix = "Structural signals observed:"
    for assumption in assumptions:
        if not assumption.startswith(legacy_prefix):
            filtered_assumptions.append(assumption)
            continue
        observed_signals = assumption[len(legacy_prefix) :].strip()
        if not observed_signals:
            continue
        legacy_structural_signals.extend(
            signal.strip() for signal in observed_signals.split(",") if signal.strip()
        )

    payload_structural_signals = payload.get("structural_signals", [])
    structural_signals = (
        [str(v) for v in payload_structural_signals if isinstance(v, str)]
        if isinstance(payload_structural_signals, list)
        else []
    )
    if not structural_signals and legacy_structural_signals:
        structural_signals = legacy_structural_signals

    return RouterState(
        task_id=str(payload.get("task_id", "")),
        task_summary=str(payload.get("task_summary", "")),
        current_bottleneck=str(payload.get("current_bottleneck", "")),
        current_regime=current_regime,
        runner_up_regime=runner_up_regime,
        regime_confidence=_regime_confidence_from_payload(payload.get("regime_confidence")),
        dominant_frame=str(payload.get("dominant_frame")) if payload.get("dominant_frame") is not None else None,
        knowns=[str(v) for v in payload.get("knowns", []) if isinstance(v, str)],
        uncertainties=[str(v) for v in payload.get("uncertainties", []) if isinstance(v, str)],
        contradictions=[str(v) for v in payload.get("contradictions", []) if isinstance(v, str)],
        assumptions=filtered_assumptions,
        substantive_assumptions=[
            str(v)
            for v in payload.get("substantive_assumptions", filtered_assumptions)
            if isinstance(v, str) and not str(v).startswith(legacy_prefix)
        ],
        risks=[str(v) for v in payload.get("risks", []) if isinstance(v, str)],
        stage_goal=str(payload.get("stage_goal", "")),
        planned_switch_condition=(
            str(payload.get("planned_switch_condition"))
            if payload.get("planned_switch_condition") is not None
            else str(payload.get("switch_trigger"))
            if payload.get("switch_trigger") is not None
            else None
        ),
        observed_switch_cause=(
            str(payload.get("observed_switch_cause"))
            if payload.get("observed_switch_cause") is not None
            else str(payload.get("switch_trigger"))
            if payload.get("switch_trigger") is not None
            else None
        ),
        switch_trigger=str(payload.get("switch_trigger")) if payload.get("switch_trigger") is not None else None,
        recommended_next_regime=recommended_next_regime,
        decision_pressure=float(payload.get("decision_pressure", 0.0)),
        fragility_pressure=float(payload.get("fragility_pressure", 0.0)),
        possibility_space_need=float(payload.get("possibility_space_need", 0.0)),
        synthesis_pressure=float(payload.get("synthesis_pressure", 0.0)),
        detected_markers=(
            dict(payload.get("detected_markers", {})) if isinstance(payload.get("detected_markers"), dict) else {}
        ),
        structural_signals=structural_signals,
        risk_tags={str(v) for v in payload.get("risk_tags", []) if isinstance(v, str)},
        evidence_demand=float(payload.get("evidence_demand", 0.0)),
        evidence_quality=float(payload.get("evidence_quality", 0.0)),
        recurrence_potential=float(payload.get("recurrence_potential", 0.0)),
        prior_regimes=prior_regimes,
        orchestration_enabled=bool(payload.get("orchestration_enabled", False)),
        max_switches=int(payload.get("max_switches", 0)),
        switches_attempted=int(payload.get("switches_attempted", 0)),
        switches_executed=int(payload.get("switches_executed", 0)),
        collapse_reentries=int(payload.get("collapse_reentries", 0)),
        orchestration_stop_reason=(
            str(payload.get("orchestration_stop_reason")) if payload.get("orchestration_stop_reason") is not None else None
        ),
        executed_regime_stages=executed_regime_stages,
        switch_history=switch_history,
        policy_events=policy_events,
        last_reentry_justification=last_reentry_justification,
        last_state_delta=str(payload.get("last_state_delta")) if payload.get("last_state_delta") is not None else None,
        last_contract_delta=(
            str(payload.get("last_contract_delta")) if payload.get("last_contract_delta") is not None else None
        ),
        task_classification=(
            payload.get("task_classification") if isinstance(payload.get("task_classification"), Mapping) else None
        ),
        latest_forward_handoff=None,
    )
