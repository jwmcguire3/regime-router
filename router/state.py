from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

from .models import Regime, RegimeConfidenceResult, RegimeExecutionResult, RoutingDecision, Stage

RegimeConfidence = RegimeConfidenceResult


@dataclass
class RegimeStep:
    regime: Stage
    reason_entered: str
    completion_signal_seen: bool
    failure_signal_seen: bool
    outcome_summary: str


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
    switch_trigger: Optional[str]
    recommended_next_regime: Optional[Regime]
    decision_pressure: float
    evidence_quality: float
    recurrence_potential: float
    prior_regimes: List[RegimeStep] = field(default_factory=list)

    def record_regime_step(
        self,
        *,
        regime: Stage,
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

    def apply_dominant_frame(self, dominant_frame: Optional[str]) -> None:
        self.dominant_frame = dominant_frame

    def update_inference_state(
        self,
        *,
        contradictions: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        uncertainties: Optional[List[str]] = None,
    ) -> None:
        if contradictions is not None:
            self.contradictions = contradictions
        if assumptions is not None:
            self.assumptions = assumptions
        if uncertainties is not None:
            self.uncertainties = uncertainties


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
) -> SessionRecord:
    return SessionRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        task=task,
        risk_profile=sorted(risk_profile),
        model=model,
        routing=to_jsonable(routing),
        regime=to_jsonable(regime),
        result=to_jsonable(result),
        handoff=to_jsonable(handoff),
        router_state=to_jsonable(router_state) if router_state else None,
    )
