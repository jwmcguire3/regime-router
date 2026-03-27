from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set

from .models import Regime, RegimeExecutionResult, RoutingDecision, Stage


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
class RouterState:
    task_id: str
    task_summary: str
    current_bottleneck: str
    current_regime: Stage
    runner_up_regime: Optional[Stage]
    regime_confidence: str
    stage_goal: str
    knowns: List[str]
    uncertainties: List[str]
    contradictions: List[str]
    assumptions: List[str]
    risks: List[str]
    decision_pressure: int
    evidence_quality: int
    recurrence_potential: int
    prior_regimes: List[Stage]
    switch_trigger: str
    recommended_next_regime: Optional[Stage]


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
        router_state=to_jsonable(router_state) if router_state is not None else None,
    )
