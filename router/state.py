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
class SessionRecord:
    timestamp_utc: str
    task: str
    risk_profile: List[str]
    model: str
    routing: Dict[str, object]
    regime: Dict[str, object]
    result: Dict[str, object]
    handoff: Dict[str, object]


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
    )
