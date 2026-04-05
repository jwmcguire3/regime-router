from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..models import RoutingDecision, Stage
from ..state import RouterState
from .collapse_detector import CollapseDetector

BUILDER_RECURRENCE_THRESHOLD = 7
STAGE_PROGRESSION = [
    Stage.EXPLORATION,
    Stage.SYNTHESIS,
    Stage.EPISTEMIC,
    Stage.ADVERSARIAL,
    Stage.OPERATOR,
    Stage.BUILDER,
]
_STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_PROGRESSION)}


@dataclass(frozen=True)
class StopDecision:
    should_stop: bool
    reason: str


class StopPolicy:
    def __init__(self, collapse_detector: Optional[CollapseDetector] = None) -> None:
        self._collapse_detector = collapse_detector or CollapseDetector()

    def should_stop(
        self,
        router_state: RouterState,
        validation_result: Mapping[str, object],
        routing_decision: Optional[RoutingDecision],
        current_stage: Stage,
    ) -> StopDecision:
        if self._collapse_signal_present(router_state, validation_result):
            return StopDecision(should_stop=False, reason="collapse_override_active")

        if self._builder_blocked(router_state):
            recurrence_label = self._format_recurrence(router_state.recurrence_potential)
            return StopDecision(
                should_stop=True,
                reason=f"Builder blocked: recurrence_potential {recurrence_label} < {BUILDER_RECURRENCE_THRESHOLD}",
            )

        artifact_complete = self._artifact_complete(validation_result)
        if not artifact_complete:
            return StopDecision(should_stop=False, reason="artifact_incomplete")

        endpoint = self._endpoint_stage(router_state, routing_decision)
        at_or_past_endpoint = _STAGE_RANK[current_stage] >= _STAGE_RANK[endpoint]
        operator_default = endpoint == Stage.OPERATOR and current_stage == Stage.OPERATOR
        if self._should_defer_stop_for_forward_recommendation(router_state, current_stage, endpoint):
            return StopDecision(should_stop=False, reason="forward_progress_recommended")
        if (artifact_complete and at_or_past_endpoint) or (artifact_complete and operator_default):
            return StopDecision(should_stop=True, reason=f"artifact_complete_at_or_past_endpoint:{endpoint.value}")
        return StopDecision(should_stop=False, reason="endpoint_not_reached")

    def _collapse_signal_present(
        self,
        state: RouterState,
        validation_result: Mapping[str, object],
    ) -> bool:
        parsed = validation_result.get("parsed", {})
        if not isinstance(parsed, dict):
            return False
        artifact = parsed.get("artifact", {})
        artifact_mapping: Mapping[str, object] = artifact if isinstance(artifact, dict) else {}
        failure_signal = str(parsed.get("failure_signal", "")).strip()
        detection = self._collapse_detector.detect(state, validation_result, artifact_mapping, failure_signal)
        return detection.collapse_detected

    def _should_defer_stop_for_forward_recommendation(
        self,
        state: RouterState,
        current_stage: Stage,
        endpoint: Stage,
    ) -> bool:
        next_regime = state.recommended_next_regime
        if next_regime is None:
            return False
        next_stage = next_regime.stage
        if next_stage == current_stage:
            return False
        if _STAGE_RANK[next_stage] <= _STAGE_RANK[current_stage]:
            return False
        return _STAGE_RANK[next_stage] <= _STAGE_RANK[endpoint]

    def _builder_blocked(self, state: RouterState) -> bool:
        if state.current_regime.stage != Stage.OPERATOR:
            return False
        if state.recommended_next_regime is None or state.recommended_next_regime.stage != Stage.BUILDER:
            return False
        return state.recurrence_potential < BUILDER_RECURRENCE_THRESHOLD

    def _artifact_complete(self, validation_result: Mapping[str, object]) -> bool:
        if not bool(validation_result.get("is_valid", False)):
            return False
        parsed = validation_result.get("parsed", {})
        if not isinstance(parsed, dict):
            return False
        completion_signal = str(parsed.get("completion_signal", "")).strip()
        failure_signal = str(parsed.get("failure_signal", "")).strip()
        if not completion_signal:
            return False
        if failure_signal and completion_signal == failure_signal:
            return False
        return True

    def _endpoint_stage(self, state: RouterState, routing_decision: Optional[RoutingDecision]) -> Stage:
        if routing_decision is not None:
            candidate = routing_decision.likely_endpoint_regime
            if isinstance(candidate, str):
                normalized = candidate.strip().lower()
                if normalized in Stage._value2member_map_:
                    return Stage(normalized)
        if state.task_classification and isinstance(state.task_classification.get("likely_endpoint_regime"), str):
            normalized = str(state.task_classification["likely_endpoint_regime"]).strip().lower()
            if normalized in Stage._value2member_map_:
                return Stage(normalized)
        return Stage.OPERATOR

    def _format_recurrence(self, recurrence_potential: float) -> str:
        as_int = int(recurrence_potential)
        if float(as_int) == float(recurrence_potential):
            return str(as_int)
        return f"{recurrence_potential:.2f}".rstrip("0").rstrip(".")
