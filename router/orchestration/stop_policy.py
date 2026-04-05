from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..models import ARTIFACT_HINTS, RoutingDecision, Stage
from ..state import RouterState
from .collapse_detector import CollapseDetector
from .misrouting_rules import failure_signal_active

STAGE_PROGRESSION = [
    Stage.EXPLORATION,
    Stage.SYNTHESIS,
    Stage.EPISTEMIC,
    Stage.ADVERSARIAL,
    Stage.OPERATOR,
    Stage.BUILDER,
]
_STAGE_RANK = {stage: idx for idx, stage in enumerate(STAGE_PROGRESSION)}
INTERMEDIATE_STAGE_ARTIFACT = {
    Stage.EXPLORATION: "candidate_frame_set",
    Stage.SYNTHESIS: "dominant_frame",
    Stage.OPERATOR: "decision_packet",
}
_EXPLICIT_DELIVERABLE_TOKENS = (
    "final",
    "final deliverable",
    "finished",
    "completed",
    "not the final output",
)
_CONCRETE_DELIVERABLE_TERMS = (
    "memo",
    "worksheet",
    "framework document",
    "spec",
    "specification",
    "document",
)


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

        artifact_complete = self._artifact_complete(router_state, validation_result, current_stage)
        if not artifact_complete:
            return StopDecision(should_stop=False, reason="artifact_incomplete")
        artifact_matches_current_stage = self._artifact_matches_current_stage(validation_result, current_stage)
        if not artifact_matches_current_stage:
            return StopDecision(should_stop=False, reason="artifact_complete_but_stage_artifact_mismatch")
        deliverable_pressure = self._requested_deliverable_pressure(router_state)
        if deliverable_pressure and current_stage in INTERMEDIATE_STAGE_ARTIFACT:
            return StopDecision(
                should_stop=False,
                reason=(
                    "deliverable_pressure_unsatisfied:"
                    f"requested={deliverable_pressure};artifact={INTERMEDIATE_STAGE_ARTIFACT[current_stage]}"
                ),
            )

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
        if _STAGE_RANK[next_stage] > _STAGE_RANK[current_stage]:
            return _STAGE_RANK[next_stage] <= _STAGE_RANK[endpoint]
        return self._has_qualified_reentry_justification(state)

    def _artifact_complete(
        self,
        state: RouterState,
        validation_result: Mapping[str, object],
        current_stage: Stage,
    ) -> bool:
        if not bool(validation_result.get("is_valid", False)):
            return False
        parsed = validation_result.get("parsed", {})
        if not isinstance(parsed, dict):
            return False
        completion_signal = str(parsed.get("completion_signal", "")).strip()
        if not completion_signal:
            return False
        artifact = parsed.get("artifact", {})
        if not isinstance(artifact, dict):
            return False
        if artifact and failure_signal_active(current_stage, state, artifact):
            return False
        return True

    def _artifact_matches_current_stage(self, validation_result: Mapping[str, object], current_stage: Stage) -> bool:
        parsed = validation_result.get("parsed", {})
        if not isinstance(parsed, Mapping):
            return False
        artifact_type = parsed.get("artifact_type")
        regime = parsed.get("regime")
        if not isinstance(artifact_type, str) or not artifact_type.strip():
            return False
        if not isinstance(regime, str) or not regime.strip():
            return False
        expected_artifact_type = ARTIFACT_HINTS[current_stage]
        return artifact_type.strip() == expected_artifact_type and regime.strip().lower() == current_stage.value

    def _has_qualified_reentry_justification(self, state: RouterState) -> bool:
        justification = state.last_reentry_justification
        if justification is None:
            return False
        fields = (
            justification.defect_class,
            justification.repair_target,
            justification.contract_delta,
            justification.state_delta,
        )
        return all(isinstance(value, str) and bool(value.strip()) for value in fields)

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

    def _requested_deliverable_pressure(self, state: RouterState) -> str:
        summary = (state.task_summary or "").strip().lower()
        if not summary:
            return ""
        has_explicit_pressure = any(token in summary for token in _EXPLICIT_DELIVERABLE_TOKENS)
        requested_terms = [term for term in _CONCRETE_DELIVERABLE_TERMS if term in summary]
        if not requested_terms:
            return ""
        if not has_explicit_pressure and "deliver" not in summary and "provide" not in summary and "return" not in summary:
            return ""
        return requested_terms[0]
