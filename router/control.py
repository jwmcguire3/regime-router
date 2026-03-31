from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Regime, RegimeConfidenceResult, RoutingFeatures, Severity, Stage
from .routing import RegimeComposer
from .evolution.revision_engine import EvolutionEngine
from .state import RouterState


from .orchestration.misrouting_detector import MisroutingDetectionResult, MisroutingDetector
from .orchestration.output_contract import RegimeOutputContract


@dataclass(frozen=True)
class EscalationPolicyResult:
    escalation_direction: str
    justification: str
    preferred_regime_biases: Dict[Stage, int]
    switch_pressure_adjustment: int
    debug_signals: List[str] = field(default_factory=list)


class EscalationPolicy:
    _STRICTER_CUES = ("certainty", "confident", "confidence", "prove", "proof", "guarantee", "certain")
    _LOOSER_CUES = (
        "brainstorm",
        "underformed",
        "map the space",
        "keep it open",
        "before narrowing",
        "premature narrowing",
        "narrowing happened too early",
    )

    def evaluate(
        self,
        *,
        state: Optional[RouterState],
        routing_features: RoutingFeatures,
        task_text: str,
        current_regime: Optional[Regime],
        regime_confidence: Optional[RegimeConfidenceResult],
        misrouting_result: Optional[MisroutingDetectionResult],
    ) -> EscalationPolicyResult:
        text = task_text.lower().replace("’", "'")
        strict_score = 0
        loose_score = 0
        strict_biases: Dict[Stage, int] = {}
        loose_biases: Dict[Stage, int] = {}
        cues: List[str] = []

        if routing_features.fragility_pressure >= 2 or any(k in text for k in ("high stakes", "irreversible", "safety", "production", "deployment", "trust")):
            strict_score += 2
            cues.append("high_consequence_or_deployment")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 1
            strict_biases[Stage.ADVERSARIAL] = strict_biases.get(Stage.ADVERSARIAL, 0) + 1

        contradiction_count = len(state.contradictions) if state else 0
        if contradiction_count >= 2:
            strict_score += 2
            cues.append("contradiction_accumulation")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 2

        if any(cue in text for cue in self._STRICTER_CUES):
            strict_score += 2
            cues.append("certainty_or_proof_request")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 2

        if routing_features.possibility_space_need >= 3 and any(cue in text for cue in ("underformed", "lack of structure", "map the space")):
            loose_score += 2
            cues.append("underformed_space")
            loose_biases[Stage.EXPLORATION] = loose_biases.get(Stage.EXPLORATION, 0) + 2

        if any(cue in text for cue in self._LOOSER_CUES):
            loose_score += 2
            cues.append("premature_narrowing_signal")
            loose_biases[Stage.EXPLORATION] = loose_biases.get(Stage.EXPLORATION, 0) + 1

        if "lack of structure" in text or "can't characterize" in text or "cannot characterize" in text:
            loose_score += 1
            cues.append("lack_of_possible_structure")
            loose_biases[Stage.SYNTHESIS] = loose_biases.get(Stage.SYNTHESIS, 0) + 1

        if misrouting_result and misrouting_result.misrouting_detected:
            strict_score += 1
            cues.append("misrouting_detected")

        if regime_confidence and regime_confidence.level == Severity.LOW.value and strict_score > 0:
            strict_score += 1
            cues.append("low_confidence_requires_stricter_grounding")

        direction = "none"
        switch_adjust = 0
        biases: Dict[Stage, int] = {}
        if strict_score >= loose_score + 2:
            direction = "stricter"
            switch_adjust = min(3, strict_score - loose_score)
            biases = strict_biases
            justification = "Escalate toward stricter regimes due to consequence/contradiction/certainty signals."
        elif loose_score >= strict_score + 2:
            direction = "looser"
            switch_adjust = -min(3, loose_score - strict_score)
            biases = loose_biases
            justification = "Relax toward looser regimes due to underformed-space or premature narrowing signals."
        else:
            justification = "Escalation policy is neutral; deterministic routing and orchestration remain primary."

        if current_regime and direction == "stricter" and current_regime.stage == Stage.EXPLORATION:
            biases[Stage.SYNTHESIS] = biases.get(Stage.SYNTHESIS, 0) + 1
        if current_regime and direction == "looser" and current_regime.stage in {Stage.OPERATOR, Stage.BUILDER}:
            biases[Stage.EXPLORATION] = biases.get(Stage.EXPLORATION, 0) + 1

        return EscalationPolicyResult(
            escalation_direction=direction,
            justification=justification,
            preferred_regime_biases=biases,
            switch_pressure_adjustment=switch_adjust,
            debug_signals=cues,
        )




@dataclass(frozen=True)
class SwitchOrchestrationResult:
    next_regime: Optional[Regime]
    switch_recommended_now: bool
    reason_for_switch: str
    updated_state: RouterState


class SwitchOrchestrator:
    _ALLOWED_PATHWAYS = {
        Stage.EXPLORATION: {Stage.SYNTHESIS},
        Stage.SYNTHESIS: {Stage.EPISTEMIC, Stage.ADVERSARIAL},
        Stage.EPISTEMIC: {Stage.OPERATOR},
        Stage.ADVERSARIAL: {Stage.OPERATOR},
        Stage.OPERATOR: {Stage.EPISTEMIC, Stage.BUILDER},
        Stage.BUILDER: set(),
    }

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
        completion_signal = self._signal_from_output(output, key="completion_signal")
        failure_signal = self._signal_from_output(output, key="failure_signal")
        semantic_operator_failure = self._operator_semantic_failure(output)
        bounded = switches_used >= max_switches

        if bounded:
            state.switch_trigger = "switch_bound_reached"
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Switch limit reached; execution remains on current regime.",
                updated_state=state,
            )

        if self._assumption_or_frame_collapse(state, failure_signal):
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

        next_stage = self._next_stage(
            state,
            completion_signal,
            failure_signal,
            detection,
            escalation,
            semantic_operator_failure=semantic_operator_failure,
        )
        if next_stage is None:
            return SwitchOrchestrationResult(
                next_regime=None,
                switch_recommended_now=False,
                reason_for_switch="Current regime should continue; no allowed transition selected.",
                updated_state=state,
            )

        next_regime = self._resolve_stage(state, next_stage)
        state.recommended_next_regime = next_regime
        state.switch_trigger = (
            "semantic_validation_failed_in_operator"
            if semantic_operator_failure
            else failure_signal or completion_signal or "misrouting_detected"
        )
        return SwitchOrchestrationResult(
            next_regime=next_regime,
            switch_recommended_now=True,
            reason_for_switch=f"Switching from {state.current_regime.stage.value} to {next_stage.value}.",
            updated_state=state,
        )

    def _next_stage(
        self,
        state: RouterState,
        completion_signal: str,
        failure_signal: str,
        detection: MisroutingDetectionResult,
        escalation: Optional[EscalationPolicyResult],
        *,
        semantic_operator_failure: bool = False,
    ) -> Optional[Stage]:
        current_stage = state.current_regime.stage
        allowed = self._ALLOWED_PATHWAYS.get(current_stage, set())
        recommended = detection.recommended_next_stage
        if detection.misrouting_detected and recommended in allowed:
            if escalation and escalation.escalation_direction == "looser" and escalation.switch_pressure_adjustment <= -2:
                # Light-touch damping: do not override explicit completion/failure pathways, only soft misrouting nudges.
                return None
            return recommended

        if current_stage == Stage.EXPLORATION and completion_signal:
            return Stage.SYNTHESIS
        if current_stage == Stage.SYNTHESIS and failure_signal:
            if recommended == Stage.ADVERSARIAL and Stage.ADVERSARIAL in allowed:
                return Stage.ADVERSARIAL
            return Stage.EPISTEMIC if Stage.EPISTEMIC in allowed else None
        if current_stage == Stage.OPERATOR and (semantic_operator_failure or (failure_signal and detection.misrouting_detected)):
            if recommended == Stage.EPISTEMIC and Stage.EPISTEMIC in allowed:
                return Stage.EPISTEMIC
            return Stage.EPISTEMIC if Stage.EPISTEMIC in allowed else None
        if current_stage in {Stage.EPISTEMIC, Stage.ADVERSARIAL} and completion_signal:
            return Stage.OPERATOR if Stage.OPERATOR in allowed else None
        if current_stage == Stage.OPERATOR and completion_signal and state.recurrence_potential >= 2.0:
            return Stage.BUILDER if Stage.BUILDER in allowed else None
        if (
            escalation
            and escalation.escalation_direction == "stricter"
            and escalation.switch_pressure_adjustment >= 2
            and detection.recommended_next_stage in allowed
        ):
            return detection.recommended_next_stage
        return None

    def _resolve_stage(self, state: RouterState, stage: Stage) -> Regime:
        return state.resolve_regime(stage, self._composer.compose)

    def _signal_from_output(self, output: RegimeOutputContract, *, key: str) -> str:
        parsed = output.validation.get("parsed", {})
        if isinstance(parsed, dict):
            value = parsed.get(key)
            if isinstance(value, str):
                return value.strip()
        return ""

    def _assumption_or_frame_collapse(self, state: RouterState, failure_signal: str) -> bool:
        normalized = " ".join(failure_signal.lower().split())
        assumption_collapse_signaled = "assumption" in normalized and "collapse" in normalized
        frame_collapse_signaled = "frame" in normalized and "collapse" in normalized and "pressure" not in normalized
        if not (assumption_collapse_signaled or frame_collapse_signaled):
            return False
        return bool(state.assumptions) and (bool(state.contradictions) or assumption_collapse_signaled)

    def _operator_semantic_failure(self, output: RegimeOutputContract) -> bool:
        if output.stage != Stage.OPERATOR:
            return False
        validation = output.validation
        semantic_failures = validation.get("semantic_failures", [])
        if not semantic_failures:
            return False
        return (
            bool(validation.get("valid_json", False))
            and bool(validation.get("required_keys_present", False))
            and bool(validation.get("artifact_fields_present", False))
            and bool(validation.get("artifact_type_matches", False))
            and bool(validation.get("contract_controls_valid", False))
            and not bool(validation.get("semantic_valid", True))
        )


# ============================================================
# Runtime
