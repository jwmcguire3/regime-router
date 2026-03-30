from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import FailureLog, LIBRARY, Regime, RegimeConfidenceResult, RoutingFeatures, RevisionProposal, Severity, Stage
from .routing import RegimeComposer
from .state import RouterState


@dataclass(frozen=True)
class RegimeOutputContract:
    stage: Stage
    raw_response: str
    validation: Dict[str, object]


@dataclass(frozen=True)
class MisroutingDetectionResult:
    current_regime: Regime
    dominant_failure_mode: str
    still_productive: bool
    misrouting_detected: bool
    justification: str
    recommended_next_regime: Optional[Regime]

    @property
    def current_stage(self) -> Stage:
        return self.current_regime.stage

    @property
    def recommended_next_stage(self) -> Optional[Stage]:
        if self.recommended_next_regime is None:
            return None
        return self.recommended_next_regime.stage


class MisroutingDetector:
    JUSTIFICATION_SWITCH = "Current regime is hitting its dominant failure mode. Switching is justified."
    JUSTIFICATION_STAY = "Current regime remains productive. Switching is not justified."
    JUSTIFICATION_INCOMPLETE = "Current regime is incomplete but not in dominant failure mode yet. Switching is not justified."

    _DEFAULT_NEXT_REGIME_BY_STAGE = {
        Stage.EXPLORATION: Stage.SYNTHESIS,
        Stage.SYNTHESIS: Stage.EPISTEMIC,
        Stage.EPISTEMIC: Stage.OPERATOR,
        Stage.ADVERSARIAL: Stage.OPERATOR,
        Stage.OPERATOR: Stage.OPERATOR,
        Stage.BUILDER: Stage.OPERATOR,
    }

    _DOMINANT_FAILURE_MODE_BY_STAGE = {
        Stage.EXPLORATION: "branch sprawl / novelty drift",
        Stage.SYNTHESIS: "false unification",
        Stage.EPISTEMIC: "reportorial drag",
        Stage.ADVERSARIAL: "attack loop",
        Stage.OPERATOR: "forced closure",
        Stage.BUILDER: "premature architecture",
    }

    def __init__(self) -> None:
        self._composer = RegimeComposer()

    def detect(self, state: RouterState, output: RegimeOutputContract) -> MisroutingDetectionResult:
        artifact = self._extract_artifact(output)
        current_regime = state.current_regime
        stage = current_regime.stage
        failure_signal = self._failure_signal_active(stage, state, artifact)
        completion_signal = self._completion_signal_active(stage, state, artifact)
        recommended_next = self._recommended_next_regime(state, artifact, failure_signal)
        if failure_signal:
            justification = self.JUSTIFICATION_SWITCH
        elif completion_signal:
            justification = self.JUSTIFICATION_STAY
        else:
            justification = self.JUSTIFICATION_INCOMPLETE
        return MisroutingDetectionResult(
            current_regime=current_regime,
            dominant_failure_mode=self._DOMINANT_FAILURE_MODE_BY_STAGE[stage],
            still_productive=completion_signal,
            misrouting_detected=failure_signal,
            justification=justification,
            recommended_next_regime=recommended_next,
        )

    def _recommended_next_regime(
        self,
        state: RouterState,
        artifact: Dict[str, object],
        signal_active: bool,
    ) -> Optional[Regime]:
        if not signal_active:
            return None
        if self._assumption_collapse_detected(state, artifact):
            return self._regime_for_stage(state, Stage.EXPLORATION)

        stage = state.current_regime.stage
        if stage == Stage.SYNTHESIS and self._adversarial_needed(artifact):
            return self._regime_for_stage(state, Stage.ADVERSARIAL)
        if stage == Stage.OPERATOR:
            if self._recurrence_established(state):
                return self._regime_for_stage(state, Stage.BUILDER)
            if self._operator_evidence_gap(artifact):
                return self._regime_for_stage(state, Stage.EPISTEMIC)
        return self._regime_for_stage(state, self._DEFAULT_NEXT_REGIME_BY_STAGE[stage])

    def _regime_for_stage(self, state: RouterState, stage: Stage) -> Regime:
        if state.current_regime.stage == stage:
            return state.current_regime
        if state.runner_up_regime and state.runner_up_regime.stage == stage:
            return state.runner_up_regime
        if state.recommended_next_regime and state.recommended_next_regime.stage == stage:
            return state.recommended_next_regime
        return self._composer.compose(stage)

    def _extract_artifact(self, output: RegimeOutputContract) -> Dict[str, object]:
        parsed = output.validation.get("parsed", {})
        if isinstance(parsed, dict):
            artifact = parsed.get("artifact", {})
            if isinstance(artifact, dict):
                return artifact
        try:
            raw = json.loads(output.raw_response)
        except json.JSONDecodeError:
            return {}
        if not isinstance(raw, dict):
            return {}
        artifact = raw.get("artifact", {})
        return artifact if isinstance(artifact, dict) else {}

    def _failure_signal_active(self, stage: Stage, state: RouterState, artifact: Dict[str, object]) -> bool:
        if stage == Stage.EXPLORATION:
            candidate_frames = self._item_count(artifact.get("candidate_frames"))
            has_differentiation = self._exploration_has_differentiation(artifact)
            return candidate_frames >= 5 and not has_differentiation

        if stage == Stage.SYNTHESIS:
            has_central_claim = self._present(artifact.get("central_claim"))
            has_organizing_idea = self._present(artifact.get("organizing_idea"))
            has_support = self._present(artifact.get("supporting_structure"))
            contradictions_live = len(state.contradictions) > 0
            has_pressure_points = self._present(artifact.get("pressure_points"))
            unsupported_unification = has_central_claim and has_organizing_idea and not has_support and not has_pressure_points
            stress_without_structure = has_central_claim and has_organizing_idea and has_pressure_points and not has_support
            contradictions_flattened = contradictions_live and not has_pressure_points and not has_support
            return unsupported_unification or stress_without_structure or contradictions_flattened

        if stage == Stage.EPISTEMIC:
            has_support_separation = self._epistemic_has_support_separation(artifact)
            has_uncertainty_handling = self._epistemic_has_uncertainty_handling(state, artifact)
            return not has_support_separation and not has_uncertainty_handling

        if stage == Stage.ADVERSARIAL:
            destabilizers = artifact.get("top_destabilizers")
            residual_risks = artifact.get("residual_risks")
            same_objections = self._normalized(destabilizers) == self._normalized(residual_risks) and self._present(destabilizers)
            no_revision_movement = not self._present(artifact.get("survivable_revisions"))
            return same_objections or (self._present(destabilizers) and no_revision_movement)

        if stage == Stage.OPERATOR:
            has_decision = self._present(artifact.get("decision"))
            missing_decision = not has_decision
            missing_tradeoff = not self._present(artifact.get("tradeoff_accepted"))
            missing_rationale = not self._present(artifact.get("rationale"))
            missing_next_actions = not self._present(artifact.get("next_actions"))
            missing_fallback = not self._present(artifact.get("fallback_trigger"))
            missing_review_point = not self._present(artifact.get("review_point"))
            assumptions_hidden = self._has_live_assumptions(state) and missing_rationale and missing_fallback
            return (
                missing_decision
                or missing_tradeoff
                or missing_rationale
                or missing_next_actions
                or missing_fallback
                or missing_review_point
                or assumptions_hidden
            )

        if stage == Stage.BUILDER:
            has_modules_or_interfaces = self._present(artifact.get("modules")) or self._present(artifact.get("interfaces"))
            return has_modules_or_interfaces and not self._recurrence_established(state)

        return False

    def _completion_signal_active(self, stage: Stage, state: RouterState, artifact: Dict[str, object]) -> bool:
        if stage == Stage.EXPLORATION:
            candidate_frames = self._item_count(artifact.get("candidate_frames"))
            has_differentiation = self._exploration_has_differentiation(artifact)
            return candidate_frames >= 3 and has_differentiation

        if stage == Stage.SYNTHESIS:
            has_central_pattern = self._present(artifact.get("central_claim")) or self._present(artifact.get("organizing_idea"))
            has_connective_structure = self._present(artifact.get("supporting_structure")) or self._present(artifact.get("pressure_points"))
            if len(state.contradictions) > 0:
                has_connective_structure = has_connective_structure or self._present(artifact.get("contradictions"))
            return has_central_pattern and has_connective_structure

        if stage == Stage.EPISTEMIC:
            has_support_separation = self._epistemic_has_support_separation(artifact)
            has_uncertainty_handling = self._epistemic_has_uncertainty_handling(state, artifact)
            return has_support_separation and has_uncertainty_handling

        return not self._failure_signal_active(stage, state, artifact)

    def _exploration_has_differentiation(self, artifact: Dict[str, object]) -> bool:
        has_selection_criteria = self._present(artifact.get("selection_criteria"))
        has_unresolved_axes = self._present(artifact.get("unresolved_axes"))
        candidate_frames = artifact.get("candidate_frames")
        rich_frames = 0
        if isinstance(candidate_frames, list):
            for frame in candidate_frames:
                if isinstance(frame, str) and len(frame.strip().split()) >= 2:
                    rich_frames += 1
        frame_text = self._normalized(candidate_frames)
        unique_tokens = {token for token in frame_text.split() if len(token) > 2}
        has_frame_diversity = len(unique_tokens) >= 6 and rich_frames >= 2
        return has_selection_criteria or has_unresolved_axes or has_frame_diversity

    def _epistemic_has_support_separation(self, artifact: Dict[str, object]) -> bool:
        has_supported = self._present(artifact.get("supported_claims"))
        has_unproven = self._present(artifact.get("plausible_but_unproven")) or self._present(artifact.get("omitted_due_to_insufficient_support"))
        return has_supported and has_unproven

    def _epistemic_has_uncertainty_handling(self, state: RouterState, artifact: Dict[str, object]) -> bool:
        has_uncertainty_markers = (
            self._present(artifact.get("contradictions"))
            or self._present(artifact.get("omitted_due_to_insufficient_support"))
            or self._present(artifact.get("hidden_assumptions"))
        )
        return has_uncertainty_markers or len(state.assumptions) > 0 or len(state.contradictions) > 0

    def _item_count(self, value: object) -> int:
        if isinstance(value, list):
            return sum(1 for item in value if self._present(item))
        return 0

    def _present(self, value: object) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, list):
            return any(self._present(item) for item in value)
        if isinstance(value, dict):
            return any(self._present(v) for v in value.values())
        return value is not None

    def _normalized(self, value: object) -> str:
        if isinstance(value, list):
            return " | ".join(self._normalized(item) for item in value if self._present(item)).strip()
        if isinstance(value, dict):
            return " | ".join(f"{k}:{self._normalized(v)}" for k, v in sorted(value.items()) if self._present(v)).strip()
        if isinstance(value, str):
            return " ".join(value.lower().split())
        if value is None:
            return ""
        return str(value).strip().lower()

    def _assumption_collapse_detected(self, state: RouterState, artifact: Dict[str, object]) -> bool:
        if not self._has_live_assumptions(state):
            return False
        return self._present(artifact.get("hidden_assumptions")) or self._present(artifact.get("contradictions"))

    def _adversarial_needed(self, artifact: Dict[str, object]) -> bool:
        return self._present(artifact.get("pressure_points"))

    def _operator_evidence_gap(self, artifact: Dict[str, object]) -> bool:
        has_decision = self._present(artifact.get("decision"))
        has_rationale = self._present(artifact.get("rationale"))
        return (not has_decision) or (has_decision and not has_rationale)

    def _recurrence_established(self, state: RouterState) -> bool:
        return state.recurrence_potential >= 2.0

    def _has_live_assumptions(self, state: RouterState) -> bool:
        return len(state.assumptions) > 0


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
        if state.current_regime.stage == stage:
            return state.current_regime
        if state.runner_up_regime and state.runner_up_regime.stage == stage:
            return state.runner_up_regime
        if state.recommended_next_regime and state.recommended_next_regime.stage == stage:
            return state.recommended_next_regime
        return self._composer.compose(stage)

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


class EvolutionEngine:
    def propose_revision(self, regime: Regime, failure: FailureLog) -> RevisionProposal:
        if failure.recurrence_count >= 2 and failure.severity == Severity.HIGH:
            revision_type = "tighten"
        else:
            revision_type = "replace"

        target_failure = failure.observed_failure
        old_instruction = regime.dominant_line.text if failure.implicated_instruction_ids and regime.dominant_line.id in failure.implicated_instruction_ids else None
        new_instruction = None

        if regime.stage == Stage.SYNTHESIS and "coherence" in target_failure.lower():
            new_instruction = "If evidence directly weakens the central frame, revise the frame before integrating surrounding material."
            revision_type = "tighten"
            old_instruction = LIBRARY["SYN-P1"].text
        elif regime.stage == Stage.ADVERSARIAL and "weak objections" in target_failure.lower():
            new_instruction = "If a critique does not change the next action, omit it."
            revision_type = "add"
        elif regime.stage == Stage.OPERATOR and "forced closure" in target_failure.lower():
            new_instruction = "If two options are close, choose the one that preserves future flexibility."
            revision_type = "add"
        else:
            new_instruction = failure.missing_instruction or "Tighten the most ambiguous line into a decision-changing rule."

        return RevisionProposal(
            regime_name=regime.name,
            revision_type=revision_type,
            target_failure=target_failure,
            old_instruction=old_instruction,
            new_instruction=new_instruction,
            expected_increase=["regime stability", "failure containment"],
            expected_decrease=[target_failure],
            likely_side_effect=["reduced flexibility", "possible overcorrection if applied globally"],
            validation_test=(
                "Run the revised regime on 3 prompts that previously triggered the failure. "
                "Adopt only if the target failure decreases without destroying the regime's core behavior."
            ),
            adoption_recommendation="test_first",
        )


# ============================================================
# Runtime
