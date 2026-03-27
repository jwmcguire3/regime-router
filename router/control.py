from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from .models import FailureLog, LIBRARY, Regime, RevisionProposal, Severity, Stage
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
        still_productive = completion_signal and not failure_signal
        recommended_next = self._recommended_next_regime(state, artifact, failure_signal)
        if failure_signal:
            justification = self.JUSTIFICATION_SWITCH
        elif still_productive:
            justification = self.JUSTIFICATION_STAY
        else:
            justification = self.JUSTIFICATION_INCOMPLETE
        return MisroutingDetectionResult(
            current_regime=current_regime,
            dominant_failure_mode=self._DOMINANT_FAILURE_MODE_BY_STAGE[stage],
            still_productive=still_productive,
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
            sprawl_directive = self._exploration_is_expansion_only(artifact)
            return (candidate_frames >= 5 and not has_differentiation) or (sprawl_directive and not has_differentiation)

        if stage == Stage.SYNTHESIS:
            has_central_claim = self._present(artifact.get("central_claim"))
            has_organizing_idea = self._present(artifact.get("organizing_idea"))
            has_support = self._present(artifact.get("supporting_structure"))
            contradictions_live = len(state.contradictions) > 0
            has_pressure_points = self._synthesis_has_substantive_pressure_points(artifact)
            unsupported_unification = has_central_claim and has_organizing_idea and not has_support and not has_pressure_points
            contradictions_flattened = contradictions_live and not has_pressure_points and not has_support
            forced_unification = self._synthesis_forced_unification(artifact)
            return unsupported_unification or contradictions_flattened or forced_unification

        if stage == Stage.EPISTEMIC:
            has_support_separation = self._epistemic_has_support_separation(artifact)
            has_uncertainty_handling = self._epistemic_has_uncertainty_handling(state, artifact)
            return not has_support_separation or not has_uncertainty_handling

        if stage == Stage.ADVERSARIAL:
            has_objections = self._adversarial_has_objections(artifact)
            no_revision_movement = not self._adversarial_has_revision_movement(artifact)
            return has_objections and no_revision_movement

        if stage == Stage.OPERATOR:
            has_decision = self._present(artifact.get("decision"))
            if not has_decision:
                return False
            missing_tradeoff = not self._present(artifact.get("tradeoff_accepted"))
            missing_rationale = not self._present(artifact.get("rationale"))
            missing_fallback = not self._present(artifact.get("fallback_trigger"))
            assumptions_hidden = self._has_live_assumptions(state) and missing_rationale and missing_fallback
            return missing_tradeoff or missing_rationale or missing_fallback or assumptions_hidden

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
            has_connective_structure = self._present(artifact.get("supporting_structure")) or self._synthesis_has_substantive_pressure_points(artifact)
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
        has_structured_comparison = self._exploration_frames_show_structure(candidate_frames)
        return has_selection_criteria or has_unresolved_axes or has_structured_comparison

    def _exploration_frames_show_structure(self, candidate_frames: object) -> bool:
        if not isinstance(candidate_frames, list):
            return False
        comparison_markers = (
            " vs ",
            " versus ",
            " compared",
            " contrast",
            " tradeoff",
            " grouped",
            " grouped by",
            " cluster",
            " category",
            " axis",
            " dimensions",
            " narrow",
            " prioritize",
            " distinguish",
        )
        rich_frames = [frame for frame in candidate_frames if isinstance(frame, str) and len(frame.strip().split()) >= 3]
        if len(rich_frames) < 2:
            return False
        return any(marker in self._normalized(rich_frames) for marker in comparison_markers)

    def _epistemic_has_support_separation(self, artifact: Dict[str, object]) -> bool:
        has_supported = self._present(artifact.get("supported_claims"))
        has_unproven = (
            self._present(artifact.get("plausible_but_unproven"))
            or self._present(artifact.get("omitted_due_to_insufficient_support"))
            or self._present(artifact.get("weakly_supported_claims"))
            or self._present(artifact.get("unsupported_claims"))
        )
        support_map_text = self._normalized(artifact)
        textual_support_map = (
            ("well-supported" in support_map_text or "supported claims" in support_map_text)
            and (
                "weakly supported" in support_map_text
                or "unsupported" in support_map_text
                or "insufficient support" in support_map_text
                or "unproven" in support_map_text
            )
        )
        return (has_supported and has_unproven) or textual_support_map

    def _epistemic_has_uncertainty_handling(self, state: RouterState, artifact: Dict[str, object]) -> bool:
        has_uncertainty_markers = (
            self._present(artifact.get("contradictions"))
            or self._present(artifact.get("omitted_due_to_insufficient_support"))
            or self._present(artifact.get("hidden_assumptions"))
            or self._present(artifact.get("assumptions"))
            or self._present(artifact.get("uncertainty"))
            or self._present(artifact.get("evidence_gaps"))
        )
        text = self._normalized(artifact)
        textual_uncertainty_markers = any(
            marker in text
            for marker in (
                "assumption",
                "uncertainty",
                "evidence gap",
                "evidence gaps",
                "unknown",
                "insufficient support",
            )
        )
        return has_uncertainty_markers or textual_uncertainty_markers or len(state.assumptions) > 0 or len(state.contradictions) > 0

    def _synthesis_forced_unification(self, artifact: Dict[str, object]) -> bool:
        text = self._normalized(artifact)
        if not text:
            return False
        unification_markers = (
            "one clean unifying",
            "ties everything together",
            "single mechanism",
            "one frame explains everything",
            "even if some observations do not fit",
            "everything together",
            "unifying explanation",
        )
        suppression_markers = (
            "do not spend time on contradictions",
            "ignore contradictions",
            "suppress contradictions",
            "ignore weak support",
            "suppress weak support",
            "do not spend time on pressure points",
            "without pressure points",
            "do not spend time on weak support",
            "do not examine contradictions",
            "do not examine pressure points",
        )
        has_unification_push = any(marker in text for marker in unification_markers)
        has_integrity_suppression = any(marker in text for marker in suppression_markers)
        return has_unification_push and has_integrity_suppression

    def _exploration_is_expansion_only(self, artifact: Dict[str, object]) -> bool:
        text = self._normalized(artifact)
        if not text:
            return False
        expansion_markers = (
            "as many possible angles",
            "keep expanding",
            "do not narrow",
            "do not compare",
            "do not distinguish",
            "do not group",
            "do not cluster",
            "without narrowing",
        )
        return any(marker in text for marker in expansion_markers)

    def _synthesis_has_substantive_pressure_points(self, artifact: Dict[str, object]) -> bool:
        pressure_points = artifact.get("pressure_points")
        if not self._present(pressure_points):
            return False
        text = self._normalized(pressure_points)
        suppression_only = any(
            marker in text
            for marker in (
                "do not spend time on pressure points",
                "without pressure points",
                "ignore pressure points",
                "suppress pressure points",
            )
        )
        return not suppression_only

    def _adversarial_has_objections(self, artifact: Dict[str, object]) -> bool:
        objection_fields = (
            artifact.get("top_destabilizers"),
            artifact.get("residual_risks"),
            artifact.get("break_conditions"),
            artifact.get("objections"),
        )
        if any(self._present(field) for field in objection_fields):
            return True
        text = self._normalized(artifact)
        return "objection" in text or "failure case" in text or "destabilizer" in text

    def _adversarial_has_revision_movement(self, artifact: Dict[str, object]) -> bool:
        revision_fields = (
            artifact.get("survivable_revisions"),
            artifact.get("revisions"),
            artifact.get("mitigations"),
            artifact.get("countermeasures"),
            artifact.get("survivable_changes"),
        )
        if any(self._present(field) for field in revision_fields):
            return True
        text = self._normalized(artifact)
        return any(
            marker in text
            for marker in (
                "survivable revision",
                "survivable change",
                "mitigation",
                "revise",
                "adaptation",
            )
        )

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
        if not self._present(artifact.get("pressure_points")):
            return False
        pressure_text = self._normalized(artifact.get("pressure_points"))
        suppressed_pressure_testing = any(
            marker in pressure_text
            for marker in (
                "without pressure points",
                "ignore pressure points",
            )
        )
        if "do not spend time on" in pressure_text and "pressure points" in pressure_text:
            suppressed_pressure_testing = True
        return not suppressed_pressure_testing

    def _operator_evidence_gap(self, artifact: Dict[str, object]) -> bool:
        return self._present(artifact.get("decision")) and not self._present(artifact.get("rationale"))

    def _recurrence_established(self, state: RouterState) -> bool:
        return state.recurrence_potential >= 2.0

    def _has_live_assumptions(self, state: RouterState) -> bool:
        return len(state.assumptions) > 0


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
