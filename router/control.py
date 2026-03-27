from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from .models import FailureLog, LIBRARY, Regime, RevisionProposal, Severity, Stage
from .state import RouterState


@dataclass(frozen=True)
class RegimeOutputContract:
    stage: Stage
    raw_response: str
    validation: Dict[str, object]


@dataclass(frozen=True)
class MisroutingDetectionResult:
    current_regime: Stage
    dominant_failure_mode: str
    still_productive: bool
    misrouting_detected: bool
    justification: str
    recommended_next_regime: Optional[Stage]


class MisroutingDetector:
    JUSTIFICATION_SWITCH = "Current regime is hitting its dominant failure mode. Switching is justified."
    JUSTIFICATION_STAY = "Current regime remains productive. Switching is not justified."

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

    def detect(self, state: RouterState, output: RegimeOutputContract) -> MisroutingDetectionResult:
        artifact = self._extract_artifact(output)
        stage = state.current_regime.stage
        signal_active = self._signal_active(stage, state, artifact)
        recommended_next = self._recommended_next_regime(state, artifact, signal_active)
        if signal_active:
            justification = self.JUSTIFICATION_SWITCH
        else:
            justification = self.JUSTIFICATION_STAY
        return MisroutingDetectionResult(
            current_regime=stage,
            dominant_failure_mode=self._DOMINANT_FAILURE_MODE_BY_STAGE[stage],
            still_productive=not signal_active,
            misrouting_detected=signal_active,
            justification=justification,
            recommended_next_regime=recommended_next,
        )

    def _recommended_next_regime(
        self,
        state: RouterState,
        artifact: Dict[str, object],
        signal_active: bool,
    ) -> Optional[Stage]:
        if not signal_active:
            return None
        if self._assumption_collapse_detected(state, artifact):
            return Stage.EXPLORATION

        stage = state.current_regime.stage
        if stage == Stage.SYNTHESIS and self._adversarial_needed(artifact):
            return Stage.ADVERSARIAL
        if stage == Stage.OPERATOR:
            if self._recurrence_established(state):
                return Stage.BUILDER
            if self._operator_evidence_gap(artifact):
                return Stage.EPISTEMIC
        return self._DEFAULT_NEXT_REGIME_BY_STAGE[stage]

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

    def _signal_active(self, stage: Stage, state: RouterState, artifact: Dict[str, object]) -> bool:
        if stage == Stage.EXPLORATION:
            candidate_frames = self._item_count(artifact.get("candidate_frames"))
            has_selection_criteria = self._present(artifact.get("selection_criteria"))
            return candidate_frames >= 4 and not has_selection_criteria

        if stage == Stage.SYNTHESIS:
            has_central_claim = self._present(artifact.get("central_claim"))
            has_organizing_idea = self._present(artifact.get("organizing_idea"))
            has_support = self._present(artifact.get("supporting_structure"))
            contradictions_live = len(state.contradictions) > 0
            has_pressure_points = self._present(artifact.get("pressure_points"))
            unsupported_unification = has_central_claim and has_organizing_idea and not has_support
            contradictions_flattened = contradictions_live and not has_pressure_points
            return unsupported_unification or contradictions_flattened

        if stage == Stage.EPISTEMIC:
            has_support_map = self._present(artifact.get("supported_claims")) or self._present(artifact.get("plausible_but_unproven"))
            has_decision_conclusion = self._present(artifact.get("decision_relevant_conclusions"))
            return has_support_map and not has_decision_conclusion

        if stage == Stage.ADVERSARIAL:
            destabilizers = artifact.get("top_destabilizers")
            residual_risks = artifact.get("residual_risks")
            same_objections = self._normalized(destabilizers) == self._normalized(residual_risks) and self._present(destabilizers)
            no_revision_movement = not self._present(artifact.get("survivable_revisions"))
            return same_objections or (self._present(destabilizers) and no_revision_movement)

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
