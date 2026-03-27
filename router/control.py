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

    _NEXT_REGIME_BY_STAGE = {
        Stage.EXPLORATION: Stage.SYNTHESIS,
        Stage.SYNTHESIS: Stage.EPISTEMIC,
        Stage.EPISTEMIC: Stage.OPERATOR,
        Stage.ADVERSARIAL: Stage.SYNTHESIS,
        Stage.OPERATOR: Stage.EPISTEMIC,
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
        recommended_next = self._recommended_next_regime(state, signal_active)
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

    def _recommended_next_regime(self, state: RouterState, signal_active: bool) -> Optional[Stage]:
        if not signal_active:
            return None
        if state.recommended_next_regime is not None:
            return state.recommended_next_regime.stage
        return self._NEXT_REGIME_BY_STAGE[state.current_regime.stage]

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
            frames = self._list_len(artifact.get("candidate_frames"))
            has_criteria = self._text_len(artifact.get("selection_criteria")) >= 5
            unresolved_axes = self._list_len(artifact.get("unresolved_axes"))
            too_many_without_criteria = frames >= 4 and not has_criteria
            novelty_outruns_relevance = frames >= 3 and unresolved_axes >= 3 and not has_criteria
            return too_many_without_criteria or novelty_outruns_relevance

        if stage == Stage.SYNTHESIS:
            support_is_thin = self._text_len(artifact.get("supporting_structure")) < 8
            claim_is_clean = self._text_len(artifact.get("central_claim")) >= 8
            contradictions_live = len(state.contradictions) > 0
            pressure_points = self._text_value(artifact.get("pressure_points"))
            flattening_contradiction = contradictions_live and "contradict" not in pressure_points
            return (claim_is_clean and support_is_thin) or flattening_contradiction

        if stage == Stage.EPISTEMIC:
            decision_useful = self._text_len(artifact.get("decision_relevant_conclusions")) >= 6
            uncertainty_items = self._list_len(artifact.get("plausible_but_unproven"))
            practical_movement = self._contains_any(
                self._text_value(artifact.get("decision_relevant_conclusions")),
                ("next", "do", "choose", "act", "trigger"),
            )
            reportorial_only = not decision_useful
            uncertainty_without_movement = uncertainty_items >= 3 and not practical_movement
            return reportorial_only or uncertainty_without_movement

        if stage == Stage.ADVERSARIAL:
            destabilizers = self._text_value(artifact.get("top_destabilizers"))
            residual_risks = self._text_value(artifact.get("residual_risks"))
            survivable_revisions = self._text_len(artifact.get("survivable_revisions"))
            repetitive_critique = destabilizers != "" and destabilizers == residual_risks
            attack_loop = self._text_len(artifact.get("break_conditions")) >= 6 and survivable_revisions < 6
            return repetitive_critique or attack_loop

        if stage == Stage.OPERATOR:
            assumptions_live = len(state.assumptions) > 0
            tradeoff_stable = self._text_len(artifact.get("tradeoff_accepted")) >= 6
            rationale_stable = self._text_len(artifact.get("rationale")) >= 8
            forced_before_stable = assumptions_live and (not tradeoff_stable or not rationale_stable)
            return forced_before_stable

        if stage == Stage.BUILDER:
            recurrence_established = state.recurrence_potential >= 2.0
            has_architecture = self._text_len(artifact.get("modules")) >= 6 or self._text_len(artifact.get("interfaces")) >= 6
            return has_architecture and not recurrence_established

        return False

    def _list_len(self, value: object) -> int:
        if isinstance(value, list):
            return len(value)
        return 0

    def _text_value(self, value: object) -> str:
        if isinstance(value, list):
            value = " ".join(str(item) for item in value)
        if not isinstance(value, str):
            return ""
        return value.strip().lower()

    def _text_len(self, value: object) -> int:
        text = self._text_value(value)
        if not text:
            return 0
        return len(text.split())

    def _contains_any(self, text: str, markers: tuple[str, ...]) -> bool:
        return any(marker in text for marker in markers)


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
