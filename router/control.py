from __future__ import annotations

from typing import List, Mapping, Optional

from .models import (
    FailureLog,
    LIBRARY,
    MisroutingDetectionResult,
    Regime,
    RevisionProposal,
    Severity,
    Stage,
)
from .state import RouterState


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


class MisroutingDetector:
    STANDARD_SWITCH_MESSAGE = "Current regime is hitting its dominant failure mode. Switching is justified."

    def detect(
        self,
        state: RouterState,
        regime_output: str = "",
        validation: Optional[Mapping[str, object]] = None,
        switch_context: Optional[str] = None,
        confidence_override: Optional[str] = None,
    ) -> MisroutingDetectionResult:
        output_text = self._normalize(regime_output)
        validation_failures = self._extract_validation_failures(validation)
        context_text = self._normalize(switch_context or "")
        confidence_level = self._normalize(confidence_override or state.regime_confidence)

        failure_mode, evidence = self._stage_failure_evidence(
            state.current_regime,
            output_text=output_text,
            validation_failures=validation_failures,
            context_text=context_text,
            state=state,
        )

        misrouting_detected = len(evidence) >= 2 or (
            len(evidence) >= 1 and confidence_level == Severity.LOW.value and state.recommended_next_regime is not None
        )

        if misrouting_detected:
            return MisroutingDetectionResult(
                current_regime=state.current_regime,
                dominant_failure_mode=failure_mode,
                misrouting_detected=True,
                justification="; ".join(evidence),
                recommended_next_regime=state.recommended_next_regime or state.runner_up_regime,
                switch_message=self.STANDARD_SWITCH_MESSAGE,
            )

        return MisroutingDetectionResult(
            current_regime=state.current_regime,
            dominant_failure_mode=failure_mode,
            misrouting_detected=False,
            justification="No clear dominant-failure pattern detected.",
            recommended_next_regime=None,
            switch_message=None,
        )

    def _stage_failure_evidence(
        self,
        stage: Stage,
        *,
        output_text: str,
        validation_failures: List[str],
        context_text: str,
        state: RouterState,
    ) -> tuple[str, List[str]]:
        evidence: List[str] = []

        if stage == Stage.EXPLORATION:
            failure = "branching growth without decision relevance"
            if self._contains_all(output_text, "branch", "more") and "criterion" not in output_text:
                evidence.append("Branches keep expanding while no selection criterion appears.")
            if "novel" in output_text and self._contains_any(output_text, "unclear utility", "not useful", "low utility"):
                evidence.append("Novelty is increasing faster than practical usefulness.")
            if "selection criterion" in context_text and self._contains_any(context_text, "missing", "none", "unclear"):
                evidence.append("No stable selection criterion is emerging from exploration.")
            return failure, evidence

        if stage == Stage.SYNTHESIS:
            failure = "coherence polish outrunning evidential support"
            if self._contains_any(output_text, "coherent narrative", "clean frame", "elegant frame") and self._contains_any(
                output_text, "limited evidence", "thin evidence", "speculative"
            ):
                evidence.append("Frame polish exceeds the evidence currently available.")
            if self._contains_any(output_text, "resolved contradiction", "integrated contradiction", "single theme") and self._contains_any(
                output_text, "quickly", "immediately", "without test"
            ):
                evidence.append("Contradictions are converted into theme too quickly.")
            if any("not grounded" in failure for failure in validation_failures):
                evidence.append("Validation reports weak grounding while synthesis remains highly elegant.")
            return failure, evidence

        if stage == Stage.EPISTEMIC:
            failure = "accuracy-preserving ambiguity blocking action"
            if self._contains_any(output_text, "reported findings", "evidence summary", "status update") and self._contains_any(
                output_text, "no recommendation", "no next step", "cannot choose"
            ):
                evidence.append("Output is reportorial but not decision-useful.")
            if self._contains_any(output_text, "unknown", "uncertain") and self._contains_any(
                output_text, "blocks action", "cannot move", "cannot proceed"
            ):
                evidence.append("Preserved ambiguity is now blocking necessary choice.")
            if "omission" in context_text and self._contains_any(context_text, "blocks", "prevents decision"):
                evidence.append("Known omissions are preventing practical movement.")
            return failure, evidence

        if stage == Stage.ADVERSARIAL:
            failure = "critique loop with diminishing decision impact"
            if self._contains_any(output_text, "same objection", "repeated objection", "repetitive critique"):
                evidence.append("Objections are repetitive and low-impact.")
            if self._contains_any(output_text, "cannot satisfy", "nothing would satisfy", "unfalsifiable critique"):
                evidence.append("No plausible revision would satisfy the current attack loop.")
            if self._contains_any(context_text, "progress blocked", "no progress", "stalled by critique"):
                evidence.append("Critique mode is suppressing forward progress.")
            return failure, evidence

        if stage == Stage.OPERATOR:
            failure = "forced choice before frame stability"
            if self._contains_any(output_text, "decide now", "forced decision", "must choose immediately"):
                evidence.append("Decision pressure is forcing closure before frame stability.")
            if self._contains_any(output_text, "tradeoff unclear", "unclear tradeoff", "trade-offs unresolved"):
                evidence.append("Accepted tradeoff is still unclear.")
            if self._contains_any(output_text, "assumption", "assuming") and self._contains_any(
                output_text, "hidden", "unstated", "implicit"
            ):
                evidence.append("Hidden assumptions are carrying the decision.")
            return failure, evidence

        failure = "premature architecture for unstable recurrence"
        if self._contains_any(output_text, "hypothetical recurrence", "might recur", "could recur someday"):
            evidence.append("Recurrence remains hypothetical.")
        if self._contains_any(output_text, "platform", "framework", "architecture") and self._contains_any(
            output_text, "before demand", "without demand", "unclear demand"
        ):
            evidence.append("Architecture scope exceeds demonstrated need.")
        if self._contains_any(output_text, "new module", "additional module", "plugin") and self._contains_any(
            output_text, "premature", "before usage", "before demand"
        ):
            evidence.append("Modules are being invented before stable demand exists.")
        return failure, evidence

    def _extract_validation_failures(self, validation: Optional[Mapping[str, object]]) -> List[str]:
        if validation is None:
            return []
        failures = validation.get("semantic_failures", [])
        if not isinstance(failures, list):
            return []
        return [self._normalize(str(item)) for item in failures]

    def _contains_any(self, text: str, *needles: str) -> bool:
        return any(needle in text for needle in needles)

    def _contains_all(self, text: str, *needles: str) -> bool:
        return all(needle in text for needle in needles)

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().split())


# ============================================================
# Runtime
