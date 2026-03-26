from __future__ import annotations

from .models import FailureLog, LIBRARY, Regime, RevisionProposal, Severity, Stage


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
