from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from ..models import Stage
from ..state import RouterState

COLLAPSE_VOCABULARY = {
    "collapse",
    "invalidated",
    "broke",
    "failed",
    "unfounded",
    "unsupported",
    "contradicted",
    "untenable",
}

CORE_STAGE_FIELDS = {
    Stage.EXPLORATION: {"candidate_frames", "selection_criteria", "unresolved_axes"},
    Stage.SYNTHESIS: {"central_claim", "key_tensions", "pressure_points"},
    Stage.EPISTEMIC: {"supported_claims", "contradictions", "decision_relevant_conclusions"},
    Stage.ADVERSARIAL: {"top_destabilizers", "break_conditions", "survivable_revisions"},
    Stage.OPERATOR: {"decision", "tradeoff_accepted", "next_actions"},
    Stage.BUILDER: {"reusable_pattern", "modules", "implementation_sequence"},
}

_NEGATION_MARKERS = {
    "not",
    "never",
    "cannot",
    "can't",
    "invalid",
    "invalidated",
    "unsupported",
    "untenable",
    "contradicted",
    "fails",
    "failed",
    "broke",
}


@dataclass(frozen=True)
class CollapseDetectionResult:
    collapse_detected: bool
    strong_signal: bool
    triggered_channels: tuple[str, ...]
    reason: str


class CollapseDetector:
    def detect(
        self,
        state: RouterState,
        validation_result: Mapping[str, object],
        artifact: Mapping[str, object],
        failure_signal: str,
    ) -> CollapseDetectionResult:
        channel_a = self._validation_strong_signal(state.current_regime.stage, validation_result)
        channel_b = self._artifact_contradiction_signal(artifact)
        channel_c = self._state_unresolved_contradictions_signal(state, artifact)
        channel_d = self._control_field_signal(failure_signal)

        weak_channels = [name for name, active in (("b", channel_b), ("c", channel_c), ("d", channel_d)) if active]
        all_channels = [name for name, active in (("a", channel_a), ("b", channel_b), ("c", channel_c), ("d", channel_d)) if active]

        if channel_a:
            return CollapseDetectionResult(
                collapse_detected=True,
                strong_signal=True,
                triggered_channels=tuple(all_channels),
                reason="structural_semantic_failure_on_core_field",
            )
        if len(weak_channels) >= 2:
            return CollapseDetectionResult(
                collapse_detected=True,
                strong_signal=False,
                triggered_channels=tuple(all_channels),
                reason="multi_channel_collapse_consensus",
            )
        return CollapseDetectionResult(
            collapse_detected=False,
            strong_signal=False,
            triggered_channels=tuple(all_channels),
            reason="insufficient_collapse_signals",
        )

    def _validation_strong_signal(self, stage: Stage, validation_result: Mapping[str, object]) -> bool:
        semantic_failures = validation_result.get("semantic_failures", [])
        if not isinstance(semantic_failures, Sequence):
            return False
        core_fields = CORE_STAGE_FIELDS.get(stage, set())
        for failure in semantic_failures:
            failure_text = str(failure).strip().lower()
            if any(failure_text.startswith(f"{field.lower()} ") for field in core_fields):
                return True
        return False

    def _artifact_contradiction_signal(self, artifact: Mapping[str, object]) -> bool:
        claim = self._compact_text(artifact.get("central_claim")) or self._compact_text(artifact.get("decision"))
        if not claim:
            return False
        statements = self._collect_text_items(artifact.get("hidden_assumptions"))
        statements.extend(self._collect_text_items(artifact.get("contradictions")))
        return any(self._textually_contradicts(claim, statement) for statement in statements)

    def _state_unresolved_contradictions_signal(self, state: RouterState, artifact: Mapping[str, object]) -> bool:
        if not state.substantive_assumptions:
            return False
        if not state.contradictions:
            return False
        artifact_contradictions = self._collect_text_items(artifact.get("contradictions"))
        if not artifact_contradictions:
            return True
        return len(artifact_contradictions) < len(state.contradictions)

    def _control_field_signal(self, failure_signal: str) -> bool:
        normalized = failure_signal.strip().lower()
        if not normalized:
            return False
        return any(token in normalized for token in COLLAPSE_VOCABULARY)

    def _textually_contradicts(self, claim: str, statement: str) -> bool:
        statement_tokens = set(statement.lower().split())
        claim_tokens = set(claim.lower().split())
        overlap = claim_tokens.intersection(statement_tokens)
        has_negation = bool(statement_tokens.intersection(_NEGATION_MARKERS))
        if overlap and has_negation:
            return True
        if ("contradict" in statement.lower() or "invalid" in statement.lower()) and claim:
            return True
        return False

    def _collect_text_items(self, value: object) -> list[str]:
        if isinstance(value, str) and value.strip():
            return [self._compact_text(value)]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [self._compact_text(item) for item in value if self._compact_text(item)]
        return []

    def _compact_text(self, value: Optional[object]) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().split())
