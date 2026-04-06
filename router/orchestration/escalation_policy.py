from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..models import Regime, RegimeConfidenceResult, Severity, Stage
from ..state import RouterState
from .escalation_rules import (
    HIGH_CONSEQUENCE_TERMS,
    LACK_OF_STRUCTURE_TERMS,
    LOOSER_CUES,
    STRICTER_CUES,
    UNDERFORMED_TERMS,
)
from .misrouting_detector import MisroutingDetectionResult


@dataclass(frozen=True)
class EscalationPolicyResult:
    escalation_direction: str
    justification: str
    preferred_regime_biases: Dict[Stage, int]
    switch_pressure_adjustment: int
    debug_signals: List[str] = field(default_factory=list)


class EscalationPolicy:
    def evaluate(
        self,
        *,
        state: Optional[RouterState],
        task_text: str,
        current_regime: Optional[Regime],
        regime_confidence: Optional[RegimeConfidenceResult],
        misrouting_result: Optional[MisroutingDetectionResult],
        **_: object,
    ) -> EscalationPolicyResult:
        text = task_text.lower().replace("’", "'")
        strict_score = 0
        loose_score = 0
        strict_biases: Dict[Stage, int] = {}
        loose_biases: Dict[Stage, int] = {}
        cues: List[str] = []

        fragility_pressure = state.fragility_pressure if state else 0
        possibility_space_need = state.possibility_space_need if state else 0

        if fragility_pressure >= 2 or any(k in text for k in HIGH_CONSEQUENCE_TERMS):
            strict_score += 2
            cues.append("high_consequence_or_deployment")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 1
            strict_biases[Stage.ADVERSARIAL] = strict_biases.get(Stage.ADVERSARIAL, 0) + 1

        contradiction_count = len(state.contradictions) if state else 0
        if contradiction_count >= 2:
            strict_score += 2
            cues.append("contradiction_accumulation")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 2

        if any(cue in text for cue in STRICTER_CUES):
            strict_score += 2
            cues.append("certainty_or_proof_request")
            strict_biases[Stage.EPISTEMIC] = strict_biases.get(Stage.EPISTEMIC, 0) + 2

        if possibility_space_need >= 3 and any(cue in text for cue in UNDERFORMED_TERMS):
            loose_score += 2
            cues.append("underformed_space")
            loose_biases[Stage.EXPLORATION] = loose_biases.get(Stage.EXPLORATION, 0) + 2

        if any(cue in text for cue in LOOSER_CUES):
            loose_score += 2
            cues.append("premature_narrowing_signal")
            loose_biases[Stage.EXPLORATION] = loose_biases.get(Stage.EXPLORATION, 0) + 1

        if any(term in text for term in LACK_OF_STRUCTURE_TERMS):
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
