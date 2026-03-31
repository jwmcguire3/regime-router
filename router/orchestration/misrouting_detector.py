from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from ..models import Regime, Stage
from ..routing import RegimeComposer
from ..state import RouterState
from .misrouting_rules import (
    adversarial_needed,
    assumption_collapse_detected,
    completion_signal_active,
    failure_signal_active,
    operator_evidence_gap,
    recurrence_established,
)
from .output_contract import RegimeOutputContract


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
        failure_signal = failure_signal_active(stage, state, artifact)
        completion_signal = completion_signal_active(stage, state, artifact)
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
        if assumption_collapse_detected(state, artifact):
            return self._regime_for_stage(state, Stage.EXPLORATION)

        stage = state.current_regime.stage
        if stage == Stage.SYNTHESIS and adversarial_needed(artifact):
            return self._regime_for_stage(state, Stage.ADVERSARIAL)
        if stage == Stage.OPERATOR:
            if recurrence_established(state):
                return self._regime_for_stage(state, Stage.BUILDER)
            if operator_evidence_gap(artifact):
                return self._regime_for_stage(state, Stage.EPISTEMIC)
        return self._regime_for_stage(state, self._DEFAULT_NEXT_REGIME_BY_STAGE[stage])

    def _regime_for_stage(self, state: RouterState, stage: Stage) -> Regime:
        return state.resolve_regime(stage, self._composer.compose)

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
