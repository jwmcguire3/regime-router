from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.models import Stage


class StubModelClient:
    def __init__(self, responses: List[Dict[str, object]]) -> None:
        self._responses = list(responses)

    def generate(self, **_: object) -> Dict[str, object]:
        if not self._responses:
            raise AssertionError("No stub responses configured")
        return self._responses.pop(0)

    def list_models(self) -> Dict[str, object]:
        return {"models": []}


def _analysis_output(
    *,
    top_stage: Stage,
    confidence: float = 0.6,
    decision_pressure: int = 5,
    fragility_pressure: int = 5,
    recurrence_potential: int = 5,
    likely_endpoint_regime: Stage = Stage.OPERATOR,
) -> Dict[str, object]:
    scores = {
        Stage.EXPLORATION.value: 0.1,
        Stage.SYNTHESIS.value: 0.2,
        Stage.EPISTEMIC.value: 0.3,
        Stage.ADVERSARIAL.value: 0.4,
        Stage.OPERATOR.value: 0.5,
        Stage.BUILDER.value: 0.6,
    }
    scores[top_stage.value] = 0.9
    return {
        "bottleneck_label": "policy test",
        "candidate_regimes": [top_stage.value, Stage.SYNTHESIS.value],
        "stage_scores": scores,
        "structural_signals": ["decision_tradeoff_commitment"],
        "decision_pressure": decision_pressure,
        "fragility_pressure": fragility_pressure,
        "possibility_space_need": 3,
        "synthesis_pressure": 2,
        "evidence_quality": 5,
        "recurrence_potential": recurrence_potential,
        "confidence": confidence,
        "rationale": "Policy behavior under test.",
        "risk_tags": ["high_stakes"],
        "likely_endpoint_regime": likely_endpoint_regime.value,
        "endpoint_confidence": 0.7,
    }


def _analyzer(payload: Dict[str, object]) -> TaskAnalyzer:
    return TaskAnalyzer(model_client=StubModelClient([{"response": json.dumps(payload)}]), model="stub")


def test_operator_primary_not_hard_demoted_when_decision_support_absent() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.OPERATOR, confidence=0.62, decision_pressure=0))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.OPERATOR
    assert "operator support weak; soft guardrail only" in decision.policy_warnings
    assert "demoted to exploration" not in (decision.analyzer_summary or "")


def test_adversarial_primary_not_hard_demoted_when_fragility_support_absent() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.ADVERSARIAL, confidence=0.45, fragility_pressure=0))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.ADVERSARIAL
    assert "adversarial support weak; advisory only" in decision.policy_warnings


def test_builder_primary_not_hard_demoted_when_recurrence_support_absent() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.BUILDER, confidence=0.7, recurrence_potential=0))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.BUILDER
    assert "builder support weak; advisory only" in decision.policy_warnings




def test_operator_weak_support_warning_suppressed_at_high_confidence() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.OPERATOR, confidence=0.8))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.OPERATOR
    assert "operator support weak; soft guardrail only" not in decision.policy_warnings


def test_adversarial_weak_support_warning_suppressed_at_high_confidence() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.ADVERSARIAL, confidence=0.85, fragility_pressure=0))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.ADVERSARIAL
    assert "adversarial support weak; advisory only" not in decision.policy_warnings


def test_builder_weak_support_warning_suppressed_at_high_confidence() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.BUILDER, confidence=0.9, recurrence_potential=0))

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.BUILDER
    assert "builder support weak; advisory only" not in decision.policy_warnings

def test_builder_endpoint_softened_when_support_absent_and_confidence_not_high() -> None:
    analyzer = _analyzer(
        _analysis_output(
            top_stage=Stage.BUILDER,
            confidence=0.7,
            recurrence_potential=0,
            likely_endpoint_regime=Stage.BUILDER,
        )
    )

    decision = analyzer.propose_route("task")

    assert decision.primary_regime == Stage.BUILDER
    assert decision.likely_endpoint_regime == Stage.OPERATOR.value
    assert "builder endpoint softened to operator" in decision.policy_actions


def test_pre_policy_and_post_policy_routing_are_serialized() -> None:
    analyzer = _analyzer(_analysis_output(top_stage=Stage.OPERATOR, confidence=0.62, decision_pressure=0))

    decision = analyzer.propose_route("task")

    assert decision.pre_policy_primary_regime == Stage.OPERATOR
    assert decision.pre_policy_runner_up_regime is not None
    assert decision.primary_regime == Stage.OPERATOR
    assert decision.runner_up_regime is not None
    assert decision.policy_warnings
