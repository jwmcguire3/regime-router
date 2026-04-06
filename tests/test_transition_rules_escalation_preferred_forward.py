import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.orchestration.escalation_policy import EscalationPolicyResult
from router.orchestration.transition_rules import _escalation_preferred_forward


def _escalation_result(
    *,
    direction: str = "stricter",
    switch_pressure_adjustment: int = 2,
    preferred_regime_biases: dict[Stage, int] | None = None,
) -> EscalationPolicyResult:
    return EscalationPolicyResult(
        escalation_direction=direction,
        justification="test",
        preferred_regime_biases=preferred_regime_biases or {},
        switch_pressure_adjustment=switch_pressure_adjustment,
        debug_signals=[],
    )


def test_escalation_preferred_forward_returns_none_when_direction_not_stricter():
    result = _escalation_preferred_forward(
        current_stage=Stage.SYNTHESIS,
        normal_next=Stage.EPISTEMIC,
        escalation=_escalation_result(direction="looser", preferred_regime_biases={Stage.ADVERSARIAL: 3}),
    )
    assert result is None


def test_escalation_preferred_forward_returns_none_when_switch_pressure_below_threshold():
    result = _escalation_preferred_forward(
        current_stage=Stage.SYNTHESIS,
        normal_next=Stage.EPISTEMIC,
        escalation=_escalation_result(
            switch_pressure_adjustment=1,
            preferred_regime_biases={Stage.ADVERSARIAL: 3},
        ),
    )
    assert result is None


def test_escalation_preferred_forward_returns_none_when_no_legal_forward_biased_stage():
    result = _escalation_preferred_forward(
        current_stage=Stage.EPISTEMIC,
        normal_next=Stage.OPERATOR,
        escalation=_escalation_result(
            preferred_regime_biases={
                Stage.OPERATOR: 4,  # already selected normal next
                Stage.SYNTHESIS: 3,  # not in EPISTEMIC forward pathway
                Stage.ADVERSARIAL: 0,  # weight below minimum
            },
        ),
    )
    assert result is None


def test_escalation_preferred_forward_selects_legal_forward_biased_stage_when_distinct():
    result = _escalation_preferred_forward(
        current_stage=Stage.SYNTHESIS,
        normal_next=Stage.EPISTEMIC,
        escalation=_escalation_result(
            preferred_regime_biases={
                Stage.EPISTEMIC: 1,
                Stage.ADVERSARIAL: 2,
            },
        ),
    )
    assert result == Stage.ADVERSARIAL


def test_escalation_preferred_forward_tie_breaking_is_deterministic():
    result = _escalation_preferred_forward(
        current_stage=Stage.SYNTHESIS,
        normal_next=Stage.OPERATOR,
        escalation=_escalation_result(
            preferred_regime_biases={
                Stage.EPISTEMIC: 2,
                Stage.ADVERSARIAL: 2,
            },
        ),
    )
    assert result == Stage.EPISTEMIC
