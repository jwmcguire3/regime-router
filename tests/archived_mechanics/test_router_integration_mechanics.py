import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from router.models import Stage
from router.routing import Router, extract_routing_features


def test_mixed_prompt_debug_contributions_show_operator_precedence_reason():
    decision = Router().route("Explore a few alternatives, then choose the best option now.")
    operator_contributions = decision.deterministic_score_contributions.get(Stage.OPERATOR, [])
    assert any("mixed_prompt:explicit_decision_now_precedence" in line for line in operator_contributions)


def test_escalation_trust_deployment_prompt_biases_stricter():
    decision = Router().route("We are near deployment and need high trust in production.")
    assert decision.primary_regime in {Stage.EPISTEMIC, Stage.ADVERSARIAL}
    assert any(
        "escalation_policy:stricter" in item
        for items in decision.deterministic_score_contributions.values()
        for item in items
    )


def test_escalation_contradiction_heavy_prompt_biases_stricter():
    decision = Router().route("Contradictions are accumulating; we need confidence and proof before deciding.")
    assert decision.primary_regime == Stage.EPISTEMIC
    assert any(
        "escalation_policy:stricter" in item
        for items in decision.deterministic_score_contributions.values()
        for item in items
    )


def test_escalation_certainty_seeking_prompt_biases_stricter():
    decision = Router().route("I need a certainty-level answer with explicit proof and confidence bounds.")
    assert decision.primary_regime == Stage.EPISTEMIC
    assert any(
        "escalation_policy:stricter" in item
        for items in decision.deterministic_score_contributions.values()
        for item in items
    )


def test_escalation_underformed_brainstorm_prompt_biases_looser():
    decision = Router().route("This space is underformed. Brainstorm and map the space before we commit.")
    assert decision.primary_regime == Stage.EXPLORATION
    assert any(
        "escalation_policy:looser" in item
        for items in decision.deterministic_score_contributions.values()
        for item in items
    )


def test_escalation_premature_narrowing_prompt_biases_looser():
    decision = Router().route("Narrowing happened too early; keep it open and explore before narrowing.")
    assert decision.primary_regime == Stage.EXPLORATION
    assert any(
        "escalation_policy:looser" in item
        for items in decision.deterministic_score_contributions.values()
        for item in items
    )


def test_routing_uncertainty_characterization_boosts_epistemic():
    features = extract_routing_features("I can’t characterize the pattern yet.")
    decision = Router().route("I can’t characterize the pattern yet.", routing_features=features)
    assert features.evidence_demand >= 1
    assert "uncertainty_characterization" in features.detected_markers
    assert decision.primary_regime == Stage.EPISTEMIC
    assert decision.primary_regime != Stage.BUILDER
