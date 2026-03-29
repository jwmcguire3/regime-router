import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import ARTIFACT_FIELDS, ARTIFACT_HINTS, COMPLETION_SIGNAL_HINTS, FAILURE_SIGNAL_HINTS, REGIME_PURPOSE_HINTS, Stage
from router.validation import OutputValidator


def _artifact_for(stage: Stage) -> dict:
    artifact = {}
    for field in ARTIFACT_FIELDS[stage]:
        if field in {"next_actions", "candidate_frames", "selection_criteria", "unresolved_axes", "key_tensions", "supporting_structure", "pressure_points", "supported_claims", "plausible_but_unproven", "contradictions", "omitted_due_to_insufficient_support", "decision_relevant_conclusions", "top_destabilizers", "hidden_assumptions", "break_conditions", "survivable_revisions", "residual_risks", "modules", "interfaces", "required_inputs", "produced_outputs", "implementation_sequence", "compounding_path"}:
            artifact[field] = [f"{field} detail"]
        else:
            artifact[field] = f"{field} detail"
    return artifact


def _contract(stage: Stage, regime: str) -> str:
    return json.dumps(
        {
            "regime": regime,
            "purpose": REGIME_PURPOSE_HINTS[stage],
            "artifact_type": ARTIFACT_HINTS[stage],
            "artifact": _artifact_for(stage),
            "completion_signal": COMPLETION_SIGNAL_HINTS[stage],
            "failure_signal": FAILURE_SIGNAL_HINTS[stage],
            "recommended_next_regime": Stage.EXPLORATION.value,
        }
    )


def test_regime_matches_operator_stage_is_valid():
    validation = OutputValidator().validate(
        Stage.OPERATOR,
        _contract(Stage.OPERATOR, regime=Stage.OPERATOR.value),
        model_profile="off",
    )

    assert validation["contract_controls_valid"] is True
    assert "regime field mismatch" not in " ".join(validation["control_failures"])


def test_regime_mismatch_operator_stage_adds_control_failure():
    validation = OutputValidator().validate(
        Stage.OPERATOR,
        _contract(Stage.OPERATOR, regime=Stage.EXPLORATION.value),
        model_profile="off",
    )

    assert validation["contract_controls_valid"] is False
    assert (
        f"regime field mismatch: output claims '{Stage.EXPLORATION.value}' "
        f"but active regime is {Stage.OPERATOR.value}"
    ) in validation["control_failures"]


def test_empty_regime_synthesis_stage_adds_control_failure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        _contract(Stage.SYNTHESIS, regime=""),
        model_profile="off",
    )

    assert validation["contract_controls_valid"] is False
    assert (
        f"regime field mismatch: output claims '' but active regime is {Stage.SYNTHESIS.value}"
    ) in validation["control_failures"]


def test_regime_matches_adversarial_stage_is_valid():
    validation = OutputValidator().validate(
        Stage.ADVERSARIAL,
        _contract(Stage.ADVERSARIAL, regime=Stage.ADVERSARIAL.value),
        model_profile="off",
    )

    assert validation["contract_controls_valid"] is True
    assert "regime field mismatch" not in " ".join(validation["control_failures"])


def test_descriptive_regime_name_containing_stage_is_valid():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        _contract(Stage.SYNTHESIS, regime="Synthesis Core"),
        model_profile="off",
    )

    assert validation["contract_controls_valid"] is True
    assert "regime field mismatch" not in " ".join(validation["control_failures"])
