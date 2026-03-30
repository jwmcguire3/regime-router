import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import (
    ARTIFACT_FIELDS,
    ARTIFACT_HINTS,
    COMPLETION_SIGNAL_HINTS,
    FAILURE_SIGNAL_HINTS,
    Stage,
)
from router.validation import OutputValidator


LIST_FIELDS = {
    "candidate_frames",
    "selection_criteria",
    "unresolved_axes",
    "key_tensions",
    "supporting_structure",
    "pressure_points",
    "supported_claims",
    "plausible_but_unproven",
    "contradictions",
    "omitted_due_to_insufficient_support",
    "decision_relevant_conclusions",
    "top_destabilizers",
    "hidden_assumptions",
    "break_conditions",
    "survivable_revisions",
    "residual_risks",
    "next_actions",
    "modules",
    "interfaces",
    "required_inputs",
    "produced_outputs",
    "implementation_sequence",
    "compounding_path",
}


def _artifact_value(stage: Stage, field: str):
    text = f"{stage.value} {field} provides concrete grounded detail"
    if field in LIST_FIELDS:
        return [text]
    return text


def build_output(
    stage,
    overrides=None,
    remove_keys=None,
    artifact_overrides=None,
    remove_artifact_keys=None,
) -> str:
    payload = {
        "regime": stage.value,
        "purpose": f"{stage.value} purpose keeps grounded structural focus",
        "artifact_type": ARTIFACT_HINTS[stage],
        "artifact": {field: _artifact_value(stage, field) for field in ARTIFACT_FIELDS[stage]},
        "completion_signal": COMPLETION_SIGNAL_HINTS[stage],
        "failure_signal": FAILURE_SIGNAL_HINTS[stage],
        "recommended_next_regime": Stage.EXPLORATION.value,
    }

    if overrides:
        payload.update(overrides)

    if remove_keys:
        for key in remove_keys:
            payload.pop(key, None)

    if artifact_overrides:
        payload["artifact"].update(artifact_overrides)

    if remove_artifact_keys:
        for key in remove_artifact_keys:
            payload["artifact"].pop(key, None)

    return json.dumps(payload)


# STRUCTURAL VALIDITY

def test_valid_output_for_each_stage_is_valid():
    validator = OutputValidator()
    for stage in Stage:
        validation = validator.validate(stage, build_output(stage), model_profile="off")
        assert validation["is_valid"] is True


def test_missing_purpose_key_fails_structure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, remove_keys=["purpose"]),
        model_profile="off",
    )
    assert validation["is_valid"] is False
    assert "purpose" in validation["missing_keys"]


def test_missing_completion_signal_key_fails_structure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, remove_keys=["completion_signal"]),
        model_profile="off",
    )
    assert validation["is_valid"] is False


def test_missing_failure_signal_key_fails_structure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, remove_keys=["failure_signal"]),
        model_profile="off",
    )
    assert validation["is_valid"] is False


def test_missing_recommended_next_regime_key_fails_structure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, remove_keys=["recommended_next_regime"]),
        model_profile="off",
    )
    assert validation["is_valid"] is False


def test_missing_artifact_field_fails_structure():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, remove_artifact_keys=["central_claim"]),
        model_profile="off",
    )
    assert validation["is_valid"] is False
    assert "central_claim" in validation["missing_artifact_fields"]


def test_wrong_artifact_type_sets_mismatch():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"artifact_type": "wrong_type"}),
        model_profile="off",
    )
    assert validation["is_valid"] is False
    assert validation["artifact_type_matches"] is False


def test_invalid_json_string_sets_valid_json_false():
    validation = OutputValidator().validate(Stage.SYNTHESIS, "{not-json", model_profile="off")
    assert validation["valid_json"] is False


def test_empty_string_sets_valid_json_false():
    validation = OutputValidator().validate(Stage.SYNTHESIS, "", model_profile="off")
    assert validation["valid_json"] is False


# REGIME FIELD VALIDATION

def test_descriptive_regime_with_synthesis_passes():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"regime": "Synthesis Core"}),
        model_profile="off",
    )
    assert validation["is_valid"] is True


def test_mismatched_regime_operator_fails_control():
    validation = OutputValidator().validate(
        Stage.OPERATOR,
        build_output(Stage.OPERATOR, overrides={"regime": "Exploration Core"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False
    assert any("regime field mismatch" in msg for msg in validation["control_failures"])


def test_exact_operator_regime_passes():
    validation = OutputValidator().validate(
        Stage.OPERATOR,
        build_output(Stage.OPERATOR, overrides={"regime": "operator"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is True


def test_empty_regime_for_synthesis_fails():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"regime": ""}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False


def test_adversarial_regime_passes():
    validation = OutputValidator().validate(
        Stage.ADVERSARIAL,
        build_output(Stage.ADVERSARIAL, overrides={"regime": "adversarial"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is True


# CONTROL FIELD

def test_completion_signal_without_expected_tokens_fails_control():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"completion_signal": "done and finished now"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False
    assert "completion_signal is not stage-appropriate" in validation["control_failures"]


def test_failure_signal_without_expected_tokens_fails_control():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"failure_signal": "this could maybe go badly"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False
    assert "failure_signal is not stage-appropriate" in validation["control_failures"]


def test_invalid_recommended_next_regime_fails_control():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"recommended_next_regime": "invalid_stage"}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False
    assert "recommended_next_regime must be a valid regime stage" in validation["control_failures"]


def test_empty_purpose_fails_control():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(Stage.SYNTHESIS, overrides={"purpose": ""}),
        model_profile="off",
    )
    assert validation["contract_controls_valid"] is False
    assert "purpose must be a non-empty string" in validation["control_failures"]


# SEMANTIC VALIDATION (STRICT)

def test_strict_field_too_short_is_flagged():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(
            Stage.SYNTHESIS,
            artifact_overrides={"central_claim": "too short"},
        ),
    )
    assert validation["semantic_valid"] is False
    assert "central_claim is too short to be meaningful" in validation["semantic_failures"]


def test_strict_duplicate_fields_detected():
    duplicate = "shared phrasing with enough unique words here"
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(
            Stage.SYNTHESIS,
            artifact_overrides={"central_claim": duplicate, "organizing_idea": duplicate},
        ),
    )
    assert validation["semantic_valid"] is False
    assert "central_claim duplicates organizing_idea" in validation["semantic_failures"]


def test_strict_forbidden_generic_domain_nouns_detected():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(
            Stage.SYNTHESIS,
            artifact_overrides={"central_claim": "this framing highlights technology impact on stakeholders deeply"},
        ),
        task_signals=["expansion_when_defined"],
    )
    assert validation["semantic_valid"] is False
    assert any("forbidden generic domain nouns" in msg for msg in validation["semantic_failures"])


def test_strict_synthesis_organizing_idea_restates_claim():
    claim = "define expand concrete spine fragment relationship"
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(
            Stage.SYNTHESIS,
            artifact_overrides={"central_claim": claim, "organizing_idea": claim},
        ),
    )
    assert validation["semantic_valid"] is False
    assert "organizing_idea restates central_claim instead of explaining it" in validation["semantic_failures"]


def test_strict_synthesis_supporting_structure_too_thin():
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        build_output(
            Stage.SYNTHESIS,
            artifact_overrides={"supporting_structure": ["too thin"]},
        ),
    )
    assert validation["semantic_valid"] is False
    assert "supporting_structure is too thin" in validation["semantic_failures"]


# PROFILE BEHAVIOR

def test_strict_fails_but_off_passes_for_generic_filler_and_forbidden_nouns():
    raw = build_output(
        Stage.SYNTHESIS,
        artifact_overrides={
            "central_claim": "careful consideration of technology and stakeholders shapes understanding",
            "organizing_idea": "navigating complexity through various factors without concrete anchors",
        },
    )

    strict_validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        raw,
        task_signals=["expansion_when_defined"],
        model_profile="strict",
    )
    off_validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        raw,
        task_signals=["expansion_when_defined"],
        model_profile="off",
    )

    assert strict_validation["semantic_valid"] is False
    assert off_validation["is_valid"] is True


def test_strict_fails_but_lenient_passes_for_generic_filler_and_forbidden_nouns():
    raw = build_output(
        Stage.SYNTHESIS,
        artifact_overrides={
            "central_claim": "careful consideration of technology and stakeholders shapes understanding",
            "organizing_idea": "navigating complexity through various factors without concrete anchors",
        },
    )

    strict_validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        raw,
        task_signals=["expansion_when_defined"],
        model_profile="strict",
    )
    lenient_validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        raw,
        task_signals=["expansion_when_defined"],
        model_profile="lenient",
    )

    assert strict_validation["semantic_valid"] is False
    assert lenient_validation["is_valid"] is True
