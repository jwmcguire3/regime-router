import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.validation import OutputValidator


SYNTHESIS_TASK = (
    "Find the strongest interpretation of what this actually is. "
    "When we define the effort it expands instead of narrowing. "
    "Concrete versions feel too small. "
    "The fragments are understood, but the spine is still missing."
)


def _contract(*, regime: str, purpose: str, artifact_type: str, artifact: dict, completion_signal: str, failure_signal: str, recommended_next_regime: str) -> str:
    return json.dumps(
        {
            "regime": regime,
            "purpose": purpose,
            "artifact_type": artifact_type,
            "artifact": artifact,
            "completion_signal": completion_signal,
            "failure_signal": failure_signal,
            "recommended_next_regime": recommended_next_regime,
        }
    )


def test_valid_synthesis_output_passes():
    raw = _contract(
        regime="Synthesis Core",
        purpose="Produce the strongest coherent interpretation from live signals.",
        artifact_type="dominant_frame",
        completion_signal="coherent_frame_stable",
        failure_signal="frame_collapses_under_pressure_points",
        recommended_next_regime="adversarial",
        artifact={
            "central_claim": "The frame expands when defined because concrete versions hide spine-level structure.",
            "organizing_idea": "Definition reveals links across fragments, so small concrete cuts miss the organizing spine.",
            "key_tensions": ["Define-and-expand dynamics vs concrete-small snapshots."],
            "supporting_structure": ["The task states fragments are understood while the spine is missing."],
            "pressure_points": ["If concrete examples preserve spine coherence without expansion, this frame weakens."],
        },
    )

    validation = OutputValidator().validate(Stage.SYNTHESIS, raw, task=SYNTHESIS_TASK)
    assert validation["is_valid"] is True


def test_valid_epistemic_output_passes():
    task = "We need to verify what claims are supported and what evidence is still missing before deciding."
    raw = _contract(
        regime="Epistemic Rigour",
        purpose="Separate supported claims from uncertainty and gaps.",
        artifact_type="evidence_map",
        completion_signal="evidence_boundary_clear",
        failure_signal="insufficient_support_for_key_claims",
        recommended_next_regime="operator",
        artifact={
            "supported_claims": ["The supported evidence shows two shortlisted options and a time constraint."],
            "plausible_but_unproven": ["Option A may reduce long-term maintenance load, but this is not yet supported evidence."],
            "contradictions": ["Stakeholder preference favors B while incident evidence supports A."],
            "omitted_due_to_insufficient_support": ["A final ROI claim is omitted because evidence is still missing."],
            "decision_relevant_conclusions": ["Verify one week of evidence before deciding."],
        },
    )

    validation = OutputValidator().validate(Stage.EPISTEMIC, raw, task=task)
    assert validation["is_valid"] is True


def test_valid_operator_output_passes():
    task = "Choose between option A and option B now, define immediate next actions, and specify fallback conditions."
    raw = _contract(
        regime="Operator Decisive",
        purpose="Commit to a concrete decision with executable next moves.",
        artifact_type="decision_packet",
        completion_signal="decision_committed_with_actions",
        failure_signal="decision_not_actionable_under_constraints",
        recommended_next_regime="epistemic",
        artifact={
            "decision": "Choose option A for this release cycle.",
            "rationale": "Option A fits the current constraints and can ship with lower immediate risk.",
            "tradeoff_accepted": "We accept reduced feature breadth to keep delivery predictable.",
            "next_actions": ["Assign owner", "Lock scope", "Start implementation by Monday"],
            "fallback_trigger": "If integration defects exceed threshold in week one, reconsider the decision.",
            "review_point": "Review at the end of the first sprint with defect and velocity metrics.",
        },
    )

    validation = OutputValidator().validate(Stage.OPERATOR, raw, task=task)
    assert validation["is_valid"] is True


def test_missing_purpose_fails():
    payload = json.loads(
        _contract(
            regime="Synthesis Core",
            purpose="placeholder",
            artifact_type="dominant_frame",
            completion_signal="coherent_frame_stable",
            failure_signal="frame_collapses_under_pressure_points",
            recommended_next_regime="adversarial",
            artifact={
                "central_claim": "c1 c2 c3",
                "organizing_idea": "o1 o2 o3",
                "key_tensions": ["k1 k2 k3"],
                "supporting_structure": ["s1 s2 s3"],
                "pressure_points": ["p1 p2 p3"],
            },
        )
    )
    payload.pop("purpose")
    validation = OutputValidator().validate(Stage.SYNTHESIS, json.dumps(payload), task=SYNTHESIS_TASK)
    assert validation["is_valid"] is False
    assert "purpose" in validation["missing_keys"]


def test_missing_completion_signal_fails():
    payload = json.loads(
        _contract(
            regime="Synthesis Core",
            purpose="Produce the strongest coherent interpretation from live signals.",
            artifact_type="dominant_frame",
            completion_signal="coherent_frame_stable",
            failure_signal="frame_collapses_under_pressure_points",
            recommended_next_regime="adversarial",
            artifact={
                "central_claim": "c1 c2 c3",
                "organizing_idea": "o1 o2 o3",
                "key_tensions": ["k1 k2 k3"],
                "supporting_structure": ["s1 s2 s3"],
                "pressure_points": ["p1 p2 p3"],
            },
        )
    )
    payload.pop("completion_signal")
    validation = OutputValidator().validate(Stage.SYNTHESIS, json.dumps(payload), task=SYNTHESIS_TASK)
    assert validation["is_valid"] is False
    assert "completion_signal" in validation["missing_keys"]


def test_missing_failure_signal_fails():
    payload = json.loads(
        _contract(
            regime="Synthesis Core",
            purpose="Produce the strongest coherent interpretation from live signals.",
            artifact_type="dominant_frame",
            completion_signal="coherent_frame_stable",
            failure_signal="frame_collapses_under_pressure_points",
            recommended_next_regime="adversarial",
            artifact={
                "central_claim": "c1 c2 c3",
                "organizing_idea": "o1 o2 o3",
                "key_tensions": ["k1 k2 k3"],
                "supporting_structure": ["s1 s2 s3"],
                "pressure_points": ["p1 p2 p3"],
            },
        )
    )
    payload.pop("failure_signal")
    validation = OutputValidator().validate(Stage.SYNTHESIS, json.dumps(payload), task=SYNTHESIS_TASK)
    assert validation["is_valid"] is False
    assert "failure_signal" in validation["missing_keys"]


def test_missing_recommended_next_regime_fails():
    payload = json.loads(
        _contract(
            regime="Synthesis Core",
            purpose="Produce the strongest coherent interpretation from live signals.",
            artifact_type="dominant_frame",
            completion_signal="coherent_frame_stable",
            failure_signal="frame_collapses_under_pressure_points",
            recommended_next_regime="adversarial",
            artifact={
                "central_claim": "c1 c2 c3",
                "organizing_idea": "o1 o2 o3",
                "key_tensions": ["k1 k2 k3"],
                "supporting_structure": ["s1 s2 s3"],
                "pressure_points": ["p1 p2 p3"],
            },
        )
    )
    payload.pop("recommended_next_regime")
    validation = OutputValidator().validate(Stage.SYNTHESIS, json.dumps(payload), task=SYNTHESIS_TASK)
    assert validation["is_valid"] is False
    assert "recommended_next_regime" in validation["missing_keys"]


def test_existing_artifact_payload_still_validates_inside_wrapper():
    artifact = {
        "central_claim": "The frame expands when defined because concrete versions feel too small.",
        "organizing_idea": "Fragments become coherent only when mapped to a spine-level relation.",
        "key_tensions": ["Concrete instance fit vs whole-system coherence."],
        "supporting_structure": ["The prompt says fragments are known but the spine is missing."],
        "pressure_points": ["If spine mapping adds no explanatory power, this frame should be dropped."],
    }
    raw = _contract(
        regime="Synthesis Core",
        purpose="Produce the strongest coherent interpretation from live signals.",
        artifact_type="dominant_frame",
        completion_signal="coherent_frame_stable",
        failure_signal="frame_collapses_under_pressure_points",
        recommended_next_regime="adversarial",
        artifact=artifact,
    )

    validation = OutputValidator().validate(Stage.SYNTHESIS, raw, task=SYNTHESIS_TASK)
    assert validation["is_valid"] is True
    assert validation["parsed"]["artifact"] == artifact
