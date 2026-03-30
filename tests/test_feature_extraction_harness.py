import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import (
    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL,
    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED,
    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED,
)
from router.routing import extract_routing_features


def test_structural_signal_expansion_when_defined_present():
    features = extract_routing_features("The scope expands whenever we define the frame.")

    assert STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED in features.structural_signals


def test_structural_signal_concrete_versions_feel_too_small_present():
    features = extract_routing_features("The concrete implementation feels too small for the problem.")

    assert STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL in features.structural_signals


def test_structural_signal_fragments_understood_spine_missed_present():
    features = extract_routing_features(
        "The parts are understood and clear, but the core is missing."
    )

    assert STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in features.structural_signals


def test_structural_signals_empty_when_no_pattern_matches():
    features = extract_routing_features("Schedule a lunch next Tuesday and send invites.")

    assert features.structural_signals == []


def test_pressure_scores_detect_expected_dimensions():
    decision = extract_routing_features("We must decide now and choose.")
    evidence = extract_routing_features("The evidence is unknown and unclear.")
    fragility = extract_routing_features("This is fragile; stress test the plan.")
    recurrence = extract_routing_features("Make this repeatable template and systematize it.")
    possibility = extract_routing_features("Explore, brainstorm alternatives before narrowing.")

    assert decision.decision_pressure > 0
    assert evidence.evidence_demand > 0
    assert fragility.fragility_pressure > 0
    assert recurrence.recurrence_potential > 0
    assert possibility.possibility_space_need > 0


def test_pressure_scores_all_zero_when_no_pressure_words_present():
    features = extract_routing_features("Catalog files alphabetically by folder name.")

    assert features.decision_pressure == 0
    assert features.evidence_demand == 0
    assert features.fragility_pressure == 0
    assert features.recurrence_potential == 0
    assert features.possibility_space_need == 0


def test_marker_detection_negated_closure_preference():
    features = extract_routing_features("Please do not decide yet.")

    assert "negated_closure_preference" in features.detected_markers


def test_marker_detection_anti_convergence_preference():
    features = extract_routing_features("Keep it open rather than converging.")

    assert "anti_convergence_preference" in features.detected_markers


def test_marker_detection_uncertainty_characterization():
    features = extract_routing_features("I can't tell what kind of issue this is.")

    assert "uncertainty_characterization" in features.detected_markers


def test_marker_detection_uncertainty_evidence_demand():
    features = extract_routing_features("Evidence is missing and unclear.")

    assert "uncertainty_evidence_demand" in features.detected_markers


def test_word_boundary_breakout_does_not_match_break():
    features = extract_routing_features("This breakout game has a polished tutorial.")

    assert features.fragility_pressure == 0


def test_word_boundary_break_matches_when_standalone():
    features = extract_routing_features("Break the system and inspect failure modes.")

    assert features.fragility_pressure > 0


def test_word_boundary_known_does_not_match_now():
    features = extract_routing_features("Known issues are tracked in the backlog.")

    assert features.decision_pressure == 0


def test_word_boundary_now_matches_when_standalone():
    features = extract_routing_features("Do it now.")

    assert features.decision_pressure > 0


def test_negated_closure_suppresses_decision_and_boosts_possibility():
    features = extract_routing_features("Do not decide yet, explore alternatives.")

    assert features.decision_pressure == 0
    assert features.possibility_space_need > 0
    assert "decision_tradeoff_commitment" not in features.detected_markers
    assert "negated_closure_preference" in features.detected_markers
