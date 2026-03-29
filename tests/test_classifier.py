import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.classifier import TaskClassifier


def test_classifies_write_breakout_game_code_as_direct():
    result = TaskClassifier().classify("write breakout game python code")
    assert result.route_type == "direct"


def test_classifies_build_rest_api_as_direct():
    result = TaskClassifier().classify("build a REST API for user management")
    assert result.route_type == "direct"


def test_classifies_draft_q3_email_as_direct():
    result = TaskClassifier().classify("draft an email to the team about the Q3 results")
    assert result.route_type == "direct"


def test_classifies_fragments_spine_statement_as_regime():
    result = TaskClassifier().classify("the fragments are understood but the spine is still missing")
    assert result.route_type == "regime"


def test_classifies_stress_test_frame_as_regime():
    result = TaskClassifier().classify("stress test this frame for hidden assumptions")
    assert result.route_type == "regime"


def test_classifies_pricing_decision_question_as_regime():
    result = TaskClassifier().classify("what should we do about the pricing decision")
    assert result.route_type == "regime"


def test_classifies_explore_interpretations_as_regime():
    result = TaskClassifier().classify("explore multiple interpretations of this signal")
    assert result.route_type == "regime"


def test_classifies_fix_bug_as_direct():
    result = TaskClassifier().classify("fix the bug in auth.py line 42")
    assert result.route_type == "direct"


def test_classifies_architecture_fragility_why_question_as_regime():
    result = TaskClassifier().classify("why does this architecture feel fragile")
    assert result.route_type == "regime"
