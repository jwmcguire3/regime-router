import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.routing import _contains_any


def test_breakout_game_does_not_match_break_single_word():
    assert _contains_any("breakout game", ("break",)) == []


def test_break_under_pressure_matches_multi_word_phrase():
    assert _contains_any("break under pressure", ("break under pressure",)) == ["break under pressure"]


def test_known_facts_does_not_match_now_single_word():
    assert _contains_any("known facts", ("now",)) == []


def test_decide_now_matches_both_single_word_phrases():
    assert _contains_any("decide now", ("now", "decide")) == ["now", "decide"]


def test_stress_test_matches_multi_word_phrase():
    assert _contains_any("stress test", ("stress test",)) == ["stress test"]


def test_breakfast_meeting_does_not_match_break_single_word():
    assert _contains_any("breakfast meeting", ("break",)) == []
