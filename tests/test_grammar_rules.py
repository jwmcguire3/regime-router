from router.models import LIBRARY
from router.routing import (
    deduplicate_lines,
    has_hard_conflict,
    has_soft_conflict,
    validate_regime_grammar,
)


def test_has_hard_conflict_detects_known_incompatible_pairs():
    assert has_hard_conflict(LIBRARY["EXP-D1"], LIBRARY["OPR-D1"])
    assert has_hard_conflict(LIBRARY["SYN-D1"], LIBRARY["EPI-D2"])


def test_has_soft_conflict_detects_dominance_and_shape_stage_mismatch():
    assert has_soft_conflict(LIBRARY["SYN-D1"], LIBRARY["SYN-D2"])
    assert has_soft_conflict(LIBRARY["OPR-S1"], LIBRARY["SYN-D1"])


def test_validate_regime_grammar_passes_for_well_formed_regime():
    lines = [
        LIBRARY["SYN-D1"],
        LIBRARY["SYN-P1"],
        LIBRARY["SYN-P2"],
        LIBRARY["SYN-S1"],
    ]
    is_valid, violations = validate_regime_grammar(lines)

    assert is_valid
    assert violations == []


def test_validate_regime_grammar_fails_for_too_many_dominance_lines():
    lines = [
        LIBRARY["EXP-D1"],
        LIBRARY["SYN-D1"],
        LIBRARY["EPI-D1"],
        LIBRARY["SYN-P1"],
    ]
    is_valid, violations = validate_regime_grammar(lines)

    assert not is_valid
    assert any("dominance" in msg.lower() for msg in violations)


def test_validate_regime_grammar_fails_for_more_than_five_lines():
    lines = [
        LIBRARY["SYN-D1"],
        LIBRARY["SYN-P1"],
        LIBRARY["SYN-P2"],
        LIBRARY["SYN-S1"],
        LIBRARY["EPI-G1"],
        LIBRARY["OPR-S2"],
    ]
    is_valid, violations = validate_regime_grammar(lines)

    assert not is_valid
    assert any("at most 5 lines" in msg.lower() for msg in violations)


def test_deduplicate_lines_preserves_dominance_and_trims_lowest_priority():
    lines = [
        LIBRARY["SYN-S1"],
        LIBRARY["SYN-P1"],
        LIBRARY["SYN-D1"],
        LIBRARY["SYN-D1"],
        LIBRARY["EPI-G1"],
        LIBRARY["OPR-S2"],
    ]

    result = deduplicate_lines(lines, max_lines=3)
    result_ids = [line.id for line in result]

    assert "SYN-D1" in result_ids
    assert "SYN-P1" in result_ids
    assert "EPI-G1" not in result_ids
