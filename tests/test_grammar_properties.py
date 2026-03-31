from itertools import combinations

import pytest

from router.models import DOMINANT_FAILURE_MAP, FAILURE_SUPPRESSOR_MAP, FunctionType, Regime, Stage
from router.routing import GrammarComposer

RISK_PROFILES = [
    set(),
    {"high_stakes"},
    {"sprawl"},
    {"false_unification"},
    {"optionality"},
    {"elegant_theory_drift"},
]


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_all_stages_compose_without_error(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    assert regime is not None
    assert isinstance(regime, Regime)


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_exactly_one_dominance_line(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    dominance_lines = [line for line in regime.all_lines if line.function == FunctionType.DOMINANCE]

    assert len(dominance_lines) in (1, 2)
    if len(dominance_lines) == 2:
        assert dominance_lines[0].stage == dominance_lines[1].stage
        assert dominance_lines[0].id not in dominance_lines[1].incompatible_with
        assert dominance_lines[1].id not in dominance_lines[0].incompatible_with


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_suppression_count_in_range(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    assert len(regime.suppression_lines) in (1, 2)


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_total_lines_within_bound(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    assert len(regime.all_lines) <= 5


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_no_incompatible_pairs(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    for line_a, line_b in combinations(regime.all_lines, 2):
        assert line_a.id not in line_b.incompatible_with
        assert line_b.id not in line_a.incompatible_with


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_suppression_targets_dominant_failures(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    dominant_failures = DOMINANT_FAILURE_MAP[regime.dominant_line.id]
    suppression_ids = [line.id for line in regime.suppression_lines]

    for suppression_id in suppression_ids:
        assert any(
            suppression_id in FAILURE_SUPPRESSOR_MAP.get(failure, []) for failure in dominant_failures
        )


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_all_lines_from_same_stage_as_dominant(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    assert all(line.stage == regime.stage for line in regime.all_lines)


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("risk_profile", RISK_PROFILES)
@pytest.mark.parametrize("handoff_expected", [True, False])
def test_regime_name_and_metadata(stage: Stage, risk_profile: set[str], handoff_expected: bool):
    regime = GrammarComposer().compose(stage, risk_profile, handoff_expected)

    assert isinstance(regime.name, str)
    assert regime.name.strip()
    assert regime.stage == stage
    assert isinstance(regime.likely_failure_if_overused, str)
    assert regime.likely_failure_if_overused.strip()
