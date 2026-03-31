from itertools import combinations

from router.models import FunctionType, Stage
from router.routing import GrammarComposer


def test_exploration_default_dominant():
    regime = GrammarComposer().compose(Stage.EXPLORATION)
    assert regime.dominant_line.id == "EXP-D1"


def test_synthesis_default_dominant():
    regime = GrammarComposer().compose(Stage.SYNTHESIS)
    assert regime.dominant_line.id == "SYN-D1"


def test_synthesis_sprawl_selects_syn_d2():
    regime = GrammarComposer().compose(Stage.SYNTHESIS, {"sprawl"})
    assert regime.dominant_line.id == "SYN-D2"


def test_adversarial_handoff_includes_adv_t1():
    regime = GrammarComposer().compose(Stage.ADVERSARIAL, handoff_expected=True)
    assert regime.tail_line is not None
    assert regime.tail_line.id == "ADV-T1"


def test_every_regime_has_exactly_one_dominance_line():
    composer = GrammarComposer()
    for stage in Stage:
        regime = composer.compose(stage)
        dominance = [line for line in regime.all_lines if line.function == FunctionType.DOMINANCE]
        assert len(dominance) == 1


def test_every_regime_has_at_most_five_total_lines():
    composer = GrammarComposer()
    for stage in Stage:
        regime = composer.compose(stage)
        assert len(regime.all_lines) <= 5


def test_every_regime_has_no_incompatible_line_pairs():
    composer = GrammarComposer()
    for stage in Stage:
        regime = composer.compose(stage)
        for line_a, line_b in combinations(regime.all_lines, 2):
            assert line_a.id not in line_b.incompatible_with
            assert line_b.id not in line_a.incompatible_with


def test_operator_optionality_includes_opr_g1():
    regime = GrammarComposer().compose(Stage.OPERATOR, {"optionality"})
    assert regime.tail_line is not None
    assert regime.tail_line.id == "OPR-G1"
