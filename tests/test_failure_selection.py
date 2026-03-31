import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import LIBRARY, Stage

MODULE_PATH = Path(__file__).resolve().parents[1] / "router" / "routing" / "failure_selection.py"
spec = importlib.util.spec_from_file_location("failure_selection", MODULE_PATH)
assert spec is not None and spec.loader is not None
failure_selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(failure_selection)

rank_failures_by_cost = failure_selection.rank_failures_by_cost
select_dominant = failure_selection.select_dominant
select_shapes = failure_selection.select_shapes
select_suppressions = failure_selection.select_suppressions
select_tail = failure_selection.select_tail


def test_select_dominant_returns_syn_d2_for_sprawl_risk():
    dominant = select_dominant(Stage.SYNTHESIS, {"sprawl"})
    assert dominant.id == "SYN-D2"


def test_select_dominant_returns_epi_d2_for_elegant_theory_drift_risk():
    dominant = select_dominant(Stage.EPISTEMIC, {"elegant_theory_drift"})
    assert dominant.id == "EPI-D2"


def test_select_dominant_returns_syn_d1_with_empty_risk_profile():
    dominant = select_dominant(Stage.SYNTHESIS, set())
    assert dominant.id == "SYN-D1"


def test_rank_failures_by_cost_puts_risk_profile_failures_first():
    dominant = LIBRARY["EXP-D1"]
    ranked = rank_failures_by_cost(dominant, {"fake_divergence"})
    assert ranked[0] == "fake_divergence"


def test_select_suppressions_returns_one_or_two_compatible_lines():
    dominant = LIBRARY["ADV-D1"]
    ranked = rank_failures_by_cost(dominant, {"critique_sludge", "tunnel_vision"})
    suppressions = select_suppressions(dominant, ranked, {"critique_sludge", "tunnel_vision"})

    assert 1 <= len(suppressions) <= 2
    for line in suppressions:
        assert line.id not in dominant.incompatible_with
        assert dominant.id not in line.incompatible_with


def test_select_suppressions_never_returns_incompatible_lines_with_dominant():
    dominant = LIBRARY["SYN-D1"]
    ranked = ["elegant_theory_drift"]

    suppressions = select_suppressions(dominant, ranked, {"elegant_theory_drift"})
    assert suppressions == []


def test_select_shapes_respects_max_two_and_stage_matching():
    dominant = LIBRARY["OPR-D1"]
    shapes = select_shapes(Stage.OPERATOR, dominant, [], set())

    assert len(shapes) <= 2
    for line in shapes:
        assert line.stage == dominant.stage
        assert line.id not in dominant.incompatible_with
        assert dominant.id not in line.incompatible_with


def test_select_tail_returns_transfer_when_handoff_expected_for_exploration():
    tail = select_tail(Stage.EXPLORATION, handoff_expected=True, risk_profile=set())
    assert tail is not None
    assert tail.id == "EXP-T1"
