import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.routing import Router


class TestExplorationRouting:
    CASES = [
        (
            "brainstorm alternatives before we commit",
            Stage.EXPLORATION,
            Stage.OPERATOR,
        ),
        (
            "explore multiple interpretations and map the space before narrowing",
            Stage.EXPLORATION,
            Stage.SYNTHESIS,
        ),
        (
            "keep it open before narrowing and explore multiple perspectives",
            Stage.EXPLORATION,
            Stage.SYNTHESIS,
        ),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_exploration_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up


class TestSynthesisRouting:
    INTERPRETATION_SHORTCUT_CASES = [
        (
            "find the strongest interpretation of these conflicting notes",
            Stage.SYNTHESIS,
            Stage.ADVERSARIAL,
        ),
        (
            "identify the strongest frame for what this actually is",
            Stage.SYNTHESIS,
            Stage.ADVERSARIAL,
        ),
        (
            "tell me what this actually is before anything else",
            Stage.SYNTHESIS,
            Stage.ADVERSARIAL,
        ),
    ]
    STRUCTURAL_SIGNAL_CASES = [
        (
            "the parts are understood but the whole is missing",
            Stage.SYNTHESIS,
            Stage.EPISTEMIC,
        ),
        (
            "components are clear; the core is missing even though parts are understood",
            Stage.SYNTHESIS,
            Stage.EPISTEMIC,
        ),
        (
            "pieces make sense but the backbone is missing while the parts are understood",
            Stage.SYNTHESIS,
            Stage.EPISTEMIC,
        ),
    ]
    GENERIC_SYNTHESIS_CASES = [
        (
            "unify the observations into one coherent picture",
            Stage.SYNTHESIS,
            Stage.EXPLORATION,
        ),
        (
            "pull this into one coherent picture from many signals",
            Stage.SYNTHESIS,
            Stage.EXPLORATION,
        ),
        (
            "there are many signals but no center yet",
            Stage.SYNTHESIS,
            Stage.EXPLORATION,
        ),
    ]

    @pytest.mark.parametrize(
        "task,expected_primary,expected_runner_up",
        INTERPRETATION_SHORTCUT_CASES + STRUCTURAL_SIGNAL_CASES + GENERIC_SYNTHESIS_CASES,
    )
    def test_synthesis_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", INTERPRETATION_SHORTCUT_CASES)
    def test_interpretation_shortcuts_are_high_confidence(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.confidence.level == "high"


class TestEpistemicRouting:
    CASES = [
        (
            "separate supported from unsupported claims and verify the claims",
            Stage.EPISTEMIC,
            Stage.EXPLORATION,
        ),
        (
            "what evidence is missing before we proceed",
            Stage.EPISTEMIC,
            Stage.EXPLORATION,
        ),
        (
            "can't tell what kind of issue this is; hard to characterize",
            Stage.EPISTEMIC,
            Stage.SYNTHESIS,
        ),
        (
            "we can't tell what kind of risk this is yet",
            Stage.EPISTEMIC,
            Stage.SYNTHESIS,
        ),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_epistemic_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up


class TestAdversarialRouting:
    SHORTCUT_CASES = [
        ("stress test this frame before launch", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("find the weakest points in this proposal", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("attack this frame and show where this breaks", Stage.ADVERSARIAL, Stage.EPISTEMIC),
    ]
    FRAGILITY_CASES = [
        ("list failure modes for this rollout", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("identify vulnerabilities in the approach", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("describe how this could fail under pressure", Stage.ADVERSARIAL, Stage.EPISTEMIC),
    ]

    @pytest.mark.parametrize(
        "task,expected_primary,expected_runner_up",
        SHORTCUT_CASES + FRAGILITY_CASES,
    )
    def test_adversarial_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", SHORTCUT_CASES)
    def test_adversarial_shortcuts_are_high_confidence(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.confidence.level == "high"


class TestOperatorRouting:
    CASES = [
        ("decide now between these two plans", Stage.OPERATOR, Stage.EXPLORATION),
        ("choose between options and make a call", Stage.OPERATOR, Stage.EXPLORATION),
        ("what should we do right now", Stage.OPERATOR, Stage.EXPLORATION),
        ("best option now with clear tradeoff between options", Stage.OPERATOR, Stage.EXPLORATION),
        ("evaluate opportunity cost and choose between options", Stage.OPERATOR, Stage.EXPLORATION),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_operator_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up


class TestBuilderRouting:
    SHORTCUT_CASES = [
        ("make this process repeatable", Stage.BUILDER, Stage.OPERATOR),
        ("create a reusable template for this workflow", Stage.BUILDER, Stage.OPERATOR),
        ("systematize and productize this into modules and interfaces", Stage.BUILDER, Stage.OPERATOR),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", SHORTCUT_CASES)
    def test_builder_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", SHORTCUT_CASES)
    def test_builder_shortcuts_are_high_confidence(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.confidence.level == "high"


class TestWordBoundaryRegressions:
    CASES = [
        ("write breakout game python code", Stage.EXPLORATION, None),
        ("known facts about the project", Stage.EXPLORATION, None),
        ("breakfast meeting agenda", Stage.EXPLORATION, None),
        ("break under pressure", Stage.ADVERSARIAL, Stage.EPISTEMIC),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_word_boundary_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up


class TestZeroScoreFallback:
    CASES = [
        ("please help with this task", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("hmm", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("interesting", Stage.EXPLORATION, Stage.SYNTHESIS),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_zero_score_fallback_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up


class TestNegatedClosure:
    CASES = [
        ("do not decide yet, map the options", Stage.EXPLORATION, Stage.OPERATOR),
        ("don't recommend, explore alternatives", Stage.EXPLORATION, Stage.OPERATOR),
        ("do not choose yet; keep it open before narrowing", Stage.EXPLORATION, Stage.SYNTHESIS),
    ]

    @pytest.mark.parametrize("task,expected_primary,expected_runner_up", CASES)
    def test_negated_closure_cases(self, task, expected_primary, expected_runner_up):
        decision = Router().route(task)
        assert decision.primary_regime == expected_primary
        if expected_runner_up is not None:
            assert decision.runner_up_regime == expected_runner_up
