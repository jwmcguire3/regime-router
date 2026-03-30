import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.classifier import TaskClassifier
from router.models import EmbeddingScore, Stage


PATTERN_DIRECT_CASES = [
    "write breakout game python code",
    "build a REST API for user management",
    "create a landing page with signup form",
    "draft an email to the team about Q3",
    "fix the bug in auth.py line 42",
    "generate a CSV report of monthly sales",
    "implement OAuth2 login flow",
    "deploy the staging environment",
    "translate this document to Spanish",
    "refactor the database module to async",
]

PATTERN_REGIME_CASES = [
    "the fragments are understood but the spine is still missing",
    "stress test this frame for hidden assumptions",
    "what should we do about the pricing decision",
    "explore multiple interpretations of this signal",
    "separate supported claims from unsupported ones",
    "find the strongest interpretation of what this is",
    "this pattern keeps recurring, systematize it",
    "map the space before narrowing",
    "why does this architecture feel fragile",
    "unify the observations into one coherent picture",
]


class FakeEmbeddingRouter:
    REGIME_KEYWORDS = [
        "stress test",
        "frame",
        "spine",
        "interpretation",
        "explore",
        "decision",
        "systematize",
        "evidence",
        "claims",
        "fragile",
        "missing",
        "supported",
        "unsupported",
        "perspectives",
        "options",
        "narrowing",
        "recurring",
        "pattern",
        "unify",
    ]

    def score(self, task: str) -> EmbeddingScore:
        task_lower = task.lower()
        if any(keyword in task_lower for keyword in self.REGIME_KEYWORDS):
            stage_scores = {stage: 0.2 for stage in Stage}
            stage_scores[Stage.SYNTHESIS] = 0.6
            return EmbeddingScore(
                stage_scores=stage_scores,
                best_stage=Stage.SYNTHESIS,
                best_score=0.6,
                below_threshold=False,
            )

        return EmbeddingScore(
            stage_scores={stage: 0.2 for stage in Stage},
            best_stage=Stage.EXPLORATION,
            best_score=0.2,
            below_threshold=True,
        )


@pytest.fixture
def embedding_classifier() -> TaskClassifier:
    return TaskClassifier(embedding_router=FakeEmbeddingRouter())



@pytest.mark.parametrize("task", PATTERN_DIRECT_CASES)
def test_pattern_only_direct_cases(task: str) -> None:
    result = TaskClassifier().classify(task)

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")
    assert result.classification_source in ("pattern", "both")
    assert result.confidence > 0.5


@pytest.mark.parametrize("task", PATTERN_REGIME_CASES)
def test_pattern_only_regime_cases(task: str) -> None:
    result = TaskClassifier().classify(task)

    assert result.route_type == "regime"
    assert result.route_type in ("regime", "direct")


@pytest.mark.parametrize("task", ["hello", "help", ""])
def test_pattern_only_edge_inputs_are_handled(task: str) -> None:
    result = TaskClassifier().classify(task)

    assert isinstance(result.route_type, str)
    assert result.route_type in ("regime", "direct")


def test_pattern_only_mixed_signal_resolves_to_direct() -> None:
    task = "write code to stress test the auth system"

    # Resolved to direct due to an early action+artifact pattern ("write" + "code").
    result = TaskClassifier().classify(task)

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")



@pytest.mark.parametrize("task", PATTERN_DIRECT_CASES)
def test_dual_path_direct_pattern_cases(task: str, embedding_classifier: TaskClassifier) -> None:
    result = embedding_classifier.classify(task)

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")
    assert result.classification_source in ("pattern", "both")
    assert result.confidence > 0.5


@pytest.mark.parametrize("task", PATTERN_REGIME_CASES)
def test_dual_path_regime_cases(task: str, embedding_classifier: TaskClassifier) -> None:
    result = embedding_classifier.classify(task)

    assert result.route_type == "regime"
    assert result.route_type in ("regime", "direct")


@pytest.mark.parametrize("task", ["hello", "help", ""])
def test_dual_path_edge_inputs_are_handled(task: str, embedding_classifier: TaskClassifier) -> None:
    result = embedding_classifier.classify(task)

    assert isinstance(result.route_type, str)
    assert result.route_type in ("regime", "direct")


def test_dual_path_mixed_signal_resolves_to_direct(embedding_classifier: TaskClassifier) -> None:
    task = "write code to stress test the auth system"

    # Resolved to direct due to an early action+artifact pattern ("write" + "code").
    result = embedding_classifier.classify(task)

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")


def test_dual_path_embedding_rejection_summarize_meeting(embedding_classifier: TaskClassifier) -> None:
    result = embedding_classifier.classify("summarize this meeting for me")

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")


def test_dual_path_embedding_rejection_non_regime_task(embedding_classifier: TaskClassifier) -> None:
    result = embedding_classifier.classify("clean up the kitchen schedule")

    assert result.route_type == "direct"
    assert result.route_type in ("regime", "direct")
