from router.models import Stage
from router.routing import Router
from router.runtime import CognitiveRouterRuntime


def _embedding_router_or_skip():
    try:
        from router.embeddings import EmbeddingRouter

        return EmbeddingRouter()
    except Exception as exc:  # pragma: no cover - environment dependent
        import pytest

        pytest.skip(f"sentence-transformers unavailable: {exc}")


def test_embedding_router_adversarial_highest_for_stress_test_phrase() -> None:
    embedding_router = _embedding_router_or_skip()
    result = embedding_router.score("stress test this frame")
    assert result.best_stage == Stage.ADVERSARIAL


def test_embedding_router_marks_non_regime_task_below_threshold() -> None:
    embedding_router = _embedding_router_or_skip()
    result = embedding_router.score("write python code for a game")
    assert result.below_threshold is True


def test_embedding_router_exploration_highest_for_structural_options() -> None:
    embedding_router = _embedding_router_or_skip()
    result = embedding_router.score("what are the structurally distinct options")
    assert result.best_stage == Stage.EXPLORATION


def test_embedding_router_operator_highest_for_concrete_decision() -> None:
    embedding_router = _embedding_router_or_skip()
    result = embedding_router.score("produce a concrete decision with next steps")
    assert result.best_stage == Stage.OPERATOR


def test_embedding_router_scores_are_bounded() -> None:
    embedding_router = _embedding_router_or_skip()
    result = embedding_router.score("stress test this frame")
    assert isinstance(result.best_score, float)
    for score in result.stage_scores.values():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


def test_router_works_without_embedding_router() -> None:
    router = Router(embedding_router=None)
    decision = router.route("Choose between these two close options and justify the decision.")
    assert decision.primary_regime is not None


def test_runtime_can_disable_embedding_router() -> None:
    runtime = CognitiveRouterRuntime(use_embedding_router=False)
    decision, regime, handoff = runtime.plan("Choose one option and define next steps.")
    assert decision.primary_regime is not None
    assert regime.stage == decision.primary_regime
    assert handoff.recommended_next_regime == decision.runner_up_regime
