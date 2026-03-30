import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.classifier import TaskClassifier
from router.models import EmbeddingScore, Stage


class FakeEmbeddingRouter:
    def __init__(self) -> None:
        self.responses = {
            "write breakout game python code": EmbeddingScore(
                stage_scores={stage: 0.2 for stage in Stage},
                best_stage=Stage.BUILDER,
                best_score=0.2,
                below_threshold=True,
            ),
            "summarize this meeting for me": EmbeddingScore(
                stage_scores={stage: 0.18 for stage in Stage},
                best_stage=Stage.SYNTHESIS,
                best_score=0.18,
                below_threshold=True,
            ),
            "stress test this frame": EmbeddingScore(
                stage_scores={
                    Stage.ADVERSARIAL: 0.72,
                    Stage.EXPLORATION: 0.14,
                    Stage.SYNTHESIS: 0.11,
                    Stage.EPISTEMIC: 0.21,
                    Stage.OPERATOR: 0.16,
                    Stage.BUILDER: 0.09,
                },
                best_stage=Stage.ADVERSARIAL,
                best_score=0.72,
                below_threshold=False,
            ),
            "the fragments are understood but the spine is missing": EmbeddingScore(
                stage_scores={
                    Stage.SYNTHESIS: 0.68,
                    Stage.EXPLORATION: 0.13,
                    Stage.ADVERSARIAL: 0.08,
                    Stage.EPISTEMIC: 0.19,
                    Stage.OPERATOR: 0.12,
                    Stage.BUILDER: 0.16,
                },
                best_stage=Stage.SYNTHESIS,
                best_score=0.68,
                below_threshold=False,
            ),
        }

    def score(self, task: str) -> EmbeddingScore:
        return self.responses[task]



def test_classifies_write_breakout_game_python_code_as_direct_from_pattern() -> None:
    classifier = TaskClassifier(embedding_router=FakeEmbeddingRouter())
    result = classifier.classify("write breakout game python code")

    assert result.route_type == "direct"
    assert result.confidence == 0.92



def test_classifies_summarize_meeting_as_direct_from_embedding_rejection() -> None:
    classifier = TaskClassifier(embedding_router=FakeEmbeddingRouter())
    result = classifier.classify("summarize this meeting for me")

    assert result.route_type == "direct"
    assert result.confidence == 0.6
    assert result.reason == "No regime has semantic affinity above threshold"



def test_classifies_stress_test_frame_as_regime_when_embedding_affinity_is_high() -> None:
    classifier = TaskClassifier(embedding_router=FakeEmbeddingRouter())
    result = classifier.classify("stress test this frame")

    assert result.route_type == "regime"



def test_classifies_fragments_spine_statement_as_regime_with_synthesis_affinity() -> None:
    classifier = TaskClassifier(embedding_router=FakeEmbeddingRouter())
    result = classifier.classify("the fragments are understood but the spine is missing")

    assert result.route_type == "regime"
