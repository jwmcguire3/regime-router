from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional, Protocol

from .models import EmbeddingScore


class EmbeddingRouterLike(Protocol):
    def score(self, task: str) -> EmbeddingScore:
        ...


@dataclass(frozen=True)
class TaskClassification:
    route_type: str
    confidence: float
    reason: str
    classification_source: str


class TaskClassifier:
    EMBEDDING_THRESHOLD = 0.35

    ACTION_VERBS = (
        "write",
        "create",
        "build",
        "make",
        "generate",
        "draft",
        "code",
        "implement",
        "design",
        "draw",
        "compose",
        "translate",
        "convert",
        "fix",
        "refactor",
        "debug",
        "test",
        "deploy",
        "install",
        "set up",
        "configure",
    )

    ARTIFACT_NOUNS = (
        "code",
        "script",
        "function",
        "class",
        "module",
        "app",
        "application",
        "page",
        "component",
        "email",
        "letter",
        "document",
        "report",
        "presentation",
        "spreadsheet",
        "database",
        "api",
        "endpoint",
        "test",
        "game",
        "website",
        "dashboard",
        "chart",
        "diagram",
        "file",
        "bug",
        "login",
        "flow",
        "environment",
    )

    def __init__(self, embedding_router: Optional[EmbeddingRouterLike] = None) -> None:
        self.embedding_router = embedding_router

    def classify(self, task: str) -> TaskClassification:
        text = task.lower().strip()
        if self._has_action_artifact_pattern_near_start(text):
            return TaskClassification(
                route_type="direct",
                confidence=0.92,
                reason="Imperative action+artifact request detected near task start.",
                classification_source="pattern",
            )
        if self.embedding_router is not None:
            embedding_score = self.embedding_router.score(task)
            has_semantic_affinity = any(
                score > self.EMBEDDING_THRESHOLD for score in embedding_score.stage_scores.values()
            )
            if embedding_score.below_threshold and not has_semantic_affinity:
                return TaskClassification(
                    route_type="direct",
                    confidence=0.6,
                    reason="No regime has semantic affinity above threshold",
                    classification_source="embedding",
                )
        return TaskClassification(
            route_type="regime",
            confidence=0.85,
            reason="No early action+artifact imperative pattern detected.",
            classification_source="fallback",
        )

    def _has_action_artifact_pattern_near_start(self, text: str) -> bool:
        words = self._tokenize_words(text)
        if not words:
            return False

        search_limit = min(10, len(words))
        first_words = words[:search_limit]
        segment = " ".join(first_words)

        for action in self.ACTION_VERBS:
            action_tokens = action.split()
            action_len = len(action_tokens)
            for i in range(search_limit):
                if i + action_len > search_limit:
                    continue
                if first_words[i : i + action_len] != action_tokens:
                    continue
                for noun in self.ARTIFACT_NOUNS:
                    noun_match = re.search(rf"\b{re.escape(noun)}\b", segment)
                    if noun_match is None:
                        continue
                    noun_word_index = len(segment[: noun_match.start()].split())
                    if noun_word_index >= i + action_len and noun_word_index < search_limit:
                        return True
        return False

    @staticmethod
    def _tokenize_words(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text)
