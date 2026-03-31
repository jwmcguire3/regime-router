from __future__ import annotations

from typing import Dict, List

from router.models import Stage


class StageScoreTracking:
    def __init__(self) -> None:
        self.stage_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        self.lexical_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        self.structural_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}
        self.stage_contributions: Dict[Stage, List[str]] = {stage: [] for stage in Stage}

    def add_score(self, stage: Stage, amount: int, bucket: str, reason: str) -> None:
        if amount <= 0:
            return
        self.stage_scores[stage] += amount
        if bucket == "lexical":
            self.lexical_scores[stage] += amount
        else:
            self.structural_scores[stage] += amount
        self.stage_contributions[stage].append(f"+{amount} {bucket}:{reason}")

    def suppress_score(self, stage: Stage, amount: int, bucket: str, reason: str) -> None:
        if amount <= 0:
            return
        self.stage_scores[stage] = max(0, self.stage_scores[stage] - amount)
        if bucket == "lexical":
            self.lexical_scores[stage] = max(0, self.lexical_scores[stage] - amount)
        else:
            self.structural_scores[stage] = max(0, self.structural_scores[stage] - amount)
        self.stage_contributions[stage].append(f"-{amount} {bucket}:{reason}")


def apply_deterministic_stage_score_overrides(
    *,
    tracking: StageScoreTracking,
    deterministic_stage_scores: Dict[Stage, int],
) -> None:
    for stage in Stage:
        if stage in deterministic_stage_scores:
            override_value = int(deterministic_stage_scores[stage])
            if override_value != tracking.stage_scores[stage]:
                tracking.stage_contributions[stage].append(f"override:external_deterministic_score={override_value}")
            tracking.stage_scores[stage] = override_value
