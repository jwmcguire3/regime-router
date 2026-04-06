import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage, TaskAnalyzerOutput


@pytest.fixture
def analyzer_output_fixture() -> TaskAnalyzerOutput:
    return TaskAnalyzerOutput(
        bottleneck_label="structural mismatch",
        candidate_regimes=[Stage.SYNTHESIS, Stage.EPISTEMIC],
        stage_scores={Stage.SYNTHESIS: 0.84, Stage.EPISTEMIC: 0.66},
        structural_signals=["expansion_when_defined", "fragments_without_spine"],
        decision_pressure=6,
        fragility_pressure=4,
        possibility_space_need=7,
        synthesis_pressure=5,
        evidence_quality=3,
        recurrence_potential=2,
        confidence=0.79,
        rationale="Signals indicate synthesis-first routing with epistemic backup.",
        risk_tags=["high_stakes", "reversible_commitment_needed"],
        likely_endpoint_regime=Stage.OPERATOR,
        endpoint_confidence=0.71,
    )
