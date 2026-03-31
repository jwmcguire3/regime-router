import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.analyzer import TaskAnalyzer
from router.cli import main
from router.models import RoutingFeatures, Stage, TaskAnalyzerOutput
from router.prompts import PromptBuilder
from router.routing import RegimeComposer, extract_routing_features, extract_structural_signals, infer_risk_profile
from router.runtime import CognitiveRuntime
from router.storage import SessionStore
from router.state import make_record
from router.validation import OutputValidator


STRUCTURAL_TASK = (
    "Find the strongest interpretation of what this actually is. "
    "When we define the effort it expands instead of narrowing. "
    "Concrete versions feel too small. "
    "The fragments are understood, but the spine is still missing; "
    "a single frame compresses too early."
)

def _route_with_mocked_analyzer(task: str, primary: Stage, runner_up: Stage = Stage.SYNTHESIS, confidence: float = 0.85):
    features = extract_routing_features(task)
    stage_scores = {stage: 0.1 for stage in Stage}
    stage_scores[primary] = 0.9
    if runner_up != primary:
        stage_scores[runner_up] = 0.6
    output = TaskAnalyzerOutput(
        bottleneck_label=task,
        candidate_regimes=[primary, runner_up],
        stage_scores=stage_scores,
        structural_signals=features.structural_signals,
        decision_pressure=features.decision_pressure,
        evidence_quality=features.evidence_demand,
        recurrence_potential=features.recurrence_potential,
        confidence=confidence,
        rationale='Mocked analyzer proposal for outcome test.',
    )
    analyzer = TaskAnalyzer(MagicMock(), model='test')
    analyzer.analyze = MagicMock(return_value=output)
    return analyzer.propose_route(task, features, features.structural_signals, set())


@pytest.mark.parametrize(
    'task,primary,runner_up',
    [
        ("Find the strongest interpretation of what this actually is.", Stage.SYNTHESIS, Stage.ADVERSARIAL),
        ("What is the strongest interpretation, and what would break it?", Stage.SYNTHESIS, Stage.ADVERSARIAL),
        ("What is the strongest interpretation, and what evidence is missing?", Stage.SYNTHESIS, Stage.EPISTEMIC),
        ("What is the strongest interpretation, and verify which parts are actually supported?", Stage.SYNTHESIS, Stage.EPISTEMIC),
        ("I need support and rigor before we commit.", Stage.EPISTEMIC, Stage.SYNTHESIS),
        ("Stress test this frame and break it before launch.", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("What are the weakest points in this interpretation?", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("What are the strongest objections to this frame?", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("What evidence supports this interpretation?", Stage.EPISTEMIC, Stage.SYNTHESIS),
        ("What would break this frame?", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("Choose between these two close options and justify the decision.", Stage.OPERATOR, Stage.EXPLORATION),
        ("What remains unknown or unclear here?", Stage.EPISTEMIC, Stage.SYNTHESIS),
        ("The parts are legible, but the whole organizing logic is missing.", Stage.SYNTHESIS, Stage.EPISTEMIC),
        ("We have multiple options, but we need a decision and next move now.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Brainstorm some options.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("Explore a few alternatives, then choose the best option now.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Explore several frames before selecting one.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Map the space before narrowing. Give multiple possible frames, perspectives, or interpretations of the situation. Keep it open rather than converging on one answer.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("Choose one course of action now. Make a decision, explain the tradeoff, and give a fallback trigger.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Map the possibility space quickly, then choose one path now with a clear tradeoff and fallback trigger.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Give me the main perspectives, but end with a recommendation for what we should do now.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Give multiple perspectives, but do not decide yet.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("Give multiple perspectives, then decide.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Do not recommend yet; map the options first.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("Recommend what we should do now.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Map the options quickly and make a call.", Stage.OPERATOR, Stage.EXPLORATION),
        ("Give multiple frames and interpretations, and keep it open rather than converging, even if a decision could be made now.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("Map the space and give multiple perspectives. Keep it open and delay convergence before narrowing.", Stage.EXPLORATION, Stage.SYNTHESIS),
        ("There’s a pattern here but I can’t tell what kind.", Stage.EPISTEMIC, Stage.SYNTHESIS),
        ("Stress test this frame for launch certainty before deployment.", Stage.ADVERSARIAL, Stage.EPISTEMIC),
        ("There is a reusable pattern here we should turn into a template.", Stage.BUILDER, Stage.OPERATOR),
        ("This is becoming a repeatable workflow.", Stage.BUILDER, Stage.OPERATOR),
        ("We can describe concrete versions and fragments, but no center is holding. A single frame compresses too early.", Stage.SYNTHESIS, Stage.EPISTEMIC),
        ("Find the strongest interpretation and also verify which evidence supports it.", Stage.EPISTEMIC, Stage.SYNTHESIS),
    ],
)
def test_outcome_routes_match_expected_stage(task, primary, runner_up):
    decision = _route_with_mocked_analyzer(task, primary, runner_up)
    assert decision.primary_regime == primary


class FakeOllama:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, model, system, prompt, stream=False, temperature=0.2, num_predict=1200):
        self.calls.append({"model": model, "system": system, "prompt": prompt})
        if not self._responses:
            raise AssertionError("No fake response left for generate().")
        return {"response": self._responses.pop(0)}


@pytest.fixture
def synthesis_ok_json() -> str:
    payload = {
        "regime": "Synthesis Core",
        "purpose": "Produce the strongest coherent interpretation from live signals.",
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
        "completion_signal": "coherent_frame_stable",
        "failure_signal": "frame_collapses_under_pressure_points",
        "recommended_next_regime": "adversarial",
        "artifact": {
            "central_claim": "The frame must expand when defined because the concrete view shrinks the real structure.",
            "organizing_idea": "Definition changes the unit of analysis: it connects fragments into a spine, which makes narrow concrete cuts look too small.",
            "key_tensions": [
                "Define vs expand: each tighter definition reveals more hidden dependencies.",
                "Concrete instance vs structural spine: local detail can hide the cross-cutting backbone.",
            ],
            "supporting_structure": [
                "The task says fragments are understood yet the spine is missed, which implies a linkage-level bottleneck.",
                "Concrete versions feeling too small indicates the structure is larger than any single instance boundary.",
            ],
            "pressure_points": [
                "If a concrete example preserves the same spine without expansion, the expand-when-defined claim weakens.",
                "If fragments stop mapping to a shared spine, the organizing interpretation fails.",
            ],
        },
    }
    return json.dumps(payload)


@pytest.fixture
def synthesis_polished_but_generic_json() -> str:
    payload = {
        "regime": "Synthesis Core",
        "purpose": "Produce the strongest coherent interpretation from live signals.",
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
        "completion_signal": "coherent_frame_stable",
        "failure_signal": "frame_collapses_under_pressure_points",
        "recommended_next_regime": "adversarial",
        "artifact": {
            "central_claim": "This effort is about navigating complexity with careful consideration across multiple perspectives.",
            "organizing_idea": "The project requires deeper analysis and understanding of various factors over time.",
            "key_tensions": ["Balancing many factors while assessing broad complexity."],
            "supporting_structure": ["A systemic approach is needed for understanding evolving dynamics."],
            "pressure_points": ["Execution may fail due to roadmap and timeline coordination pressures."],
        },
    }
    return json.dumps(payload)


@pytest.fixture
def synthesis_bad_pressure_points_json() -> str:
    payload = {
        "regime": "Synthesis Core",
        "purpose": "Produce the strongest coherent interpretation from live signals.",
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
        "completion_signal": "coherent_frame_stable",
        "failure_signal": "frame_collapses_under_pressure_points",
        "recommended_next_regime": "adversarial",
        "artifact": {
            "central_claim": "Defining the work expands scope because each concrete cut removes the spine-level relation.",
            "organizing_idea": "Fragments make sense locally, but a shared spine only appears when the frame tracks expansion under definition.",
            "key_tensions": ["Concrete smallness versus spine coherence across fragments."],
            "supporting_structure": ["The task directly links fragments understood with spine missed under single-frame compression."],
            "pressure_points": [
                "Implementation risk: timeline, coordination, and resourcing may block execution quality."
            ],
        },
    }
    return json.dumps(payload)

def test_synthesis_prompt_contains_anchor_contract_and_role_guidance():
    task_signals = extract_structural_signals(STRUCTURAL_TASK)
    risks = infer_risk_profile(STRUCTURAL_TASK, set())
    regime = RegimeComposer().compose(Stage.SYNTHESIS, risk_profile=risks)

    prompt = PromptBuilder.build_system_prompt(regime, task_signals=task_signals, risk_profile=risks)

    assert "required anchors" in prompt
    assert "must reinterpret, connect, or test those anchors" in prompt
    assert "signal-anchored observations tied to exact extracted signals" in prompt
    for field_name in [
        "central_claim",
        "organizing_idea",
        "key_tensions",
        "supporting_structure",
        "pressure_points",
    ]:
        assert field_name in prompt


def test_validator_accepts_grounded_synthesis_artifact(synthesis_ok_json):
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        synthesis_ok_json,
        task=STRUCTURAL_TASK,
        task_signals=extract_structural_signals(STRUCTURAL_TASK),
        risk_profile=infer_risk_profile(STRUCTURAL_TASK, set()),
    )
    assert validation["is_valid"] is True
    assert validation["semantic_failures"] == []


def test_validator_rejects_polished_but_generic_synthesis_artifact(synthesis_polished_but_generic_json):
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        synthesis_polished_but_generic_json,
        task=STRUCTURAL_TASK,
        task_signals=extract_structural_signals(STRUCTURAL_TASK),
        risk_profile=infer_risk_profile(STRUCTURAL_TASK, set()),
    )

    assert validation["is_valid"] is False
    failures = "\n".join(validation["semantic_failures"])
    assert "generic filler" in failures or "forbidden generic domain nouns" in failures
    assert "stage" not in failures


def test_validator_rejects_pressure_points_as_execution_risks(synthesis_bad_pressure_points_json):
    validation = OutputValidator().validate(
        Stage.SYNTHESIS,
        synthesis_bad_pressure_points_json,
        task=STRUCTURAL_TASK,
        task_signals=extract_structural_signals(STRUCTURAL_TASK),
        risk_profile=infer_risk_profile(STRUCTURAL_TASK, set()),
    )

    assert validation["is_valid"] is False
    failures = "\n".join(validation["semantic_failures"])
    assert "pressure_points use generic execution language" in failures



@pytest.mark.parametrize(
    "task",
    [
        "Find the strongest interpretation of what this actually is.",
        "Are you sure we have enough evidence, or are key unknowns unresolved?",
        "Stress test this framing and break it before launch.",
        "We need to decide the next move now under time pressure.",
    ],
)
def test_smoke_main_plan_entrypoint_runs_without_crashing(task, capsys):
    rc = main(["plan", "--task", task, "--no-use-task-analyzer"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "=== Routing summary ===" in out
    assert "Regime:" in out

def _analyzer_valid_payload() -> dict:
    return {
        "bottleneck_label": "unclear bottleneck",
        "candidate_regimes": ["epistemic", "operator"],
        "stage_scores": {
            "exploration": 0.1,
            "synthesis": 0.2,
            "epistemic": 0.9,
            "adversarial": 0.1,
            "operator": 0.7,
            "builder": 0.0,
        },
        "structural_signals": ["expansion_when_defined"],
        "decision_pressure": 4,
        "evidence_quality": 2,
        "recurrence_potential": 1,
        "confidence": 0.8,
        "rationale": "Evidence uncertainty dominates.",
    }


def test_plan_debug_routing_flag_prints_observability_details(capsys):
    rc = main(["plan", "--task", "Can you help?", "--debug-routing", "--no-use-task-analyzer"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "=== Debug ===" in out
    assert "Feature pressures" in out
    assert "Analyzer state" in out


def test_plan_populates_router_state():
    runtime = CognitiveRuntime()
    decision, regime, handoff = runtime.plan("Find the strongest interpretation of what this actually is.")

    assert runtime.router_state is not None
    assert runtime.router_state.current_regime.name == regime.name
    assert runtime.router_state.current_bottleneck == decision.bottleneck
    assert runtime.router_state.switch_trigger == decision.switch_trigger
    assert runtime.router_state.runner_up_regime is not None
    assert runtime.router_state.runner_up_regime.stage == decision.runner_up_regime
    assert runtime.router_state.recommended_next_regime is not None
    assert runtime.router_state.recommended_next_regime.stage == decision.runner_up_regime
    assert handoff.current_bottleneck == runtime.router_state.current_bottleneck
    assert handoff.assumptions_in_play == runtime.router_state.assumptions
    assert handoff.recommended_next_regime == decision.runner_up_regime


def test_execute_populates_router_state_and_prior_regimes(synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json])

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    assert runtime.router_state.current_regime.name == regime.name
    assert runtime.router_state.prior_regimes[-1].regime.stage == decision.primary_regime
    assert runtime.router_state.prior_regimes[-1].completion_signal_seen is True
    assert "Execution yielded a valid artifact." in runtime.router_state.prior_regimes[-1].outcome_summary
    assert handoff.dominant_frame == (runtime.router_state.dominant_frame or "")
    assert result.validation["is_valid"] is True


def test_router_state_serializes_in_saved_runs(tmp_path, synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json])
    store = SessionStore(root=str(tmp_path))

    decision, regime, result, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")
    record = make_record(STRUCTURAL_TASK, set(), "fake", decision, regime, result, handoff, runtime.router_state)
    saved = store.save(record, filename="stateful.json")
    loaded = store.load(saved.name)

    assert loaded["router_state"] is not None
    assert loaded["router_state"]["current_regime"]["stage"] == decision.primary_regime.value
    assert loaded["router_state"]["runner_up_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["router_state"]["recommended_next_regime"]["stage"] == decision.runner_up_regime.value
    assert loaded["router_state"]["prior_regimes"][0]["regime"]["stage"] == decision.primary_regime.value


def test_router_state_prior_regimes_helper_updates_correctly():
    runtime = CognitiveRuntime()
    runtime.plan("We have multiple options, but we need a decision and next move now.")
    assert runtime.router_state is not None

    runtime.router_state.record_regime_step(
        regime=runtime.composer.compose(Stage.OPERATOR),
        reason_entered="Operator pressure remains dominant.",
        completion_signal_seen=False,
        failure_signal_seen=True,
        outcome_summary="Escalated due to blocked decision criteria.",
    )
    step = runtime.router_state.prior_regimes[-1]
    assert step.regime.stage == Stage.OPERATOR
    assert step.failure_signal_seen is True
    assert step.completion_signal_seen is False


def test_contradictions_and_assumptions_persist_across_simple_transition(synthesis_bad_pressure_points_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_bad_pressure_points_json, synthesis_bad_pressure_points_json])
    runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    assert "Soft LLM behavior vs hard system control" in runtime.router_state.contradictions
    assert any("Validation semantic failures were observed" in item for item in runtime.router_state.assumptions)


def test_handoff_remains_router_state_backed(synthesis_ok_json):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([synthesis_ok_json])

    _, _, _, handoff = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert runtime.router_state is not None
    assert handoff.active_contradictions == runtime.router_state.contradictions
    assert handoff.assumptions_in_play == runtime.router_state.assumptions
