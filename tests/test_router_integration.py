import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cognitive_router_prototype import (
    CognitiveRuntime,
    PromptBuilder,
    RegimeComposer,
    RoutingFeatures,
    Router,
    Stage,
    OutputValidator,
    extract_routing_features,
    extract_structural_signals,
    infer_risk_profile,
    main,
)


STRUCTURAL_TASK = (
    "Find the strongest interpretation of what this actually is. "
    "When we define the effort it expands instead of narrowing. "
    "Concrete versions feel too small. "
    "The fragments are understood, but the spine is still missing; "
    "a single frame compresses too early."
)


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
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
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
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
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
        "stage": "synthesis",
        "artifact_type": "dominant_frame",
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


def test_routing_strongest_interpretation_routes_synthesis_then_adversarial():
    decision = Router().route("Find the strongest interpretation of what this actually is.")
    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.runner_up_regime == Stage.ADVERSARIAL


def test_routing_strongest_interpretation_plus_break_prompt_keeps_adversarial_runner_up():
    decision = Router().route("What is the strongest interpretation, and what would break it?")
    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.runner_up_regime == Stage.ADVERSARIAL


def test_routing_strongest_interpretation_plus_evidence_missing_allows_epistemic_runner_up():
    decision = Router().route("What is the strongest interpretation, and what evidence is missing?")
    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.runner_up_regime == Stage.EPISTEMIC


def test_routing_strongest_interpretation_plus_verify_supported_allows_epistemic_runner_up():
    decision = Router().route(
        "What is the strongest interpretation, and verify which parts are actually supported?"
    )
    assert decision.primary_regime == Stage.SYNTHESIS
    assert decision.runner_up_regime == Stage.EPISTEMIC


def test_routing_uncertainty_seeking_can_route_epistemic_when_rigor_language_present():
    decision = Router().route("I need support and rigor before we commit.")
    assert decision.primary_regime == Stage.EPISTEMIC


def test_routing_stress_test_routes_adversarial():
    decision = Router().route("Stress test this frame and break it before launch.")
    assert decision.primary_regime == Stage.ADVERSARIAL
    assert decision.confidence.level == "high"


def test_routing_choose_between_options_routes_operator_not_exploration():
    decision = Router().route("Choose between these two close options and justify the decision.")
    assert decision.primary_regime == Stage.OPERATOR


def test_routing_unknown_or_unclear_routes_epistemic_not_exploration():
    decision = Router().route("What remains unknown or unclear here?")
    assert decision.primary_regime == Stage.EPISTEMIC


def test_routing_parts_whole_spine_gap_routes_synthesis_not_exploration():
    decision = Router().route("The parts are legible, but the whole organizing logic is missing.")
    assert decision.primary_regime == Stage.SYNTHESIS


def test_routing_options_plus_decision_prefers_operator():
    decision = Router().route("We have multiple options, but we need a decision and next move now.")
    assert decision.primary_regime == Stage.OPERATOR


def test_routing_structural_signals_can_drive_synthesis_precedence():
    task = (
        "We can describe concrete versions and fragments, but no center is holding."
        " A single frame compresses too early."
    )
    signals = extract_structural_signals(task)
    risks = infer_risk_profile(task, set())
    decision = Router().route(task, task_signals=signals, risk_profile=risks)
    assert decision.primary_regime == Stage.SYNTHESIS


def test_extract_routing_features_returns_typed_inspectable_feature_object():
    task = (
        "Parts are clear but the whole backbone is missing. "
        "We need evidence before we commit now, and we should stress test before launch. "
        "Also make it repeatable with a reusable template while keeping exploration open."
    )
    features = extract_routing_features(task)

    assert isinstance(features, RoutingFeatures)
    assert "parts_whole_mismatch" in features.detected_markers
    assert features.evidence_demand >= 1
    assert features.decision_pressure >= 1
    assert features.fragility_pressure >= 1
    assert features.recurrence_potential >= 1
    assert features.possibility_space_need >= 1


def test_router_can_consume_precomputed_routing_features():
    task = "We should choose now, but first verify unknowns and evidence gaps."
    features = extract_routing_features(task)
    decision = Router().route(task, routing_features=features)
    assert decision.primary_regime in {Stage.OPERATOR, Stage.EPISTEMIC}


def test_confidence_is_medium_for_close_mixed_prompt():
    decision = Router().route("We should choose now, but first verify unknowns and evidence gaps.")
    assert decision.confidence.level == "medium"
    assert "close" in decision.confidence.rationale or "sequencing" in decision.confidence.rationale


def test_confidence_is_low_for_weak_underspecified_prompt():
    decision = Router().route("Can you help?")
    assert decision.confidence.level == "low"
    assert decision.confidence.top_stage_score == 0


def test_structural_signals_and_risk_profile_plumbed_into_prompts_and_synthesis_suppressions(synthesis_ok_json):
    runtime = CognitiveRuntime()
    fake = FakeOllama([synthesis_ok_json])
    runtime.ollama = fake

    decision, regime, result, _ = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert decision.primary_regime == Stage.SYNTHESIS
    assert any(line.id == "SYN-P2" for line in regime.suppression_lines)

    signals = extract_structural_signals(STRUCTURAL_TASK)
    risks = infer_risk_profile(STRUCTURAL_TASK, set())

    assert "abstract_structural_task" in risks
    assert "false_unification" in risks

    system_prompt = fake.calls[0]["system"]
    user_prompt = fake.calls[0]["prompt"]

    for sig in signals:
        assert sig.replace("_", " ") in system_prompt
        assert sig in user_prompt

    assert "Active risk profile:" in system_prompt
    assert "false_unification" in system_prompt
    assert result.validation["is_valid"] is True


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
    "first_response,expected_mode",
    [
        ("{not-json", PromptBuilder.REPAIR_MODE_SCHEMA),
        (
            json.dumps(
                {
                    "regime": "Synthesis Core",
                    "stage": "synthesis",
                    "artifact_type": "dominant_frame",
                    "artifact": {
                        "central_claim": "This effort is about navigating complexity across various factors and multiple perspectives.",
                        "organizing_idea": "A deeper analysis helps understanding of systemic issues over time.",
                        "key_tensions": ["Careful consideration is needed while assessing complexity broadly."],
                        "supporting_structure": ["A generic solution aligns the team and stakeholders around innovation."],
                        "pressure_points": ["Implementation roadmap risks and timeline coordination."],
                    },
                }
            ),
            PromptBuilder.REPAIR_MODE_REDUCE_GENERICITY,
        ),
        (
            json.dumps(
                {
                    "regime": "Synthesis Core",
                    "stage": "synthesis",
                    "artifact_type": "dominant_frame",
                    "artifact": {
                        "central_claim": "Define and expand the frame through fragments and spine linkage.",
                        "organizing_idea": "Define and expand the frame through fragments and spine linkage.",
                        "key_tensions": ["Fragment versus spine in concrete boundaries."],
                        "supporting_structure": ["Concrete slices become small while definition expands scope."],
                        "pressure_points": ["If fragment-spine mapping fails, this frame weakens."],
                    },
                }
            ),
            PromptBuilder.REPAIR_MODE_SEMANTIC,
        ),
    ],
)
def test_repair_dispatch_selects_expected_mode(first_response, expected_mode, synthesis_ok_json, monkeypatch):
    runtime = CognitiveRuntime()
    runtime.ollama = FakeOllama([first_response, synthesis_ok_json])

    selected_modes = []
    original = runtime.prompt_builder.build_repair_prompt

    def capture_mode(*args, **kwargs):
        selected_modes.append(kwargs.get("repair_mode"))
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.prompt_builder, "build_repair_prompt", capture_mode)

    _, _, result, _ = runtime.execute(task=STRUCTURAL_TASK, model="fake")

    assert selected_modes == [expected_mode]
    assert result.validation["repair_attempted"] is True


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
    rc = main(["plan", "--task", task])
    out = capsys.readouterr().out

    assert rc == 0
    assert "ROUTING HEADER" in out
    assert "Regime:" in out
