import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from router.models import Stage
from router.orchestration.misrouting_detector import MisroutingDetectionResult
from router.runtime import CognitiveRuntime


class FakeModelClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("No more mocked model responses available")
        return {"response": self._responses.pop(0)}

    def list_models(self):
        return {"models": []}


@pytest.fixture
def runtime_factory(monkeypatch):
    def _make(responses):
        runtime = CognitiveRuntime(provider="ollama")
        fake_client = FakeModelClient(responses)
        runtime.ollama = fake_client

        def _always_valid(stage, raw_text, **_kwargs):
            parsed = json.loads(raw_text)
            return {
                "is_valid": True,
                "valid_json": True,
                "required_keys_present": True,
                "artifact_fields_present": True,
                "artifact_type_matches": True,
                "contract_controls_valid": True,
                "semantic_valid": True,
                "parsed": parsed,
            }

        monkeypatch.setattr(runtime.validator, "validate", _always_valid)
        return runtime, fake_client

    return _make


def _analyzer_json():
    return json.dumps(
        {
            "bottleneck_label": "analyzer-specific bottleneck",
            "candidate_regimes": ["exploration", "synthesis"],
            "stage_scores": {
                "exploration": 0.95,
                "synthesis": 0.8,
                "epistemic": 0.1,
                "adversarial": 0.1,
                "operator": 0.3,
                "builder": 0.1,
            },
            "structural_signals": ["needs alternatives"],
            "decision_pressure": 5,
            "evidence_quality": 4,
            "recurrence_potential": 2,
            "confidence": 0.92,
            "rationale": "Need to explore before converging",
            "likely_endpoint_regime": "operator",
            "endpoint_confidence": 0.8,
        }
    )


def _exploration_payload():
    return json.dumps(
        {
            "regime": "exploration",
            "purpose": "exploration purpose",
            "artifact_type": "candidate_frame_set",
            "artifact": {
                "candidate_frames": [
                    "FRAME_ALPHA: orchestration-first framing for switch validation",
                    "FRAME_DELTA: risk-first framing",
                ],
                "selection_criteria": [
                    "CRITERION_BETA: pick the frame that preserves handoff specificity"
                ],
                "unresolved_axes": ["How much confidence is enough before synthesis"],
            },
            "completion_signal": "selection_criteria_ready",
            "failure_signal": "",
            "recommended_next_regime": "synthesis",
        }
    )


def _synthesis_payload():
    return json.dumps(
        {
            "regime": "synthesis",
            "purpose": "synthesis purpose",
            "artifact_type": "thesis_structure",
            "artifact": {
                "central_claim": "The winning frame is FRAME_ALPHA under CRITERION_BETA.",
                "organizing_idea": "Use explicit handoff continuity.",
                "key_tensions": ["speed vs precision"],
                "supporting_structure": ["handoff extracts concrete findings"],
                "pressure_points": ["context can become boilerplate if not constrained"],
            },
            "completion_signal": "thesis_stable",
            "failure_signal": "",
            "recommended_next_regime": "operator",
        }
    )


def _extract_prior_summary(prompt: str) -> str:
    match = re.search(r"Prior artifact summary:\s*(.+)", prompt)
    assert match, "Expected a prior artifact summary line in second-stage prompt"
    return match.group(1).strip()


def _build_multistage_runtime(runtime_factory, monkeypatch):
    runtime, client = runtime_factory([_analyzer_json(), _exploration_payload(), _synthesis_payload()])

    def _force_switch_detection(state, _output):
        return MisroutingDetectionResult(
            current_regime=state.current_regime,
            dominant_failure_mode="mock dominant mode",
            still_productive=False,
            misrouting_detected=True,
            justification="force switch to synthesis",
            recommended_next_regime=runtime.composer.compose(Stage.SYNTHESIS),
        )

    monkeypatch.setattr(runtime.misrouting_detector, "detect", _force_switch_detection)
    return runtime, client


def test_handoff_content_in_second_stage_prompt(runtime_factory, monkeypatch):
    runtime, client = _build_multistage_runtime(runtime_factory, monkeypatch)
    task = "Task raw text that must not leak as current bottleneck"
    captured_prior_handoff = {}

    original_execute_once = runtime.executor.execute_once

    def _capture_execute_once(**kwargs):
        if kwargs.get("regime").stage == Stage.SYNTHESIS:
            captured_prior_handoff["value"] = kwargs.get("prior_handoff")
        return original_execute_once(**kwargs)

    monkeypatch.setattr(runtime.executor, "execute_once", _capture_execute_once)

    _decision, _regime, _result, handoff = runtime.execute(
        task,
        model="fake",
        bounded_orchestration=True,
        max_switches=1,
    )

    second_prompt = client.calls[2]["prompt"]
    prior_summary = _extract_prior_summary(second_prompt)
    synthesis_handoff = captured_prior_handoff["value"]

    assert "## Prior Stage Context" in second_prompt
    assert synthesis_handoff is not None
    assert synthesis_handoff.dominant_frame in second_prompt
    assert prior_summary in second_prompt
    assert len(prior_summary) < 500
    assert "candidate_frames:" not in prior_summary
    assert "selection_criteria:" not in prior_summary
    assert handoff.current_bottleneck != task

    banned_boilerplate = [
        "dominant failure mode",
        "Soft LLM behavior",
        "bottleneck has been classified",
    ]
    for phrase in banned_boilerplate:
        assert phrase not in second_prompt


def test_first_stage_prompt_has_no_handoff(runtime_factory, monkeypatch):
    runtime, client = _build_multistage_runtime(runtime_factory, monkeypatch)

    runtime.execute(
        "Any task text",
        model="fake",
        bounded_orchestration=True,
        max_switches=1,
    )

    first_stage_prompt = client.calls[1]["prompt"]
    assert "## Prior Stage Context" not in first_stage_prompt


def test_handoff_carries_task_specific_content(runtime_factory, monkeypatch):
    runtime, client = _build_multistage_runtime(runtime_factory, monkeypatch)

    runtime.execute(
        "Another task",
        model="fake",
        bounded_orchestration=True,
        max_switches=1,
    )

    second_prompt = client.calls[2]["prompt"]
    assert "FRAME_ALPHA" in second_prompt or "CRITERION_BETA" in second_prompt
