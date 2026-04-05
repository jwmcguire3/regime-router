from __future__ import annotations

from router.execution.executor import RegimeExecutor
from router.models import LIBRARY, Regime


class FakeModelClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def generate(self, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return {"response": response}


class FakePromptBuilder:
    REPAIR_MODE_SCHEMA = "schema_repair"
    REPAIR_MODE_SEMANTIC = "semantic_repair"
    REPAIR_MODE_REDUCE_GENERICITY = "reduce_genericity"

    def build_system_prompt(self, regime, **kwargs):
        return "system"

    def build_user_prompt(self, task, regime, **kwargs):
        return "user"

    def build_repair_prompt(self, *args, **kwargs):
        return "repair"


class FakeValidator:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def validate(self, *args, **kwargs):
        result = self._results[self.calls]
        self.calls += 1
        return result


def _regime() -> Regime:
    line = LIBRARY["EPI-D1"]
    return Regime(name="epistemic-core", stage=line.stage, dominant_line=line)


def test_no_repair_when_first_pass_valid():
    model_client = FakeModelClient(['{"ok": true}'])
    validator = FakeValidator([{"is_valid": True, "semantic_valid": True, "semantic_failures": [], "control_failures": []}])
    executor = RegimeExecutor(model_client=model_client, prompt_builder=FakePromptBuilder(), validator=validator)

    result = executor.execute_once(task="task", model="fake", regime=_regime(), task_signals=[], risk_profile=set())

    assert result.validation["repair_attempted"] is False
    assert "repair_succeeded" not in result.validation
    assert "repair_mode" not in result.validation


def test_repair_triggers_when_first_pass_invalid():
    model_client = FakeModelClient(['{"bad": true}', '{"fixed": true}'])
    validator = FakeValidator(
        [
            {"is_valid": False, "semantic_valid": False, "semantic_failures": ["needs evidence"]},
            {"is_valid": True, "semantic_valid": True, "semantic_failures": []},
        ]
    )
    executor = RegimeExecutor(model_client=model_client, prompt_builder=FakePromptBuilder(), validator=validator)

    result = executor.execute_once(task="task", model="fake", regime=_regime(), task_signals=[], risk_profile=set())

    assert result.validation["repair_attempted"] is True
    assert "repair_succeeded" in result.validation


def test_repair_attempted_false_means_no_repair_call():
    model_client = FakeModelClient(['{"ok": true}'])
    validator = FakeValidator([{"is_valid": True, "semantic_valid": True, "semantic_failures": [], "control_failures": []}])
    executor = RegimeExecutor(model_client=model_client, prompt_builder=FakePromptBuilder(), validator=validator)

    result = executor.execute_once(task="task", model="fake", regime=_regime(), task_signals=[], risk_profile=set())

    assert result.validation["repair_attempted"] is False
    assert model_client.calls == 1


def test_invalid_output_recovery_empty_output_repair_exhausted():
    model_client = FakeModelClient(["", ""])
    validator = FakeValidator(
        [
            {"is_valid": False, "valid_json": False, "semantic_valid": False, "semantic_failures": ["empty response"]},
            {"is_valid": False, "valid_json": False, "semantic_valid": False, "semantic_failures": ["empty response"]},
        ]
    )
    executor = RegimeExecutor(model_client=model_client, prompt_builder=FakePromptBuilder(), validator=validator)

    result = executor.execute_once(task="task", model="fake", regime=_regime(), task_signals=[], risk_profile=set())

    assert result.raw_response == ""
    assert result.validation["is_valid"] is False
    assert result.validation["repair_attempted"] is True
    assert result.validation["repair_succeeded"] is False
    assert model_client.calls == 2


def test_invalid_output_recovery_structural_invalidity_persists_after_repair():
    model_client = FakeModelClient(['{"artifact":', '{"still":"bad"}'])
    validator = FakeValidator(
        [
            {
                "is_valid": False,
                "valid_json": False,
                "required_keys_present": False,
                "semantic_valid": False,
                "semantic_failures": ["invalid json"],
            },
            {
                "is_valid": False,
                "valid_json": False,
                "required_keys_present": False,
                "semantic_valid": False,
                "semantic_failures": ["invalid json"],
            },
        ]
    )
    executor = RegimeExecutor(model_client=model_client, prompt_builder=FakePromptBuilder(), validator=validator)

    result = executor.execute_once(task="task", model="fake", regime=_regime(), task_signals=[], risk_profile=set())

    assert result.validation["is_valid"] is False
    assert result.validation["repair_attempted"] is True
    assert result.validation["repair_succeeded"] is False
    # Current behavior keeps first-pass output when repair does not validate.
    assert result.raw_response == '{"artifact":'
