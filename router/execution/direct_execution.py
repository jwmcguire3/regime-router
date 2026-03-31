from __future__ import annotations

from ..llm import ModelClient
from ..models import Regime, RegimeExecutionResult


def execute_direct_task(*, model_client: ModelClient, task: str, model: str, regime: Regime) -> RegimeExecutionResult:
    system_prompt = "Complete this task directly."
    response = model_client.generate(model=model, system=system_prompt, prompt=task, stream=False)
    raw_text = str(response.get("response", "")).strip()
    return RegimeExecutionResult(
        task=task,
        model=model,
        regime_name=regime.name,
        stage=regime.stage,
        system_prompt=system_prompt,
        user_prompt=task,
        raw_response=raw_text,
        artifact_text=raw_text,
        validation={"is_valid": True, "direct_execution": True},
        ollama_meta={k: v for k, v in response.items() if k != "response"},
    )
