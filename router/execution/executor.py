from __future__ import annotations

from typing import List, Set

from ..llm import ModelClient
from ..models import Regime, RegimeExecutionResult
from ..prompts import PromptBuilder
from ..validation import OutputValidator
from .repair_policy import select_repair_mode


class RegimeExecutor:
    def __init__(self, *, model_client: ModelClient, prompt_builder: PromptBuilder, validator: OutputValidator) -> None:
        self.model_client = model_client
        self.prompt_builder = prompt_builder
        self.validator = validator

    def execute_once(
        self,
        *,
        task: str,
        model: str,
        regime: Regime,
        task_signals: List[str],
        risk_profile: Set[str],
    ) -> RegimeExecutionResult:
        system_prompt = self.prompt_builder.build_system_prompt(regime, task_signals=task_signals, risk_profile=risk_profile)
        user_prompt = self.prompt_builder.build_user_prompt(task, regime, task_signals=task_signals, risk_profile=risk_profile)

        response = self.model_client.generate(model=model, system=system_prompt, prompt=user_prompt, stream=False)
        raw_text = str(response.get("response", "")).strip()
        validation = self.validator.validate(
            regime.stage,
            raw_text,
            task=task,
            task_signals=task_signals,
            risk_profile=risk_profile,
        )

        repaired = False
        repair_mode = PromptBuilder.REPAIR_MODE_SEMANTIC
        if not validation.get("is_valid", False):
            repair_mode = select_repair_mode(validation)
            repair_prompt = self.prompt_builder.build_repair_prompt(
                task,
                regime,
                raw_text,
                validation,
                task_signals=task_signals,
                repair_mode=repair_mode,
            )
            repair_response = self.model_client.generate(model=model, system=system_prompt, prompt=repair_prompt, stream=False)
            repaired_text = str(repair_response.get("response", "")).strip()
            repaired_validation = self.validator.validate(
                regime.stage,
                repaired_text,
                task=task,
                task_signals=task_signals,
                risk_profile=risk_profile,
            )

            if repaired_validation.get("is_valid", False):
                raw_text = repaired_text
                validation = repaired_validation
                response = repair_response
                repaired = True

        return RegimeExecutionResult(
            task=task,
            model=model,
            regime_name=regime.name,
            stage=regime.stage,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_response=raw_text,
            artifact_text=raw_text,
            validation={
                **validation,
                "repair_attempted": True,
                "repair_succeeded": repaired,
                "repair_mode": repair_mode,
            },
            ollama_meta={k: v for k, v in response.items() if k != "response"},
        )
