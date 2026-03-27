from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import json
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .analyzer import TaskAnalyzer
from .control import EvolutionEngine
from .models import (
    CANONICAL_FAILURE_IF_OVERUSED,
    Regime,
    RegimeExecutionResult,
    RoutingDecision,
    RoutingFeatures,
    Stage,
    TaskAnalyzerOutput,
)
from .prompts import PromptBuilder
from .routing import RegimeComposer, Router, extract_routing_features, extract_structural_signals, infer_risk_profile
from .state import Handoff, RouterState
from .validation import OutputValidator

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.2,
        num_predict: int = 1200,
    ) -> Dict[str, object]:
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        url = f"{self.base_url}/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {e.code}: {detail}") from e
        except URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}. Is it running?") from e

    def list_models(self) -> Dict[str, object]:
        url = f"{self.base_url}/api/tags"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except Exception as e:
            raise RuntimeError(f"Could not list Ollama models from {self.base_url}: {e}") from e

class CognitiveRouterRuntime:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        use_task_analyzer: bool = False,
        task_analyzer_model: str = "llama3",
    ) -> None:
        self.router = Router()
        self.composer = RegimeComposer()
        self.validator = OutputValidator()
        self.prompt_builder = PromptBuilder()
        self.evolver = EvolutionEngine()
        self.ollama = OllamaClient(base_url=ollama_base_url)
        self.use_task_analyzer = use_task_analyzer
        self.task_analyzer = TaskAnalyzer(self.ollama, model=task_analyzer_model) if use_task_analyzer else None
        self.router_state: Optional[RouterState] = None

    def plan(
        self,
        bottleneck: str,
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = True,
        task_signals: Optional[List[str]] = None,
        risks_inferred: bool = False,
    ) -> Tuple[RoutingDecision, Regime, Handoff]:
        features = extract_routing_features(bottleneck)
        signals = task_signals if task_signals is not None else features.structural_signals
        risks = set(risk_profile or set()) if risks_inferred else infer_risk_profile(bottleneck, risk_profile)
        deterministic_decision = self.router.route(
            bottleneck,
            task_signals=signals,
            risk_profile=risks,
            routing_features=features,
        )
        analysis: Optional[TaskAnalyzerOutput] = None
        analyzer_attempted = False
        if (
            self.use_task_analyzer
            and self.task_analyzer
            and self.router.should_use_analyzer(deterministic_decision.confidence, score_gap_threshold=1)
        ):
            analyzer_attempted = True
            analysis = self.task_analyzer.analyze(
                bottleneck,
                routing_features=features,
                task_signals=signals,
                risk_profile=risks,
            )

        decision = self.router.route(
            bottleneck,
            task_signals=signals,
            risk_profile=risks,
            routing_features=features,
            deterministic_stage_scores=deterministic_decision.deterministic_stage_scores,
            deterministic_confidence=deterministic_decision.confidence,
            analyzer_enabled=self.use_task_analyzer,
            analyzer_result=analysis,
            analyzer_gap_threshold=1,
        )
        if analyzer_attempted and analysis is None and decision.analyzer_summary is None:
            analyzer_error = (
                self.task_analyzer.last_error_summary
                if self.task_analyzer and self.task_analyzer.last_error_summary
                else "Analyzer returned invalid/non-JSON output."
            )
            decision.analyzer_summary = f"{analyzer_error} Deterministic routing retained."

        regime = self.composer.compose(decision.primary_regime, risk_profile=risks, handoff_expected=handoff_expected)
        handoff = Handoff(
            current_bottleneck=bottleneck,
            dominant_frame=f"Primary regime is {decision.primary_regime.value}; optimize for its core motion.",
            what_is_known=[
                f"Bottleneck classified as: {decision.primary_regime.value}",
                f"Runner-up regime: {decision.runner_up_regime.value if decision.runner_up_regime else 'none'}",
            ],
            what_remains_uncertain=["Whether the first regime will hit its dominant failure mode quickly."],
            active_contradictions=["Soft LLM behavior vs hard system control"],
            assumptions_in_play=[
                "The bottleneck has been classified correctly.",
                f"Structural signals observed: {', '.join(signals) if signals else 'none'}",
            ],
            main_risk_if_continue=CANONICAL_FAILURE_IF_OVERUSED[decision.primary_regime],
            recommended_next_regime=decision.runner_up_regime,
            minimum_useful_artifact="A typed artifact from the current regime plus a switch trigger.",
        )
        self.router_state = self._build_router_state(
            task_id=f"task-{uuid4().hex[:12]}",
            task_summary=bottleneck,
            bottleneck=bottleneck,
            decision=decision,
            regime=regime,
            handoff=handoff,
            features=features,
            risks=risks,
            prior_regimes=[],
        )
        return decision, regime, handoff

    def execute(self, task: str, model: str, risk_profile: Optional[Set[str]] = None, handoff_expected: bool = True) -> Tuple[RoutingDecision, Regime, RegimeExecutionResult, Handoff]:
        task_signals = extract_structural_signals(task)
        inferred_risks = infer_risk_profile(task, risk_profile)
        decision, regime, handoff = self.plan(
            task,
            risk_profile=inferred_risks,
            handoff_expected=handoff_expected,
            task_signals=task_signals,
            risks_inferred=True,
        )
        system_prompt = self.prompt_builder.build_system_prompt(regime, task_signals=task_signals, risk_profile=inferred_risks)
        user_prompt = self.prompt_builder.build_user_prompt(task, regime, task_signals=task_signals, risk_profile=inferred_risks)

        response = self.ollama.generate(model=model, system=system_prompt, prompt=user_prompt, stream=False)
        raw_text = str(response.get("response", "")).strip()
        validation = self.validator.validate(
            regime.stage,
            raw_text,
            task=task,
            task_signals=task_signals,
            risk_profile=inferred_risks,
        )

        repaired = False
        repair_mode = PromptBuilder.REPAIR_MODE_SEMANTIC
        if not validation.get("is_valid", False):
            repair_mode = self._select_repair_mode(validation)
            repair_prompt = self.prompt_builder.build_repair_prompt(
                task,
                regime,
                raw_text,
                validation,
                task_signals=task_signals,
                repair_mode=repair_mode,
            )
            repair_response = self.ollama.generate(model=model, system=system_prompt, prompt=repair_prompt, stream=False)
            repaired_text = str(repair_response.get("response", "")).strip()
            repaired_validation = self.validator.validate(
                regime.stage,
                repaired_text,
                task=task,
                task_signals=task_signals,
                risk_profile=inferred_risks,
            )

            if repaired_validation.get("is_valid", False):
                raw_text = repaired_text
                validation = repaired_validation
                response = repair_response
                repaired = True

        result = RegimeExecutionResult(
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
        self.router_state = self._build_router_state(
            task_id=self.router_state.task_id if self.router_state else f"task-{uuid4().hex[:12]}",
            task_summary=task,
            bottleneck=task,
            decision=decision,
            regime=regime,
            handoff=handoff,
            features=extract_routing_features(task),
            risks=inferred_risks,
            prior_regimes=[decision.primary_regime],
        )
        return decision, regime, result, handoff

    def _select_repair_mode(self, validation: Dict[str, object]) -> str:
        if not validation.get("valid_json", False):
            return PromptBuilder.REPAIR_MODE_SCHEMA

        semantic_failures = [str(f).lower() for f in validation.get("semantic_failures", [])]
        genericity_markers = (
            "generic filler",
            "forbidden generic domain nouns",
            "ungrounded generic domain terms",
        )
        if any(marker in failure for failure in semantic_failures for marker in genericity_markers):
            return PromptBuilder.REPAIR_MODE_REDUCE_GENERICITY
        return PromptBuilder.REPAIR_MODE_SEMANTIC

    def _build_router_state(
        self,
        *,
        task_id: str,
        task_summary: str,
        bottleneck: str,
        decision: RoutingDecision,
        regime: Regime,
        handoff: Handoff,
        features: RoutingFeatures,
        risks: Set[str],
        prior_regimes: List[Stage],
    ) -> RouterState:
        return RouterState(
            task_id=task_id,
            task_summary=task_summary,
            current_bottleneck=bottleneck,
            current_regime=decision.primary_regime,
            runner_up_regime=decision.runner_up_regime,
            regime_confidence=decision.confidence.level,
            stage_goal=regime.dominant_line.text,
            knowns=list(handoff.what_is_known),
            uncertainties=list(handoff.what_remains_uncertain),
            contradictions=list(handoff.active_contradictions),
            assumptions=list(handoff.assumptions_in_play),
            risks=sorted(risks),
            decision_pressure=features.decision_pressure,
            evidence_quality=features.evidence_demand,
            recurrence_potential=features.recurrence_potential,
            prior_regimes=prior_regimes,
            switch_trigger=decision.switch_trigger,
            recommended_next_regime=handoff.recommended_next_regime,
        )

# ============================================================
# JSON persistence

CognitiveRuntime = CognitiveRouterRuntime
