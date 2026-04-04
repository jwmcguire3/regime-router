from __future__ import annotations

import os
from typing import Dict, List, Optional, Set, Tuple

from ..analyzer import TaskAnalyzer
from ..classifier import TaskClassification, TaskClassifier
from ..control import EscalationPolicy, EvolutionEngine, MisroutingDetector, SwitchOrchestrator
from ..models import Regime, RegimeExecutionResult, RoutingDecision, RoutingFeatures
from ..prompts import PromptBuilder
from ..routing import RegimeComposer, Router, extract_routing_features, extract_structural_signals, infer_risk_profile
from ..settings import DEFAULT_DEEPSEEK_API_KEY_ENV, DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL
from ..state import Handoff, RouterState
from ..validation import OutputValidator
from ..llm import ModelClient, OllamaModelClient, OpenAIModelClient
from ..execution.direct_execution import execute_direct_task
from ..execution.executor import RegimeExecutor
from ..execution.repair_policy import select_repair_mode
from ..orchestration.stop_policy import StopPolicy
from .planner import RuntimePlanner
from .restore import restore_router_state
from .session_runtime import SessionRuntime
from .state_updater import compute_forward_handoff, handoff_from_state, update_router_state_from_execution


def create_model_client(
    *,
    provider: str,
    ollama_base_url: str,
    openai_base_url: str,
    openai_api_key_env: str,
) -> ModelClient:
    if provider == "ollama":
        return OllamaModelClient(base_url=ollama_base_url)

    if provider in {"openai", "deepseek"}:
        api_key = os.getenv(openai_api_key_env, "").strip()
        if not api_key:
            provider_label = "OpenAI" if provider == "openai" else "DeepSeek"
            raise RuntimeError(
                f"{provider_label} provider requires environment variable '{openai_api_key_env}' to be set and non-empty."
            )
        return OpenAIModelClient(api_key=api_key, base_url=openai_base_url)

    raise ValueError(
        f"Unsupported provider '{provider}'. Expected one of: ollama, openai, deepseek."
    )


class CognitiveRouterRuntime:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        # Runtime defaults mirror platform settings defaults for OpenAI-compatible providers.
        provider: str = "deepseek",
        openai_base_url: str = DEFAULT_DEEPSEEK_BASE_URL,
        openai_api_key_env: str = DEFAULT_DEEPSEEK_API_KEY_ENV,
        use_task_analyzer: bool = True,
        task_analyzer_model: str = DEFAULT_DEEPSEEK_MODEL,
    ) -> None:
        self.router = Router()
        self.composer = RegimeComposer()
        self.validator = OutputValidator()
        self.prompt_builder = PromptBuilder()
        self.evolver = EvolutionEngine()
        self.misrouting_detector = MisroutingDetector()
        self.escalation_policy = EscalationPolicy()
        self.switch_orchestrator = SwitchOrchestrator()
        self.stop_policy = StopPolicy()
        self.model_client: ModelClient = create_model_client(
            provider=provider,
            ollama_base_url=ollama_base_url,
            openai_base_url=openai_base_url,
            openai_api_key_env=openai_api_key_env,
        )
        self.use_task_analyzer = use_task_analyzer
        self.task_analyzer = TaskAnalyzer(self.model_client, model=task_analyzer_model)
        self.task_classifier = TaskClassifier()
        self.router_state: Optional[RouterState] = None

        self.planner = RuntimePlanner(
            router=self.router,
            composer=self.composer,
            escalation_policy=self.escalation_policy,
            task_classifier=self.task_classifier,
        )
        self.executor = RegimeExecutor(
            model_client=self.model_client,
            prompt_builder=self.prompt_builder,
            validator=self.validator,
        )
        self.session_runtime = SessionRuntime(
            misrouting_detector=self.misrouting_detector,
            escalation_policy=self.escalation_policy,
            switch_orchestrator=self.switch_orchestrator,
            stop_policy=self.stop_policy,
        )

    @property
    def ollama(self) -> ModelClient:
        return self.model_client

    @ollama.setter
    def ollama(self, client: ModelClient) -> None:
        self.model_client = client
        self.executor.model_client = client
        if self.task_analyzer is not None:
            self.task_analyzer.model_client = client

    def plan(
        self,
        bottleneck: str,
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = True,
        task_signals: Optional[List[str]] = None,
        risks_inferred: bool = False,
    ) -> Tuple[RoutingDecision, Regime, Handoff]:
        analyzer_result = None
        if self.use_task_analyzer and self.task_analyzer is not None:
            features = extract_routing_features(bottleneck)
            signals = task_signals if task_signals is not None else features.structural_signals
            risks = set(risk_profile or set()) if risks_inferred else infer_risk_profile(bottleneck, risk_profile)
            classification = self.task_classifier.classify(bottleneck)
            classifier_signal = {
                "route_type": classification.route_type,
                "confidence": classification.confidence,
                "classification_source": classification.classification_source,
            }
            try:
                analyzer_result = self.task_analyzer.analyze(
                    bottleneck,
                    routing_features=features,
                    task_signals=signals,
                    risk_profile=risks,
                    classifier_signal=classifier_signal,
                )
            except Exception as exc:  # pragma: no cover - defensive runtime fallback
                if hasattr(self.task_analyzer, "last_error_summary"):
                    self.task_analyzer.last_error_summary = f"Analyzer call failed: {exc}"
                analyzer_result = None

        decision, regime, handoff, state, _classification = self.planner.plan(
            bottleneck,
            router_state=self.router_state,
            use_task_analyzer=self.use_task_analyzer,
            task_analyzer=self.task_analyzer,
            risk_profile=risk_profile,
            handoff_expected=handoff_expected,
            task_signals=task_signals,
            risks_inferred=risks_inferred,
            analyzer_result=analyzer_result,
        )
        self.router_state = state
        return decision, regime, handoff

    def execute(
        self,
        task: str,
        model: str,
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = True,
        bounded_orchestration: bool = False,
        max_switches: int = 2,
    ) -> Tuple[RoutingDecision, Regime, RegimeExecutionResult, Handoff]:
        task_signals = extract_structural_signals(task)
        routing_features = extract_routing_features(task)
        inferred_risks = infer_risk_profile(task, risk_profile)
        decision, regime, handoff = self.plan(
            task,
            risk_profile=inferred_risks,
            handoff_expected=handoff_expected,
            task_signals=task_signals,
            risks_inferred=True,
        )
        result = self._execute_regime_once(
            task=task,
            model=model,
            regime=regime,
            task_signals=task_signals,
            risk_profile=inferred_risks,
        )
        if self.router_state is not None:
            self.router_state.orchestration_enabled = bounded_orchestration
            self.router_state.max_switches = max_switches
            self.router_state.switches_attempted = 0
            self.router_state.switches_executed = 0
            self.router_state.orchestration_stop_reason = None
            self.router_state.executed_regime_stages = []
            self.router_state.switch_history = []
        self._update_router_state_from_execution(self.router_state, result, reason_entered=decision.why_primary_wins_now)
        if self.router_state is not None:
            self.router_state.latest_forward_handoff = compute_forward_handoff(result, self.router_state, regime, composer=self.composer)

        if bounded_orchestration and self.router_state is not None:
            result = self._run_orchestration_loop(
                task=task,
                model=model,
                initial_result=result,
                task_signals=task_signals,
                risk_profile=inferred_risks,
                routing_features=routing_features,
                max_switches=max_switches,
                routing_decision=decision,
            )
        elif self.router_state is not None:
            self.router_state.orchestration_stop_reason = "single_step_mode"

        handoff = self._handoff_from_state(self.router_state) if self.router_state else handoff
        return decision, regime, result, handoff

    def list_models(self) -> Dict[str, object]:
        return self.model_client.list_models()

    def _run_orchestration_loop(
        self,
        *,
        task: str,
        model: str,
        initial_result: RegimeExecutionResult,
        task_signals: List[str],
        risk_profile: Set[str],
        routing_features: RoutingFeatures,
        max_switches: int,
        routing_decision: Optional[RoutingDecision],
    ) -> RegimeExecutionResult:
        if self.router_state is None:
            return initial_result
        result = self.session_runtime.run_orchestration_loop(
            state=self.router_state,
            task=task,
            model=model,
            initial_result=initial_result,
            task_signals=task_signals,
            risk_profile=risk_profile,
            routing_features=routing_features,
            max_switches=max_switches,
            routing_decision=routing_decision,
            execute_regime_once=self._execute_regime_once,
            update_state_from_execution=self._update_router_state_from_execution,
            handoff_from_state=self._handoff_from_state,
            compute_forward_handoff=self._compute_forward_handoff,
        )
        return result

    def _plan_direct(
        self,
        task: str,
        *,
        handoff_expected: bool,
        classification: TaskClassification,
    ) -> Tuple[RoutingDecision, Regime, Handoff]:
        decision, regime, handoff, state = self.planner.plan_direct(
            task,
            handoff_expected=handoff_expected,
            classification=classification,
        )
        self.router_state = state
        return decision, regime, handoff

    def _execute_direct_task(self, *, task: str, model: str, regime: Regime) -> RegimeExecutionResult:
        return execute_direct_task(model_client=self.model_client, task=task, model=model, regime=regime)

    def _execute_regime_once(
        self,
        *,
        task: str,
        model: str,
        regime: Regime,
        task_signals: List[str],
        risk_profile: Set[str],
        prior_handoff: Optional[Handoff] = None,
    ) -> RegimeExecutionResult:
        return self.executor.execute_once(
            task=task,
            model=model,
            regime=regime,
            task_signals=task_signals,
            risk_profile=risk_profile,
            prior_handoff=prior_handoff,
        )

    def _select_repair_mode(self, validation: Dict[str, object]) -> str:
        return select_repair_mode(validation)

    def _update_router_state_from_execution(
        self,
        state: Optional[RouterState],
        result: RegimeExecutionResult,
        *,
        reason_entered: str,
    ) -> None:
        update_router_state_from_execution(
            state,
            result,
            reason_entered=reason_entered,
            composer=self.composer,
        )

    def _handoff_from_state(self, state: Optional[RouterState]) -> Handoff:
        return handoff_from_state(state)

    def _compute_forward_handoff(self, result: RegimeExecutionResult, state: RouterState, regime: Regime) -> Handoff:
        return compute_forward_handoff(result, state, regime, composer=self.composer)

    def restore_router_state(self, payload: object) -> Optional[RouterState]:
        self.router_state = restore_router_state(payload, composer=self.composer)
        return self.router_state


CognitiveRuntime = CognitiveRouterRuntime
