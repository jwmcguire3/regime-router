from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import hashlib

from .analyzer import TaskAnalyzer
from .classifier import TaskClassification, TaskClassifier
from .control import EscalationPolicy, EvolutionEngine, MisroutingDetector, RegimeOutputContract, SwitchOrchestrator
from .models import CANONICAL_FAILURE_IF_OVERUSED, Regime, RegimeExecutionResult, RoutingDecision, RoutingFeatures, Stage, TaskAnalyzerOutput
from .prompts import PromptBuilder
from .routing import RegimeComposer, Router, extract_routing_features, extract_structural_signals, infer_risk_profile
from .state import Handoff, RouterState, router_state_from_jsonable
from .validation import OutputValidator
from .llm import ModelClient, OllamaModelClient

class CognitiveRouterRuntime:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        use_task_analyzer: bool = False,
        task_analyzer_model: str = "llama3",
        use_embedding_router: bool = True,
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        embedding_router = None
        if use_embedding_router:
            try:
                from .embeddings import EmbeddingRouter

                embedding_router = EmbeddingRouter(model_name=embedding_model_name)
            except Exception:
                embedding_router = None
        self.router = Router(embedding_router=embedding_router)
        self.composer = RegimeComposer()
        self.validator = OutputValidator()
        self.prompt_builder = PromptBuilder()
        self.evolver = EvolutionEngine()
        self.misrouting_detector = MisroutingDetector()
        self.escalation_policy = EscalationPolicy()
        self.switch_orchestrator = SwitchOrchestrator()
        self.model_client: ModelClient = OllamaModelClient(base_url=ollama_base_url)
        self.use_task_analyzer = use_task_analyzer
        self.task_analyzer = TaskAnalyzer(self.model_client, model=task_analyzer_model) if use_task_analyzer else None
        self.task_classifier = TaskClassifier(embedding_router=embedding_router)
        self.router_state: Optional[RouterState] = None


    @property
    def ollama(self) -> ModelClient:
        return self.model_client

    @ollama.setter
    def ollama(self, client: ModelClient) -> None:
        self.model_client = client
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
        classification = self.task_classifier.classify(bottleneck)
        if classification.route_type == "direct" and classification.classification_source == "pattern":
            return self._plan_direct(bottleneck, handoff_expected=handoff_expected, classification=classification)

        features = extract_routing_features(bottleneck)
        escalation = self.escalation_policy.evaluate(
            state=self.router_state,
            routing_features=features,
            task_text=bottleneck,
            current_regime=self.router_state.current_regime if self.router_state else None,
            regime_confidence=self.router_state.regime_confidence if self.router_state else None,
            misrouting_result=None,
        )
        signals = task_signals if task_signals is not None else features.structural_signals
        risks = set(risk_profile or set()) if risks_inferred else infer_risk_profile(bottleneck, risk_profile)
        deterministic_decision = self.router.route(
            bottleneck,
            task_signals=signals,
            risk_profile=risks,
            routing_features=features,
            escalation_policy_result=escalation,
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
            escalation_policy_result=escalation,
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
        self.router_state = self._build_router_state(
            bottleneck=bottleneck,
            decision=decision,
            regime=regime,
            signals=signals,
            risks=risks,
            features=features,
        )
        if self.router_state is not None:
            self.router_state.escalation_debug = {
                "direction": escalation.escalation_direction,
                "justification": escalation.justification,
                "biases": {stage.value: v for stage, v in escalation.preferred_regime_biases.items()},
                "switch_pressure_adjustment": escalation.switch_pressure_adjustment,
                "signals": escalation.debug_signals,
            }
            self.router_state.task_classification = {
                "route_type": classification.route_type,
                "confidence": classification.confidence,
                "reason": classification.reason,
            }
        handoff = self._handoff_from_state(self.router_state)
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
        classification = self.task_classifier.classify(task)
        if classification.route_type == "direct" and classification.classification_source == "pattern":
            decision, regime, handoff = self._plan_direct(task, handoff_expected=handoff_expected, classification=classification)
            result = self._execute_direct_task(task=task, model=model, regime=regime)
            self._update_router_state_from_execution(self.router_state, result, reason_entered=decision.why_primary_wins_now)
            handoff = self._handoff_from_state(self.router_state) if self.router_state else handoff
            return decision, regime, result, handoff

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

        if bounded_orchestration and self.router_state is not None:
            result = self._run_orchestration_loop(
                task=task,
                model=model,
                initial_result=result,
                task_signals=task_signals,
                risk_profile=inferred_risks,
                routing_features=routing_features,
                max_switches=max_switches,
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
    ) -> RegimeExecutionResult:
        current_result = initial_result
        while True:
            self.router_state.switches_attempted += 1
            switch_index = self.router_state.switches_attempted
            if self.router_state.switches_executed >= max_switches:
                self.router_state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=self.router_state.current_regime.stage,
                    to_stage=None,
                    switch_recommended=False,
                    switch_executed=False,
                    reason=f"Switch limit reached; max_switches={max_switches}.",
                    switch_trigger=self.router_state.switch_trigger,
                )
                self.router_state.orchestration_stop_reason = "switch_limit_reached"
                break
            output_contract = RegimeOutputContract(
                stage=current_result.stage,
                raw_response=current_result.raw_response,
                validation=current_result.validation,
            )
            detection = self.misrouting_detector.detect(self.router_state, output_contract)
            escalation = self.escalation_policy.evaluate(
                state=self.router_state,
                routing_features=routing_features,
                task_text=task,
                current_regime=self.router_state.current_regime,
                regime_confidence=self.router_state.regime_confidence,
                misrouting_result=detection,
            )
            self.router_state.escalation_debug = {
                "direction": escalation.escalation_direction,
                "justification": escalation.justification,
                "biases": {stage.value: v for stage, v in escalation.preferred_regime_biases.items()},
                "switch_pressure_adjustment": escalation.switch_pressure_adjustment,
                "signals": escalation.debug_signals,
            }
            prior_recommended_next = self.router_state.recommended_next_regime
            orchestrated = self.switch_orchestrator.orchestrate(
                self.router_state,
                output_contract,
                detection,
                switches_used=self.router_state.switches_executed,
                max_switches=max_switches,
                escalation=escalation,
            )
            self.router_state = orchestrated.updated_state
            if not orchestrated.switch_recommended_now or orchestrated.next_regime is None:
                self.router_state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=self.router_state.current_regime.stage,
                    to_stage=None,
                    switch_recommended=False,
                    switch_executed=False,
                    reason=orchestrated.reason_for_switch,
                    switch_trigger=self.router_state.switch_trigger,
                )
                self.router_state.orchestration_stop_reason = "switch_not_recommended"
                break
            if orchestrated.next_regime.stage == self.router_state.current_regime.stage:
                self.router_state.recommended_next_regime = prior_recommended_next
                self.router_state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=self.router_state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage,
                    switch_recommended=True,
                    switch_executed=False,
                    reason="Switch denied to avoid same-stage loop.",
                    switch_trigger=self.router_state.switch_trigger,
                )
                self.router_state.orchestration_stop_reason = "loop_prevented_same_stage"
                break
            if orchestrated.next_regime.stage in self.router_state.executed_regime_stages:
                self.router_state.recommended_next_regime = prior_recommended_next
                self.router_state.record_switch_decision(
                    switch_index=switch_index,
                    from_stage=self.router_state.current_regime.stage,
                    to_stage=orchestrated.next_regime.stage,
                    switch_recommended=True,
                    switch_executed=False,
                    reason="Switch denied to avoid re-entering a previously executed stage.",
                    switch_trigger=self.router_state.switch_trigger,
                )
                self.router_state.orchestration_stop_reason = "loop_prevented_prior_stage"
                break
            self.router_state.record_switch_decision(
                switch_index=switch_index,
                from_stage=self.router_state.current_regime.stage,
                to_stage=orchestrated.next_regime.stage,
                switch_recommended=True,
                switch_executed=True,
                reason=orchestrated.reason_for_switch,
                switch_trigger=self.router_state.switch_trigger,
            )
            self.router_state.switches_executed += 1
            self.router_state.current_regime = orchestrated.next_regime
            current_result = self._execute_regime_once(
                task=task,
                model=model,
                regime=orchestrated.next_regime,
                task_signals=task_signals,
                risk_profile=risk_profile,
            )
            self._update_router_state_from_execution(
                self.router_state,
                current_result,
                reason_entered=orchestrated.reason_for_switch,
            )
        return current_result

    def _plan_direct(
        self,
        task: str,
        *,
        handoff_expected: bool,
        classification: TaskClassification,
    ) -> Tuple[RoutingDecision, Regime, Handoff]:
        decision = RoutingDecision(
            bottleneck=task,
            primary_regime=None,
            runner_up_regime=None,
            why_primary_wins_now="Direct execution — no reasoning bottleneck detected.",
            switch_trigger="Execute immediately; no regime switching needed.",
        )
        regime = self.composer.compose(Stage.OPERATOR, risk_profile=set(), handoff_expected=handoff_expected)
        regime.name = "Direct Passthrough"
        regime.likely_failure_if_overused = "May skip deeper reasoning when hidden ambiguity exists."
        self.router_state = self._build_router_state(
            bottleneck=task,
            decision=decision,
            regime=regime,
            signals=[],
            risks=set(),
            features=extract_routing_features(task),
        )
        if self.router_state is not None:
            self.router_state.task_classification = {
                "route_type": classification.route_type,
                "confidence": classification.confidence,
                "reason": classification.reason,
            }
        handoff = self._handoff_from_state(self.router_state)
        return decision, regime, handoff

    def _execute_direct_task(self, *, task: str, model: str, regime: Regime) -> RegimeExecutionResult:
        system_prompt = "Complete this task directly."
        response = self.model_client.generate(model=model, system=system_prompt, prompt=task, stream=False)
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

    def _execute_regime_once(
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
            repair_mode = self._select_repair_mode(validation)
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

    def _select_repair_mode(self, validation: Dict[str, object]) -> str:
        if not validation.get("valid_json", False):
            return PromptBuilder.REPAIR_MODE_SCHEMA
        if (
            not validation.get("required_keys_present", False)
            or not validation.get("artifact_fields_present", False)
            or not validation.get("artifact_type_matches", False)
            or not validation.get("contract_controls_valid", False)
            or bool(validation.get("missing_keys", []))
            or bool(validation.get("missing_artifact_fields", []))
            or bool(validation.get("control_failures", []))
        ):
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
        bottleneck: str,
        decision: RoutingDecision,
        regime: Regime,
        signals: List[str],
        risks: Set[str],
        features: object,
    ) -> RouterState:
        task_hash = hashlib.sha1(bottleneck.encode("utf-8")).hexdigest()[:12]
        stage_goal = regime.tail_line.text if regime.tail_line else "Produce the minimum useful typed artifact for this regime."
        runner_up_regime = (
            self.composer.compose(decision.runner_up_regime, risk_profile=risks)
            if decision.runner_up_regime
            else None
        )
        primary_name = decision.primary_regime.value if decision.primary_regime else "direct"
        primary_failure = (
            CANONICAL_FAILURE_IF_OVERUSED[decision.primary_regime]
            if decision.primary_regime
            else "Bypassing routing can miss hidden reasoning bottlenecks."
        )
        return RouterState(
            task_id=f"task-{task_hash}",
            task_summary=bottleneck[:180],
            current_bottleneck=bottleneck,
            current_regime=regime,
            runner_up_regime=runner_up_regime,
            regime_confidence=decision.confidence,
            dominant_frame=(
                f"Primary regime is {decision.primary_regime.value}; optimize for its core motion."
                if decision.primary_regime
                else "Direct execution path selected."
            ),
            knowns=[
                f"Bottleneck classified as: {primary_name}",
                f"Runner-up regime: {decision.runner_up_regime.value if decision.runner_up_regime else 'none'}",
            ],
            uncertainties=["Whether the first regime will hit its dominant failure mode quickly."],
            contradictions=["Soft LLM behavior vs hard system control"],
            assumptions=[
                "The bottleneck has been classified correctly.",
                f"Structural signals observed: {', '.join(signals) if signals else 'none'}",
            ],
            risks=sorted(risks) + [primary_failure],
            stage_goal=stage_goal,
            switch_trigger=decision.switch_trigger,
            recommended_next_regime=runner_up_regime,
            decision_pressure=float(getattr(features, "decision_pressure", 0)),
            evidence_quality=float(getattr(features, "evidence_demand", 0)),
            recurrence_potential=float(getattr(features, "recurrence_potential", 0)),
        )

    def _update_router_state_from_execution(
        self,
        state: Optional[RouterState],
        result: RegimeExecutionResult,
        *,
        reason_entered: str,
    ) -> None:
        if state is None:
            return
        parsed = result.validation.get("parsed", {})
        structurally_trustworthy = bool(
            result.validation.get("valid_json", False)
            and result.validation.get("required_keys_present", False)
            and result.validation.get("artifact_fields_present", False)
            and result.validation.get("artifact_type_matches", False)
            and result.validation.get("contract_controls_valid", False)
        )
        completion_signal = ""
        failure_signal = ""
        if isinstance(parsed, dict):
            completion_signal = str(parsed.get("completion_signal", "")).strip()
            failure_signal = str(parsed.get("failure_signal", "")).strip()

            if structurally_trustworthy:
                artifact = parsed.get("artifact", {})
                if isinstance(artifact, dict):
                    central_claim = artifact.get("central_claim")
                    if isinstance(central_claim, str) and central_claim.strip():
                        state.apply_dominant_frame(central_claim.strip())
                recommended_next = parsed.get("recommended_next_regime")
                if isinstance(recommended_next, str):
                    normalized_stage = recommended_next.strip().lower()
                    if normalized_stage in Stage._value2member_map_:
                        state.recommended_next_regime = self._resolve_next_regime(state, Stage(normalized_stage))

        semantic_failures = [str(f) for f in result.validation.get("semantic_failures", [])]
        if semantic_failures:
            state.update_inference_state(
                contradictions=state.contradictions + semantic_failures,
                assumptions=state.assumptions + ["Validation semantic failures were observed and should shape next regime."],
            )

        is_valid = bool(result.validation.get("is_valid", False))
        summary_chunks = ["Execution yielded a valid artifact." if is_valid else "Execution yielded validation failures."]
        if completion_signal:
            summary_chunks.append(f"completion_signal={completion_signal}")
        if failure_signal:
            summary_chunks.append(f"failure_signal={failure_signal}")
        state.record_regime_step(
            regime=state.current_regime,
            reason_entered=reason_entered,
            completion_signal_seen=is_valid,
            failure_signal_seen=not is_valid,
            outcome_summary=" ".join(summary_chunks),
        )

    def _handoff_from_state(self, state: Optional[RouterState]) -> Handoff:
        if state is None:
            return Handoff(
                current_bottleneck="",
                dominant_frame="",
                what_is_known=[],
                what_remains_uncertain=[],
                active_contradictions=[],
                assumptions_in_play=[],
                main_risk_if_continue="",
                recommended_next_regime=None,
                minimum_useful_artifact="",
                recommended_next_regime_full=None,
            )
        return Handoff(
            current_bottleneck=state.current_bottleneck,
            dominant_frame=state.dominant_frame or "",
            what_is_known=state.knowns,
            what_remains_uncertain=state.uncertainties,
            active_contradictions=state.contradictions,
            assumptions_in_play=state.assumptions,
            main_risk_if_continue=state.risks[-1] if state.risks else "",
            recommended_next_regime=state.recommended_next_regime.stage if state.recommended_next_regime else None,
            minimum_useful_artifact=state.stage_goal,
            recommended_next_regime_full=state.recommended_next_regime,
        )

    def _resolve_next_regime(self, state: RouterState, stage: Stage) -> Regime:
        return state.resolve_regime(stage, self.composer.compose)

    def restore_router_state(self, payload: object) -> Optional[RouterState]:
        self.router_state = router_state_from_jsonable(payload, self.composer.compose)
        return self.router_state

# ============================================================
# JSON persistence

CognitiveRuntime = CognitiveRouterRuntime
