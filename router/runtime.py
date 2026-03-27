from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import json
import hashlib
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .analyzer import TaskAnalyzer
from .control import EvolutionEngine
from .models import CANONICAL_FAILURE_IF_OVERUSED, Regime, RegimeExecutionResult, RoutingDecision, Stage, TaskAnalyzerOutput
from .prompts import PromptBuilder
from .routing import RegimeComposer, Router, extract_routing_features, extract_structural_signals, infer_risk_profile
from .state import Handoff, RouterState, router_state_from_jsonable
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
        self.router_state = self._build_router_state(
            bottleneck=bottleneck,
            decision=decision,
            regime=regime,
            signals=signals,
            risks=risks,
            features=features,
        )
        handoff = self._handoff_from_state(self.router_state)
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
        self._update_router_state_from_execution(self.router_state, decision, result)
        handoff = self._handoff_from_state(self.router_state) if self.router_state else handoff
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
        return RouterState(
            task_id=f"task-{task_hash}",
            task_summary=bottleneck[:180],
            current_bottleneck=bottleneck,
            current_regime=regime,
            runner_up_regime=runner_up_regime,
            regime_confidence=decision.confidence,
            dominant_frame=f"Primary regime is {decision.primary_regime.value}; optimize for its core motion.",
            knowns=[
                f"Bottleneck classified as: {decision.primary_regime.value}",
                f"Runner-up regime: {decision.runner_up_regime.value if decision.runner_up_regime else 'none'}",
            ],
            uncertainties=["Whether the first regime will hit its dominant failure mode quickly."],
            contradictions=["Soft LLM behavior vs hard system control"],
            assumptions=[
                "The bottleneck has been classified correctly.",
                f"Structural signals observed: {', '.join(signals) if signals else 'none'}",
            ],
            risks=sorted(risks) + [CANONICAL_FAILURE_IF_OVERUSED[decision.primary_regime]],
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
        decision: RoutingDecision,
        result: RegimeExecutionResult,
    ) -> None:
        if state is None:
            return
        parsed = result.validation.get("parsed", {})
        completion_signal = ""
        failure_signal = ""
        if isinstance(parsed, dict):
            completion_signal = str(parsed.get("completion_signal", "")).strip()
            failure_signal = str(parsed.get("failure_signal", "")).strip()

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
            reason_entered=decision.why_primary_wins_now,
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
        if state.current_regime.stage == stage:
            return state.current_regime
        if state.runner_up_regime and state.runner_up_regime.stage == stage:
            return state.runner_up_regime
        if state.recommended_next_regime and state.recommended_next_regime.stage == stage:
            return state.recommended_next_regime
        return self.composer.compose(stage)

    def restore_router_state(self, payload: object) -> Optional[RouterState]:
        self.router_state = router_state_from_jsonable(payload, self.composer.compose)
        return self.router_state

# ============================================================
# JSON persistence

CognitiveRuntime = CognitiveRouterRuntime
