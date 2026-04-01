from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from .models import (
    RegimeConfidenceResult,
    RoutingDecision,
    RoutingFeatures,
    Stage,
    TaskAnalyzerOutput,
)

if TYPE_CHECKING:
    from .control import EscalationPolicyResult


def _load_routing_support_module(module_name: str, filename: str):
    spec = spec_from_file_location(module_name, Path(__file__).with_name("routing").joinpath(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from router/routing/{filename}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_feature_extraction_module = _load_routing_support_module("router.routing.feature_extraction", "feature_extraction.py")
_risk_inference_module = _load_routing_support_module("router.routing.risk_inference", "risk_inference.py")
_grammar_rules_module = _load_routing_support_module("router.routing.grammar_rules", "grammar_rules.py")

deduplicate_lines = _grammar_rules_module.deduplicate_lines
has_hard_conflict = _grammar_rules_module.has_hard_conflict
has_soft_conflict = _grammar_rules_module.has_soft_conflict
resolve_conflict = _grammar_rules_module.resolve_conflict
validate_regime_grammar = _grammar_rules_module.validate_regime_grammar


def _contains_any(text: str, phrases: Tuple[str, ...]) -> List[str]:
    return _feature_extraction_module.contains_any(text, phrases)


def _has_phrase(text: str, phrase: str) -> bool:
    return _feature_extraction_module.has_phrase(text, phrase)


def _score_from_matches(*matches: List[str]) -> int:
    return _feature_extraction_module.score_from_matches(*matches)


def extract_routing_features(task: str) -> RoutingFeatures:
    return _feature_extraction_module.extract_routing_features(task)


def extract_structural_signals(task: str) -> List[str]:
    return _feature_extraction_module.extract_structural_signals(task)


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    return _risk_inference_module.infer_risk_profile(task, risk_profile)


# ============================================================
# Core enums


class Router:
    def __init__(self) -> None:
        pass


    def route(
        self,
        bottleneck: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        routing_features: Optional[RoutingFeatures] = None,
        escalation_policy_result: Optional["EscalationPolicyResult"] = None,
        deterministic_stage_scores: Optional[Dict[Stage, int]] = None,
        deterministic_confidence: Optional[RegimeConfidenceResult] = None,
        analyzer_result: Optional[TaskAnalyzerOutput] = None,
        analyzer_enabled: bool = False,
        analyzer_gap_threshold: int = 1,
    ) -> RoutingDecision:
        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=Stage.EXPLORATION,
            runner_up_regime=Stage.SYNTHESIS,
            why_primary_wins_now="Deterministic scoring removed; awaiting LLM proposer.",
            switch_trigger="LLM proposer will determine switch conditions.",
            confidence=RegimeConfidenceResult.low_default(),
            analyzer_enabled=analyzer_enabled,
        )
# ============================================================
# Composer
# ============================================================

_composer_spec = spec_from_file_location(
    "router.routing.composer",
    Path(__file__).with_name("routing").joinpath("composer.py"),
)
if _composer_spec is None or _composer_spec.loader is None:
    raise ImportError("Unable to load RegimeComposer from router/routing/composer.py")
_composer_module = module_from_spec(_composer_spec)
_composer_spec.loader.exec_module(_composer_module)
RegimeComposer = _composer_module.RegimeComposer

_grammar_composer_spec = spec_from_file_location(
    "router.routing.grammar_composer",
    Path(__file__).with_name("routing").joinpath("grammar_composer.py"),
)
if _grammar_composer_spec is None or _grammar_composer_spec.loader is None:
    raise ImportError("Unable to load GrammarComposer from router/routing/grammar_composer.py")
_grammar_composer_module = module_from_spec(_grammar_composer_spec)
_grammar_composer_spec.loader.exec_module(_grammar_composer_module)
GrammarComposer = _grammar_composer_module.GrammarComposer


# ============================================================
# Ollama adapter
# ============================================================
