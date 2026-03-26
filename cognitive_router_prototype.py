from __future__ import annotations

import argparse
import json
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED = "expansion_when_defined"
STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL = "concrete_versions_feel_too_small"
STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED = "fragments_understood_spine_missed"


@dataclass(frozen=True)
class RoutingFeatures:
    structural_signals: List[str]
    decision_pressure: int
    evidence_demand: int
    fragility_pressure: int
    recurrence_potential: int
    possibility_space_need: int
    detected_markers: Dict[str, List[str]]


def _contains_any(text: str, phrases: Tuple[str, ...]) -> List[str]:
    return [phrase for phrase in phrases if phrase in text]


def _score_from_matches(*matches: List[str]) -> int:
    return min(10, sum(len(group) for group in matches))


def extract_routing_features(task: str) -> RoutingFeatures:
    text = task.lower()

    # Grouped deterministic pattern families, optimized for task-shape markers.
    expansion_words = ("expand", "expands", "expansion", "broadens", "gets bigger", "widens", "balloons")
    define_words = ("define", "defined", "definition", "specify", "specified", "scope", "frame")
    concrete_words = ("concrete", "specific", "instance", "version", "example", "implementation")
    too_small_words = ("too small", "small", "narrow", "shrinks", "feels tiny", "cramped", "thin slice")
    parts_words = ("fragment", "fragments", "pieces", "parts", "components")
    whole_words = ("whole", "spine", "core", "throughline", "center", "backbone", "organizing logic")
    missing_words = ("missed", "missing", "lost", "not seen", "not grasped", "not holding")
    understood_words = ("understood", "clear", "comprehensible", "makes sense", "legible")

    evidence_words = ("evidence", "support", "verify", "unknown", "unclear", "unresolved", "proof", "confidence")
    uncertainty_words = ("uncertain", "ambigu", "not sure", "missing information", "what is missing")

    decision_words = ("decide", "decision", "choose", "commit", "next move", "time pressure", "ship now", "now")
    tradeoff_words = ("tradeoff", "trade-off", "between options", "selection", "opportunity cost")

    fragility_words = ("fragile", "break", "stress test", "failure mode", "risk", "destabil", "brittle")
    launch_words = ("launch", "production", "deploy", "deployment", "go-live", "trust", "customer-facing")

    recurrence_words = ("repeatable", "reusable", "template", "playbook", "pattern", "systemat", "standardize")
    builder_words = ("productize", "modules", "interfaces", "workflow", "automation")

    possibility_words = ("possibility", "explore", "exploration", "brainstorm", "alternatives", "option space", "open")
    convergence_words = ("too early", "premature", "locked in", "single frame", "compresses", "narrowing")

    matches: Dict[str, List[str]] = {}

    expansion_hits = _contains_any(text, expansion_words)
    define_hits = _contains_any(text, define_words)
    concrete_hits = _contains_any(text, concrete_words)
    too_small_hits = _contains_any(text, too_small_words)
    parts_hits = _contains_any(text, parts_words)
    whole_hits = _contains_any(text, whole_words)
    missing_hits = _contains_any(text, missing_words)
    understood_hits = _contains_any(text, understood_words)
    evidence_hits = _contains_any(text, evidence_words)
    uncertainty_hits = _contains_any(text, uncertainty_words)
    decision_hits = _contains_any(text, decision_words)
    tradeoff_hits = _contains_any(text, tradeoff_words)
    fragility_hits = _contains_any(text, fragility_words)
    launch_hits = _contains_any(text, launch_words)
    recurrence_hits = _contains_any(text, recurrence_words)
    builder_hits = _contains_any(text, builder_words)
    possibility_hits = _contains_any(text, possibility_words)
    convergence_hits = _contains_any(text, convergence_words)

    structural_signals: List[str] = []

    # expansion-when-defined
    if expansion_hits and define_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED)
        matches["expansion_when_defined"] = sorted(set(expansion_hits + define_hits))

    # concrete-form-too-small / abstraction overflow
    if concrete_hits and too_small_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL)
        matches["concrete_form_too_small"] = sorted(set(concrete_hits + too_small_hits))

    # parts/whole mismatch (legacy-compatible signal name retained)
    if parts_hits and whole_hits and missing_hits and understood_hits:
        structural_signals.append(STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED)
        matches["parts_whole_mismatch"] = sorted(set(parts_hits + whole_hits + missing_hits + understood_hits))

    if parts_hits and whole_hits and missing_hits:
        matches.setdefault("parts_whole_mismatch", sorted(set(parts_hits + whole_hits + missing_hits)))

    if evidence_hits or uncertainty_hits:
        matches["uncertainty_evidence_demand"] = sorted(set(evidence_hits + uncertainty_hits))
    if decision_hits or tradeoff_hits:
        matches["decision_tradeoff_commitment"] = sorted(set(decision_hits + tradeoff_hits))
    if fragility_hits or launch_hits:
        matches["fragility_launch_trust"] = sorted(set(fragility_hits + launch_hits))
    if recurrence_hits or builder_hits:
        matches["recurrence_systemization"] = sorted(set(recurrence_hits + builder_hits))
    if possibility_hits or convergence_hits:
        matches["open_possibility_space"] = sorted(set(possibility_hits + convergence_hits))

    return RoutingFeatures(
        structural_signals=structural_signals,
        decision_pressure=_score_from_matches(decision_hits, tradeoff_hits),
        evidence_demand=_score_from_matches(evidence_hits, uncertainty_hits),
        fragility_pressure=_score_from_matches(fragility_hits, launch_hits),
        recurrence_potential=_score_from_matches(recurrence_hits, builder_hits),
        possibility_space_need=_score_from_matches(possibility_hits, convergence_hits),
        detected_markers=matches,
    )


def extract_structural_signals(task: str) -> List[str]:
    return extract_routing_features(task).structural_signals


def infer_risk_profile(task: str, risk_profile: Optional[Set[str]]) -> Set[str]:
    inferred = set(risk_profile or set())
    text = task.lower()
    features = extract_routing_features(task)
    signals = set(features.structural_signals)

    if signals:
        inferred.add("abstract_structural_task")
    if (
        STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in signals
        and any(
        k in text for k in ("single frame", "one frame", "unif", "compress", "organizing idea")
        )
    ):
        inferred.add("false_unification")
    if features.fragility_pressure >= 2:
        inferred.add("fragility_pressure")
    if features.evidence_demand >= 2:
        inferred.add("evidence_gap")
    if features.decision_pressure >= 2:
        inferred.add("decision_urgency")

    return inferred


# ============================================================
# Core enums
# ============================================================

class Stage(str, Enum):
    EXPLORATION = "exploration"
    SYNTHESIS = "synthesis"
    EPISTEMIC = "epistemic"
    ADVERSARIAL = "adversarial"
    OPERATOR = "operator"
    BUILDER = "builder"


class FunctionType(str, Enum):
    DOMINANCE = "dominance"
    SUPPRESSION = "suppression"
    SHAPE = "shape"
    GATE = "gate"
    TRANSFER = "transfer"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ============================================================
# Data models
# ============================================================

@dataclass(frozen=True)
class LinePrimitive:
    id: str
    text: str
    stage: Stage
    function: FunctionType
    attractor: str
    suppresses: Tuple[str, ...] = ()
    tension: str = ""
    risks: Tuple[str, ...] = ()
    compatible_with: Tuple[str, ...] = ()
    incompatible_with: Tuple[str, ...] = ()


@dataclass
class Regime:
    name: str
    stage: Stage
    dominant_line: LinePrimitive
    suppression_lines: List[LinePrimitive] = field(default_factory=list)
    shape_lines: List[LinePrimitive] = field(default_factory=list)
    tail_line: Optional[LinePrimitive] = None
    rejected_lines: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    likely_failure_if_overused: str = ""

    @property
    def all_lines(self) -> List[LinePrimitive]:
        lines = [self.dominant_line]
        lines.extend(self.suppression_lines)
        lines.extend(self.shape_lines)
        if self.tail_line:
            lines.append(self.tail_line)
        return lines

    def instruction_block(self) -> str:
        return "\n".join(f"- {line.text}" for line in self.all_lines)

    def render(self) -> str:
        return (
            f"Regime: {self.name}\n"
            f"Stage: {self.stage.value}\n"
            f"Likely failure if overused: {self.likely_failure_if_overused or 'n/a'}\n\n"
            f"Instructions:\n{self.instruction_block()}"
        )


@dataclass
class RoutingDecision:
    bottleneck: str
    primary_regime: Stage
    runner_up_regime: Optional[Stage]
    why_primary_wins_now: str
    switch_trigger: str


@dataclass
class Handoff:
    current_bottleneck: str
    dominant_frame: str
    what_is_known: List[str]
    what_remains_uncertain: List[str]
    active_contradictions: List[str]
    assumptions_in_play: List[str]
    main_risk_if_continue: str
    recommended_next_regime: Optional[Stage]
    minimum_useful_artifact: str


@dataclass
class FailureLog:
    regime_name: str
    observed_failure: str
    severity: Severity
    recurrence_count: int
    likely_trigger: str
    implicated_instruction_ids: List[str] = field(default_factory=list)
    missing_instruction: Optional[str] = None


@dataclass
class RevisionProposal:
    regime_name: str
    revision_type: str
    target_failure: str
    old_instruction: Optional[str]
    new_instruction: Optional[str]
    expected_increase: List[str]
    expected_decrease: List[str]
    likely_side_effect: List[str]
    validation_test: str
    adoption_recommendation: str


@dataclass
class RegimeExecutionResult:
    task: str
    model: str
    regime_name: str
    stage: Stage
    system_prompt: str
    user_prompt: str
    raw_response: str
    artifact_text: str
    validation: Dict[str, object]
    ollama_meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class SessionRecord:
    timestamp_utc: str
    task: str
    risk_profile: List[str]
    model: str
    routing: Dict[str, object]
    regime: Dict[str, object]
    result: Dict[str, object]
    handoff: Dict[str, object]


# ============================================================
# Primitive library
# ============================================================

LIBRARY: Dict[str, LinePrimitive] = {
    # Exploration
    "EXP-D1": LinePrimitive(
        id="EXP-D1",
        text="Generate multiple viable frames before selecting one.",
        stage=Stage.EXPLORATION,
        function=FunctionType.DOMINANCE,
        attractor="branch_generation",
        suppresses=("premature_narrowing",),
        tension="breadth_over_closure",
        risks=("sprawl",),
        compatible_with=("EXP-S1", "EXP-S2", "EXP-P1", "EXP-T1"),
        incompatible_with=("SYN-D1", "OPR-D1", "EPI-D2"),
    ),
    "EXP-S1": LinePrimitive(
        id="EXP-S1",
        text="Explore directions that are structurally distinct, not cosmetically different.",
        stage=Stage.EXPLORATION,
        function=FunctionType.SHAPE,
        attractor="meaningful_divergence",
        suppresses=("fake_variety",),
        tension="true_alternatives_over_superficial_options",
        risks=("misses_useful_variants",),
        compatible_with=("EXP-D1", "EXP-P1", "EXP-T1"),
    ),
    "EXP-P1": LinePrimitive(
        id="EXP-P1",
        text="Do not continue branching once a dominant path clearly improves the decision space.",
        stage=Stage.EXPLORATION,
        function=FunctionType.SUPPRESSION,
        attractor="anti_sprawl",
        suppresses=("sprawl", "endless_branching"),
        tension="selection_over_expansion",
        risks=("early_convergence",),
        compatible_with=("EXP-D1", "EXP-S1", "EXP-S2", "EXP-T1"),
    ),
    "EXP-S2": LinePrimitive(
        id="EXP-S2",
        text="Prefer branches that change the problem framing, not just the answer.",
        stage=Stage.EXPLORATION,
        function=FunctionType.SHAPE,
        attractor="reframing",
        suppresses=("answer_only_ideation",),
        tension="frame_discovery_over_solution_listing",
        risks=("underdeveloped_practical_options",),
        compatible_with=("EXP-D1", "EXP-P1", "EXP-T1"),
    ),
    "EXP-T1": LinePrimitive(
        id="EXP-T1",
        text="End by identifying the selection criterion that should determine the next regime.",
        stage=Stage.EXPLORATION,
        function=FunctionType.TRANSFER,
        attractor="handoff_readiness",
        suppresses=("open_ended_exploration",),
        tension="transition_over_drift",
        risks=("rigid_criterion_choice",),
        compatible_with=("EXP-D1", "EXP-P1", "EXP-S1", "EXP-S2"),
    ),
    # Synthesis
    "SYN-D1": LinePrimitive(
        id="SYN-D1",
        text="Follow the strongest live possibility before justifying it.",
        stage=Stage.SYNTHESIS,
        function=FunctionType.DOMINANCE,
        attractor="early_frame_selection",
        suppresses=("over_hesitation",),
        tension="commitment_over_delay",
        risks=("premature_lock",),
        compatible_with=("SYN-D2", "SYN-S1", "SYN-P1", "SYN-P2"),
        incompatible_with=("EXP-D1", "EPI-D2"),
    ),
    "SYN-D2": LinePrimitive(
        id="SYN-D2",
        text="Compress toward a vivid organizing idea, even when provisional.",
        stage=Stage.SYNTHESIS,
        function=FunctionType.DOMINANCE,
        attractor="conceptual_compression",
        suppresses=("descriptive_sprawl",),
        tension="center_over_coverage",
        risks=("false_unification",),
        compatible_with=("SYN-D1", "SYN-S1", "SYN-P1", "SYN-P2"),
        incompatible_with=("EPI-G1",),
    ),
    "SYN-S1": LinePrimitive(
        id="SYN-S1",
        text="Prefer generative connections over cautious completeness.",
        stage=Stage.SYNTHESIS,
        function=FunctionType.SHAPE,
        attractor="connection_making",
        suppresses=("inventory_style_output",),
        tension="synthesis_over_completeness",
        risks=("weakly_supported_linkage",),
        compatible_with=("SYN-D1", "SYN-D2", "SYN-P1", "SYN-P2"),
        incompatible_with=("EPI-D2",),
    ),
    "SYN-P1": LinePrimitive(
        id="SYN-P1",
        text="Do not flatten decisive counterevidence just to preserve coherence.",
        stage=Stage.SYNTHESIS,
        function=FunctionType.SUPPRESSION,
        attractor="anti_self_sealing",
        suppresses=("coherence_over_truth",),
        tension="disconfirmation_over_elegance",
        risks=("fragmented_reporting",),
        compatible_with=("SYN-D1", "SYN-D2", "SYN-S1", "SYN-P2"),
    ),
    "SYN-P2": LinePrimitive(
        id="SYN-P2",
        text="If evidence directly weakens the central frame, revise the frame before integrating surrounding material.",
        stage=Stage.SYNTHESIS,
        function=FunctionType.SUPPRESSION,
        attractor="frame_revision_discipline",
        suppresses=("aestheticized_contradiction",),
        tension="revision_over_preservation",
        risks=("synthesis_hesitation",),
        compatible_with=("SYN-D1", "SYN-D2", "SYN-S1", "SYN-P1"),
    ),
    # Epistemic
    "EPI-D1": LinePrimitive(
        id="EPI-D1",
        text="Do not let fluency outrun evidence.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.DOMINANCE,
        attractor="evidence_discipline",
        suppresses=("polished_overreach",),
        tension="support_over_flow",
        risks=("flat_output",),
        compatible_with=("EPI-D2", "EPI-P1", "EPI-P2", "EPI-S1", "EPI-G1"),
    ),
    "EPI-D2": LinePrimitive(
        id="EPI-D2",
        text="Prefer repeated observations over coherent explanations.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.DOMINANCE,
        attractor="anti_overinference",
        suppresses=("elegant_theory_drift",),
        tension="observation_over_explanation",
        risks=("weak_gestalt",),
        compatible_with=("EPI-D1", "EPI-P1", "EPI-P2", "EPI-S1", "EPI-G1"),
        incompatible_with=("SYN-D1", "SYN-S1", "EXP-D1"),
    ),
    "EPI-P1": LinePrimitive(
        id="EPI-P1",
        text="Match claim strength to evidence strength.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.SUPPRESSION,
        attractor="claim_size_control",
        suppresses=("plausibility_inflation",),
        tension="proportionality_over_assertion",
        risks=("excessive_caution",),
        compatible_with=("EPI-D1", "EPI-D2", "EPI-P2", "EPI-S1", "EPI-G1"),
    ),
    "EPI-P2": LinePrimitive(
        id="EPI-P2",
        text="Preserve contradictions unless stronger evidence resolves them.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.SUPPRESSION,
        attractor="contradiction_preservation",
        suppresses=("forced_coherence",),
        tension="tension_over_cleanup",
        risks=("decision_drag",),
        compatible_with=("EPI-D1", "EPI-D2", "EPI-P1", "EPI-S1", "EPI-G1", "ADV-D1"),
    ),
    "EPI-G1": LinePrimitive(
        id="EPI-G1",
        text="It is better to omit than to over-infer.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.GATE,
        attractor="omission_discipline",
        suppresses=("filler_inference",),
        tension="omission_over_speculation",
        risks=("under_answering",),
        compatible_with=("EPI-D1", "EPI-D2", "EPI-P1", "EPI-P2", "EPI-S1"),
        incompatible_with=("SYN-D2",),
    ),
    "EPI-S1": LinePrimitive(
        id="EPI-S1",
        text="When multiple valid observations exist, prioritize the one that most changes the user's next decision.",
        stage=Stage.EPISTEMIC,
        function=FunctionType.SHAPE,
        attractor="decision_relevant_rigor",
        suppresses=("sterile_report_completeness",),
        tension="action_relevance_over_neutral_reporting",
        risks=("under_reporting_context",),
        compatible_with=("EPI-D1", "EPI-D2", "EPI-P1", "EPI-P2", "EPI-G1"),
    ),
    # Adversarial
    "ADV-D1": LinePrimitive(
        id="ADV-D1",
        text="Find the objection or failure case that most changes the decision if true.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.DOMINANCE,
        attractor="decision_relevant_destabilization",
        suppresses=("diffuse_critique",),
        tension="highest_impact_critique_over_broad_critique",
        risks=("tunnel_vision",),
        compatible_with=("ADV-P1", "ADV-P2", "ADV-S1", "ADV-S2", "ADV-T1", "EPI-P2"),
    ),
    "ADV-P1": LinePrimitive(
        id="ADV-P1",
        text="Attack the strongest version of the frame, not a weaker substitute.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.SUPPRESSION,
        attractor="anti_strawman",
        suppresses=("cheap_critique",),
        tension="fair_target_over_easy_target",
        risks=("over_time_strengthening_target",),
        compatible_with=("ADV-D1", "ADV-P2", "ADV-S1", "ADV-S2", "ADV-T1"),
    ),
    "ADV-P2": LinePrimitive(
        id="ADV-P2",
        text="Do not generate weak objections for completeness.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.SUPPRESSION,
        attractor="anti_critique_sludge",
        suppresses=("low_value_skepticism",),
        tension="relevance_over_coverage",
        risks=("over_pruning",),
        compatible_with=("ADV-D1", "ADV-P1", "ADV-S1", "ADV-S2", "ADV-T1"),
    ),
    "ADV-S1": LinePrimitive(
        id="ADV-S1",
        text="Prefer strong destabilizers over broad coverage.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.SHAPE,
        attractor="concentrated_pressure",
        suppresses=("objection_spam",),
        tension="force_over_completeness",
        risks=("misses_secondary_interacting_risks",),
        compatible_with=("ADV-D1", "ADV-P1", "ADV-P2", "ADV-S2", "ADV-T1"),
    ),
    "ADV-S2": LinePrimitive(
        id="ADV-S2",
        text="Prefer concrete break conditions over abstract doubt.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.SHAPE,
        attractor="operational_critique",
        suppresses=("vague_skepticism",),
        tension="concrete_failure_over_generalized_doubt",
        risks=("misses_conceptual_fragility",),
        compatible_with=("ADV-D1", "ADV-P1", "ADV-P2", "ADV-S1", "ADV-T1"),
    ),
    "ADV-T1": LinePrimitive(
        id="ADV-T1",
        text="End with the best revision that preserves what still survives.",
        stage=Stage.ADVERSARIAL,
        function=FunctionType.TRANSFER,
        attractor="revision_pressure",
        suppresses=("nihilistic_teardown",),
        tension="repair_over_destruction",
        risks=("premature_salvage",),
        compatible_with=("ADV-D1", "ADV-P1", "ADV-P2", "ADV-S1", "ADV-S2"),
    ),
    # Operator
    "OPR-D1": LinePrimitive(
        id="OPR-D1",
        text="If a decision is required, produce one.",
        stage=Stage.OPERATOR,
        function=FunctionType.DOMINANCE,
        attractor="closure_under_constraint",
        suppresses=("analysis_paralysis",),
        tension="action_over_delay",
        risks=("forced_closure",),
        compatible_with=("OPR-P1", "OPR-S1", "OPR-S2", "OPR-G1"),
        incompatible_with=("EXP-D1",),
    ),
    "OPR-S1": LinePrimitive(
        id="OPR-S1",
        text="Select the highest expected value option under current constraints.",
        stage=Stage.OPERATOR,
        function=FunctionType.SHAPE,
        attractor="constrained_optimization",
        suppresses=("indecision_by_perfectionism",),
        tension="expected_value_over_ideal_completeness",
        risks=("underweights_tail_risk",),
        compatible_with=("OPR-D1", "OPR-P1", "OPR-S2", "OPR-G1"),
    ),
    "OPR-P1": LinePrimitive(
        id="OPR-P1",
        text="Do not continue analysis unless it would change the decision.",
        stage=Stage.OPERATOR,
        function=FunctionType.SUPPRESSION,
        attractor="anti_delay",
        suppresses=("recursive_analysis",),
        tension="sufficiency_over_more_thinking",
        risks=("truncates_needed_validation",),
        compatible_with=("OPR-D1", "OPR-S1", "OPR-S2", "OPR-G1"),
    ),
    "OPR-S2": LinePrimitive(
        id="OPR-S2",
        text="Name the tradeoff being accepted.",
        stage=Stage.OPERATOR,
        function=FunctionType.SHAPE,
        attractor="explicit_sacrifice",
        suppresses=("hidden_cost_decisions",),
        tension="conscious_loss_over_implicit_loss",
        risks=("verbose_decision_ritual",),
        compatible_with=("OPR-D1", "OPR-P1", "OPR-S1", "OPR-G1"),
    ),
    "OPR-G1": LinePrimitive(
        id="OPR-G1",
        text="If two options are close, choose the one that preserves future flexibility.",
        stage=Stage.OPERATOR,
        function=FunctionType.GATE,
        attractor="optionality",
        suppresses=("irreversible_premature_bets",),
        tension="reversibility_over_narrow_optimization",
        risks=("chronic_deferral",),
        compatible_with=("OPR-D1", "OPR-P1", "OPR-S1", "OPR-S2"),
    ),
    # Builder
    "BLD-D1": LinePrimitive(
        id="BLD-D1",
        text="Translate the solution into a reusable structure.",
        stage=Stage.BUILDER,
        function=FunctionType.DOMINANCE,
        attractor="abstraction_into_infrastructure",
        suppresses=("one_off_thinking",),
        tension="reuse_over_singularity",
        risks=("premature_productization",),
        compatible_with=("BLD-P1", "BLD-S1", "BLD-S2", "BLD-T1"),
    ),
    "BLD-S1": LinePrimitive(
        id="BLD-S1",
        text="Identify modules, interfaces, repeated inputs, repeated outputs, and implementation order.",
        stage=Stage.BUILDER,
        function=FunctionType.SHAPE,
        attractor="systems_decomposition",
        suppresses=("vague_architecture",),
        tension="explicit_structure_over_intuition",
        risks=("overformalization",),
        compatible_with=("BLD-D1", "BLD-P1", "BLD-S2", "BLD-T1"),
    ),
    "BLD-P1": LinePrimitive(
        id="BLD-P1",
        text="Do not design beyond proven recurrence.",
        stage=Stage.BUILDER,
        function=FunctionType.SUPPRESSION,
        attractor="anti_overbuild",
        suppresses=("speculative_complexity",),
        tension="recurrence_over_imagined_scale",
        risks=("underbuilding",),
        compatible_with=("BLD-D1", "BLD-S1", "BLD-S2", "BLD-T1"),
    ),
    "BLD-S2": LinePrimitive(
        id="BLD-S2",
        text="Start with the simplest structure that supports repeated use.",
        stage=Stage.BUILDER,
        function=FunctionType.SHAPE,
        attractor="minimal_durable_architecture",
        suppresses=("gold_plating",),
        tension="sufficiency_over_elegance",
        risks=("weak_extensibility",),
        compatible_with=("BLD-D1", "BLD-P1", "BLD-S1", "BLD-T1"),
    ),
    "BLD-T1": LinePrimitive(
        id="BLD-T1",
        text="Convert the current insight into a template, playbook, or product candidate.",
        stage=Stage.BUILDER,
        function=FunctionType.TRANSFER,
        attractor="compounding_artifact_generation",
        suppresses=("non_reusable_output",),
        tension="asset_creation_over_ephemeral_advice",
        risks=("premature_packaging",),
        compatible_with=("BLD-D1", "BLD-P1", "BLD-S1", "BLD-S2"),
    ),
}


CANONICAL_DOMINANTS: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: ["EXP-D1"],
    Stage.SYNTHESIS: ["SYN-D1", "SYN-D2"],
    Stage.EPISTEMIC: ["EPI-D1", "EPI-D2"],
    Stage.ADVERSARIAL: ["ADV-D1"],
    Stage.OPERATOR: ["OPR-D1"],
    Stage.BUILDER: ["BLD-D1"],
}

SUPPRESSION_BY_STAGE: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: ["EXP-P1"],
    Stage.SYNTHESIS: ["SYN-P1", "SYN-P2"],
    Stage.EPISTEMIC: ["EPI-P1", "EPI-P2"],
    Stage.ADVERSARIAL: ["ADV-P1", "ADV-P2"],
    Stage.OPERATOR: ["OPR-P1"],
    Stage.BUILDER: ["BLD-P1"],
}

SHAPES_BY_STAGE: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: ["EXP-S1", "EXP-S2"],
    Stage.SYNTHESIS: ["SYN-S1"],
    Stage.EPISTEMIC: ["EPI-S1"],
    Stage.ADVERSARIAL: ["ADV-S1", "ADV-S2"],
    Stage.OPERATOR: ["OPR-S1", "OPR-S2"],
    Stage.BUILDER: ["BLD-S1", "BLD-S2"],
}

TAILS_BY_STAGE: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: ["EXP-T1"],
    Stage.SYNTHESIS: [],
    Stage.EPISTEMIC: ["EPI-G1"],
    Stage.ADVERSARIAL: ["ADV-T1"],
    Stage.OPERATOR: ["OPR-G1"],
    Stage.BUILDER: ["BLD-T1"],
}

CANONICAL_FAILURE_IF_OVERUSED: Dict[Stage, str] = {
    Stage.EXPLORATION: "branch sprawl",
    Stage.SYNTHESIS: "false unification",
    Stage.EPISTEMIC: "under-synthesis / decision drag",
    Stage.ADVERSARIAL: "nihilistic or repetitive critique",
    Stage.OPERATOR: "forced closure",
    Stage.BUILDER: "over-engineering",
}

ARTIFACT_HINTS: Dict[Stage, str] = {
    Stage.EXPLORATION: "candidate_frame_set",
    Stage.SYNTHESIS: "dominant_frame",
    Stage.EPISTEMIC: "evidence_map",
    Stage.ADVERSARIAL: "stress_report",
    Stage.OPERATOR: "decision_packet",
    Stage.BUILDER: "system_blueprint",
}

ARTIFACT_FIELDS: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: [
        "candidate_frames",
        "selection_criteria",
        "unresolved_axes",
    ],
    Stage.SYNTHESIS: [
        "central_claim",
        "organizing_idea",
        "key_tensions",
        "supporting_structure",
        "pressure_points",
    ],
    Stage.EPISTEMIC: [
        "supported_claims",
        "plausible_but_unproven",
        "contradictions",
        "omitted_due_to_insufficient_support",
        "decision_relevant_conclusions",
    ],
    Stage.ADVERSARIAL: [
        "top_destabilizers",
        "hidden_assumptions",
        "break_conditions",
        "survivable_revisions",
        "residual_risks",
    ],
    Stage.OPERATOR: [
        "decision",
        "rationale",
        "tradeoff_accepted",
        "next_actions",
        "fallback_trigger",
        "review_point",
    ],
    Stage.BUILDER: [
        "reusable_pattern",
        "modules",
        "interfaces",
        "required_inputs",
        "produced_outputs",
        "implementation_sequence",
        "compounding_path",
    ],
}


# ============================================================
# Router
# ============================================================

class Router:
    def route(
        self,
        bottleneck: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        routing_features: Optional[RoutingFeatures] = None,
    ) -> RoutingDecision:
        b = bottleneck.lower().strip()
        features = routing_features or extract_routing_features(bottleneck)
        signals = set(task_signals or features.structural_signals)
        risks = set(risk_profile or set())

        interpretation_shortcut_markers = ["strongest interpretation", "strongest frame", "what this actually is"]
        epistemic_markers = ["evidence", "support", "verify", "unknown", "unclear", "unresolved"]

        if any(k in b for k in interpretation_shortcut_markers) and not any(k in b for k in epistemic_markers):
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.SYNTHESIS,
                runner_up_regime=Stage.ADVERSARIAL,
                why_primary_wins_now="The task asks for interpretation-level compression first, then pressure-testing against break conditions.",
                switch_trigger="Switch when the strongest frame is identified and the next bottleneck becomes exposing how it fails under stress.",
            )

        if any(k in b for k in ["stress test this frame", "stress test", "break it", "too clean", "fragile", "launch"]):
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.ADVERSARIAL,
                runner_up_regime=Stage.EPISTEMIC,
                why_primary_wins_now="The bottleneck is hidden fragility, not idea generation.",
                switch_trigger="Switch when the top destabilizer is identified and revisions are clear.",
            )

        if any(k in b for k in ["repeatable", "template", "playbook", "system", "productize", "reusable"]):
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.BUILDER,
                runner_up_regime=Stage.OPERATOR,
                why_primary_wins_now="The pattern should be turned into durable structure.",
                switch_trigger="Switch when modules, recurrence, and implementation order are clear.",
            )

        stage_scores: Dict[Stage, int] = {stage: 0 for stage in Stage}

        def add_phrase_weights(stage: Stage, weighted_terms: Dict[str, int]) -> None:
            for phrase, weight in weighted_terms.items():
                if phrase in b:
                    stage_scores[stage] += weight

        add_phrase_weights(
            Stage.OPERATOR,
            {
                "decide": 4,
                "decision": 4,
                "choose": 4,
                "commit": 3,
                "next move": 4,
                "tradeoff": 4,
                "select between options": 5,
                "choose between": 5,
                "time pressure": 3,
            },
        )
        add_phrase_weights(
            Stage.EPISTEMIC,
            {
                "unknown": 4,
                "unclear": 4,
                "unresolved": 4,
                "what is missing": 5,
                "what do we not know": 5,
                "support": 3,
                "evidence": 4,
                "verify": 4,
                "rigor": 3,
                "are you sure": 4,
            },
        )
        add_phrase_weights(
            Stage.SYNTHESIS,
            {
                "strongest interpretation": 10,
                "strongest frame": 10,
                "what this actually is": 10,
                "many signals": 4,
                "no center": 4,
                "parts are legible": 5,
                "whole organizing logic is missing": 6,
                "fragments but no spine": 6,
                "fragments are understood": 5,
                "spine is still missing": 6,
                "hidden spine": 4,
                "what this really is": 4,
            },
        )
        if any(k in b for k in interpretation_shortcut_markers):
            stage_scores[Stage.SYNTHESIS] += 4
            if any(k in b for k in epistemic_markers):
                stage_scores[Stage.EPISTEMIC] += 2

        add_phrase_weights(
            Stage.EXPLORATION,
            {
                "possibility": 3,
                "brainstorm": 4,
                "explore": 2,
                "alternatives": 3,
                "option space": 3,
                "open possibilities": 4,
            },
        )

        # "options" alone is intentionally weak to avoid swallowing decision tasks.
        if "options" in b:
            stage_scores[Stage.EXPLORATION] += 1

        if any(
            phrase in b
            for phrase in [
                "parts are legible",
                "whole organizing logic is missing",
                "fragments but no spine",
                "fragments are understood",
                "spine is still missing",
                "many signals but no center",
            ]
        ):
            stage_scores[Stage.SYNTHESIS] += 4

        if STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED in signals:
            stage_scores[Stage.SYNTHESIS] += 5
        if STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL in signals:
            stage_scores[Stage.SYNTHESIS] += 2
        if STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED in signals:
            stage_scores[Stage.SYNTHESIS] += 2
        if "parts_whole_mismatch" in features.detected_markers:
            stage_scores[Stage.SYNTHESIS] += 3
        if "abstract_structural_task" in risks:
            stage_scores[Stage.SYNTHESIS] += 2
        if "false_unification" in risks:
            stage_scores[Stage.SYNTHESIS] += 2
        if features.evidence_demand >= 2:
            stage_scores[Stage.EPISTEMIC] += 3
        if features.decision_pressure >= 2:
            stage_scores[Stage.OPERATOR] += 3
        if features.fragility_pressure >= 2:
            stage_scores[Stage.ADVERSARIAL] += 3
        if features.recurrence_potential >= 2:
            stage_scores[Stage.BUILDER] += 3
        if features.possibility_space_need >= 2:
            stage_scores[Stage.EXPLORATION] += 2

        precedence_order = [
            Stage.OPERATOR,
            Stage.EPISTEMIC,
            Stage.ADVERSARIAL,
            Stage.SYNTHESIS,
            Stage.BUILDER,
            Stage.EXPLORATION,
        ]
        ranked = sorted(stage_scores.items(), key=lambda x: (-x[1], precedence_order.index(x[0])))
        top_stage, top_score = ranked[0]

        if top_score <= 0:
            return RoutingDecision(
                bottleneck=bottleneck,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="No specific regime has enough signal; exploration is the safest fallback.",
                switch_trigger="Switch when one frame becomes more decision-relevant than the others.",
            )

        runner_up = next((stage for stage, score in ranked[1:] if score > 0), Stage.EXPLORATION if top_stage != Stage.EXPLORATION else Stage.SYNTHESIS)

        reasons = {
            Stage.OPERATOR: (
                "Decision-intent language dominates; the immediate need is commitment and explicit tradeoffs.",
                "Switch when the decision, tradeoff, and fallback trigger are explicit.",
            ),
            Stage.EPISTEMIC: (
                "Uncertainty/evidence language dominates; the current risk is claims outrunning support.",
                "Switch when supported vs unsupported claims are separated and the next decision becomes clear.",
            ),
            Stage.SYNTHESIS: (
                "Structural-compression signals dominate; the work needs an organizing spine before action.",
                "Switch when a dominant frame emerges that can guide exclusion or action.",
            ),
            Stage.EXPLORATION: (
                "The bottleneck indicates missing possibility space more than proof or commitment.",
                "Switch when 2-5 distinct frames exist and one begins to dominate.",
            ),
            Stage.BUILDER: (
                "The pattern should be turned into durable structure.",
                "Switch when modules, recurrence, and implementation order are clear.",
            ),
            Stage.ADVERSARIAL: (
                "The bottleneck is hidden fragility, not idea generation.",
                "Switch when the top destabilizer is identified and revisions are clear.",
            ),
        }
        why_primary_wins_now, switch_trigger = reasons[top_stage]

        return RoutingDecision(
            bottleneck=bottleneck,
            primary_regime=top_stage,
            runner_up_regime=runner_up,
            why_primary_wins_now=why_primary_wins_now,
            switch_trigger=switch_trigger,
        )
# ============================================================
# Composer
# ============================================================

class RegimeComposer:
    def compose(self, stage: Stage, risk_profile: Optional[Set[str]] = None, handoff_expected: bool = False) -> Regime:
        risk_profile = risk_profile or set()
        dominant = self._choose_dominant(stage, risk_profile)
        suppression = self._choose_suppressions(stage, risk_profile)
        shapes = self._choose_shapes(stage, risk_profile)
        tail = self._choose_tail(stage, handoff_expected, risk_profile)

        candidate_lines = [dominant] + suppression + shapes + ([tail] if tail else [])
        final_lines, rejected, reasons = self._resolve_conflicts(candidate_lines)
        final_lines = self._dedupe_and_trim(final_lines, max_lines=5)

        dom = next(l for l in final_lines if l.function == FunctionType.DOMINANCE)
        sup = [l for l in final_lines if l.function == FunctionType.SUPPRESSION]
        shp = [l for l in final_lines if l.function == FunctionType.SHAPE]
        tail_final = next((l for l in final_lines if l.function in (FunctionType.GATE, FunctionType.TRANSFER)), None)

        return Regime(
            name=f"{stage.value.title()} Core",
            stage=stage,
            dominant_line=dom,
            suppression_lines=sup,
            shape_lines=shp,
            tail_line=tail_final,
            rejected_lines=rejected,
            rejection_reasons=reasons,
            likely_failure_if_overused=CANONICAL_FAILURE_IF_OVERUSED[stage],
        )

    def _choose_dominant(self, stage: Stage, risk_profile: Set[str]) -> LinePrimitive:
        if stage == Stage.SYNTHESIS:
            return LIBRARY["SYN-D2"] if "sprawl" in risk_profile else LIBRARY["SYN-D1"]
        if stage == Stage.EPISTEMIC:
            return LIBRARY["EPI-D2"] if "elegant_theory_drift" in risk_profile else LIBRARY["EPI-D1"]
        return LIBRARY[CANONICAL_DOMINANTS[stage][0]]

    def _choose_suppressions(self, stage: Stage, risk_profile: Set[str]) -> List[LinePrimitive]:
        if stage == Stage.SYNTHESIS:
            chosen = ["SYN-P1"]
            if self._requires_synthesis_break_condition_pressure(risk_profile):
                chosen.append("SYN-P2")
        elif stage == Stage.EPISTEMIC:
            chosen = ["EPI-P1", "EPI-P2"]
        elif stage == Stage.ADVERSARIAL:
            chosen = ["ADV-P1", "ADV-P2"]
        elif stage == Stage.OPERATOR:
            chosen = ["OPR-P1"]
        elif stage == Stage.EXPLORATION:
            chosen = ["EXP-P1"]
        else:
            chosen = ["BLD-P1"]
        return [LIBRARY[i] for i in chosen]

    def _choose_shapes(self, stage: Stage, risk_profile: Set[str]) -> List[LinePrimitive]:
        if stage == Stage.EXPLORATION:
            chosen = ["EXP-S1"]
            if "need_reframing" in risk_profile:
                chosen.append("EXP-S2")
        elif stage == Stage.SYNTHESIS:
            chosen = [] if "high_stakes" in risk_profile else ["SYN-S1"]
        elif stage == Stage.EPISTEMIC:
            chosen = ["EPI-S1"]
        elif stage == Stage.ADVERSARIAL:
            chosen = ["ADV-S2"]
            if "single_point_failure" not in risk_profile:
                chosen.append("ADV-S1")
        elif stage == Stage.OPERATOR:
            chosen = ["OPR-S1", "OPR-S2"]
        else:
            chosen = ["BLD-S1", "BLD-S2"]
        return [LIBRARY[i] for i in chosen]

    @staticmethod
    def _requires_synthesis_break_condition_pressure(risk_profile: Set[str]) -> bool:
        high_risk_synthesis_conditions = {
            "coherence_over_truth",
            "false_unification",
            "high_stakes",
            "abstract_structural_task",
        }
        return bool(high_risk_synthesis_conditions & risk_profile)

    def _choose_tail(self, stage: Stage, handoff_expected: bool, risk_profile: Set[str]) -> Optional[LinePrimitive]:
        if stage == Stage.EXPLORATION and handoff_expected:
            return LIBRARY["EXP-T1"]
        if stage == Stage.EPISTEMIC and "strict" in risk_profile:
            return LIBRARY["EPI-G1"]
        if stage == Stage.ADVERSARIAL and handoff_expected:
            return LIBRARY["ADV-T1"]
        if stage == Stage.OPERATOR and "optionality" in risk_profile:
            return LIBRARY["OPR-G1"]
        if stage == Stage.BUILDER and handoff_expected:
            return LIBRARY["BLD-T1"]
        return None

    def _resolve_conflicts(self, lines: List[LinePrimitive]) -> Tuple[List[LinePrimitive], List[str], List[str]]:
        kept: List[LinePrimitive] = []
        rejected: List[str] = []
        reasons: List[str] = []

        for line in lines:
            conflict = False
            for existing in kept:
                if line.id in existing.incompatible_with or existing.id in line.incompatible_with:
                    rejected.append(line.id)
                    reasons.append(f"Rejected {line.id} because it conflicts with {existing.id}.")
                    conflict = True
                    break
            if not conflict:
                kept.append(line)

        dominance_lines = [l for l in kept if l.function == FunctionType.DOMINANCE]
        if len(dominance_lines) > 2:
            for extra in dominance_lines[2:]:
                kept = [l for l in kept if l.id != extra.id]
                rejected.append(extra.id)
                reasons.append(f"Rejected {extra.id} because more than two dominance lines weakens regime asymmetry.")

        if len(dominance_lines) == 2:
            a, b = dominance_lines[0], dominance_lines[1]
            if a.id in b.incompatible_with or b.id in a.incompatible_with:
                kept = [l for l in kept if l.id != b.id]
                rejected.append(b.id)
                reasons.append(f"Rejected {b.id} because it creates opposing motion with {a.id}.")

        return kept, rejected, reasons

    def _dedupe_and_trim(self, lines: List[LinePrimitive], max_lines: int = 5) -> List[LinePrimitive]:
        seen: Set[str] = set()
        deduped: List[LinePrimitive] = []
        for line in lines:
            if line.id not in seen:
                deduped.append(line)
                seen.add(line.id)

        priority = {
            FunctionType.DOMINANCE: 0,
            FunctionType.SUPPRESSION: 1,
            FunctionType.SHAPE: 2,
            FunctionType.GATE: 3,
            FunctionType.TRANSFER: 3,
        }
        deduped.sort(key=lambda x: priority[x.function])
        return deduped[:max_lines]


# ============================================================
# Ollama adapter
# ============================================================

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


# ============================================================
# Prompt builder + validator
# ============================================================

class PromptBuilder:
    REPAIR_MODE_SCHEMA = "schema_repair"
    REPAIR_MODE_SEMANTIC = "semantic_repair"
    REPAIR_MODE_REDUCE_GENERICITY = "reduce_genericity_repair"

    @staticmethod
    def build_system_prompt(regime: Regime, task_signals: Optional[List[str]] = None, risk_profile: Optional[Set[str]] = None) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_list = "\n".join(f"- {f}" for f in fields)
        task_signals = task_signals or []

        field_rules = PromptBuilder._field_rules(regime.stage)
        signal_lines = (
            "\n".join(f"- {sig.replace('_', ' ')}" for sig in task_signals)
            if task_signals
            else "- no explicit structural signals detected"
        )
        risk_line = ", ".join(sorted(risk_profile or set())) or "none"
        synthesis_constraints = ""
        if regime.stage == Stage.SYNTHESIS:
            synthesis_constraints = textwrap.dedent(
                """
                Synthesis anchoring constraints:
                - Treat the extracted structural signals above as required anchors.
                - Every synthesis field must reinterpret, connect, or test those anchors.
                - If a sentence still works after removing task-specific signals, it is too generic and must be rewritten.
                - pressure_points must falsify or weaken the interpretation, not describe implementation difficulty.
                """
            ).strip()

        return textwrap.dedent(
            f"""
            You are executing exactly one reasoning regime.

            Active regime: {regime.name}
            Stage: {regime.stage.value}

            Follow these instructions exactly:
            {regime.instruction_block()}

            Output requirements:
            - Stay inside the active regime.
            - Do not switch regimes on your own.
            - Return only valid JSON.
            - Do not wrap the JSON in markdown fences.
            - Keep prose concise, concrete, and specific to the input.
            - Do not restate the task label in multiple fields.
            - Each artifact field must contribute non-overlapping information.
            - If the input is too thin to support a strong artifact, say so concretely inside the relevant fields rather than padding with abstractions.

            Top-level JSON keys must be:
            - regime
            - stage
            - artifact_type
            - artifact

            artifact_type must be exactly: {artifact_name}

            artifact must include these keys:
            {field_list}

            Field rules for this stage:
            {field_rules}
            
            Do not introduce external domains, industries, or generic project types.

            All frames must be constructed directly from the structural signals in the input:
            {signal_lines}

            If a frame could apply to any generic project, it is invalid.

            Frames must reinterpret these signals, not replace them.
            Frames must describe what the project *is structurally*, not what it *does*.
            {synthesis_constraints}
            Active risk profile: {risk_line}

            Avoid phrases like:
            - "technology"
            - "solution"
            - "effort"
            - "project to build"

            Prefer:
            - "the project is a system for..."
            - "the project behaves like..."
            - "the project is an attempt to resolve..."
            """
        ).strip()

    @staticmethod
    def build_user_prompt(task: str, regime: Regime, task_signals: Optional[List[str]] = None, risk_profile: Optional[Set[str]] = None) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        signals = ", ".join(task_signals or []) or "none"
        risks = ", ".join(sorted(risk_profile or set())) or "none"
        return textwrap.dedent(
            f"""
            Task:
            {task}

            Structural signals:
            {signals}

            Risk profile:
            {risks}

          Return one JSON object with exactly these top-level keys:
          regime, stage, artifact_type, artifact

            Use only the information in the task.
            Do not invent missing specifics.
            If the task is abstract or underspecified, reflect that explicitly instead of using generic filler.
            """
        ).strip()

    @staticmethod
    def _field_rules(stage: Stage) -> str:
        rules = {
            Stage.SYNTHESIS: """
            - central_claim: one-sentence structural interpretation that explicitly uses at least two task signals (directly or close transforms)
            - organizing_idea: explain the mechanism that generates central_claim from the signals; do not restate central_claim
            - key_tensions: 2-4 pressure pairs, each rooted in specific task signals
            - supporting_structure: 2-4 signal-anchored observations tied to exact extracted signals, not broad project language
            - pressure_points: frame-break conditions that would falsify or materially weaken the interpretation; not execution risks
            """,
            Stage.EPISTEMIC: """
            - supported_claims: claims directly supported by the input
            - plausible_but_unproven: ideas that fit the input but are not established
            - contradictions: tensions or mismatches that remain unresolved
            - omitted_due_to_insufficient_support: things intentionally not claimed
            - decision_relevant_conclusions: conclusions that most affect the next decision
            """,
            Stage.ADVERSARIAL: """
            - top_destabilizers: the most decision-changing objections
            - hidden_assumptions: assumptions the frame depends on
            - break_conditions: specific conditions that would cause failure
            - survivable_revisions: best fixes that preserve what still works
            - residual_risks: risks that remain even after revision
            """,
            Stage.OPERATOR: """
            - decision: one clear choice
            - rationale: why this choice wins now under current constraints
            - tradeoff_accepted: what is being knowingly sacrificed
            - next_actions: immediate executable steps
            - fallback_trigger: what condition should cause reconsideration
            - review_point: when or under what condition to review the decision
            """,
            Stage.EXPLORATION: """
            - candidate_frames: 2-5 structurally distinct framings
            - selection_criteria: what should determine which frame wins
            - unresolved_axes: what key unknowns still separate the frames
            """,
            Stage.BUILDER: """
            - reusable_pattern: the core repeatable pattern
            - modules: main components
            - interfaces: how the modules connect
            - required_inputs: what the system needs
            - produced_outputs: what it creates
            - implementation_sequence: build order
            - compounding_path: how this becomes more reusable over time
            """,
            
        }
        return textwrap.dedent(rules.get(stage, "")).strip()
        
    @staticmethod
    def build_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
        repair_mode: str = REPAIR_MODE_SEMANTIC,
    ) -> str:
        if repair_mode == PromptBuilder.REPAIR_MODE_SCHEMA:
            return PromptBuilder._build_schema_repair_prompt(task, regime, invalid_output, validation)
        if repair_mode == PromptBuilder.REPAIR_MODE_REDUCE_GENERICITY:
            return PromptBuilder._build_reduce_genericity_repair_prompt(
                task,
                regime,
                invalid_output,
                validation,
                task_signals=task_signals,
            )
        return PromptBuilder._build_semantic_repair_prompt(
            task,
            regime,
            invalid_output,
            validation,
            task_signals=task_signals,
        )

    @staticmethod
    def _build_schema_repair_prompt(task: str, regime: Regime, invalid_output: str, validation: Dict[str, object]) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        required_fields = ", ".join(ARTIFACT_FIELDS[regime.stage])
        missing_keys = validation.get("missing_keys", [])
        missing_fields = validation.get("missing_artifact_fields", [])
        parse_error = validation.get("error", "n/a")

        return textwrap.dedent(
            f"""
            Your previous output failed structural/schema validation.

            Original task:
            {task}

            Required top-level keys: regime, stage, artifact_type, artifact
            artifact_type must be exactly: {artifact_name}
            Required artifact fields: {required_fields}

            Schema failures:
            - parse_or_structure_error: {parse_error}
            - missing_keys: {missing_keys}
            - missing_artifact_fields: {missing_fields}

            Previous invalid output:
            {invalid_output}

            Repair rules:
            - Return only valid JSON.
            - Keep existing content where possible; perform minimal structural edits.
            - Do not add commentary or markdown fences.
            """
        ).strip()

    @staticmethod
    def _build_semantic_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
    ) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_text = ", ".join(fields)
        extracted_signals = task_signals or extract_structural_signals(task)
        signal_text = ", ".join(extracted_signals) if extracted_signals else "none"

        failures = validation.get("semantic_failures", [])
        failure_text = "\n".join(f"- {f}" for f in failures) if failures else "- output failed validation"
        failed_fields = PromptBuilder._failed_fields(failures, fields)
        failed_field_text = ", ".join(failed_fields) if failed_fields else "unknown (repair only where failures apply)"

        return textwrap.dedent(
            f"""
            Your previous output failed semantic validation.

            Original task:
            {task}

            Extracted structural signals:
            {signal_text}

            Required artifact:
            {artifact_name}

            Required fields:
            {field_text}

            Validation failures:
            {failure_text}

            Fields requiring rewrite:
            {failed_field_text}

            Previous invalid output:
            {invalid_output}

            Repair instructions:
            - Return only valid JSON with the same top-level schema.
            - Make minimal edits.
            - Rewrite only fields that failed validation; keep non-failed fields unchanged.
            - Keep central_claim and organizing_idea distinct (no paraphrase duplication).
            - Treat pressure_points as frame-break conditions that can falsify or materially weaken the frame.
            - Do not introduce external domains, industries, teams, stakeholders, or generic project nouns.
            - Do not explain your edits.
            - Do not include markdown fences.
            """
        ).strip()

    @staticmethod
    def _build_reduce_genericity_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
    ) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_text = ", ".join(fields)
        extracted_signals = task_signals or extract_structural_signals(task)
        signal_text = ", ".join(extracted_signals) if extracted_signals else "none"
        failures = validation.get("semantic_failures", [])
        failure_text = "\n".join(f"- {f}" for f in failures) if failures else "- output is too generic"
        failed_fields = PromptBuilder._failed_fields(failures, fields)
        failed_field_text = ", ".join(failed_fields) if failed_fields else "unknown (repair only where failures apply)"

        return textwrap.dedent(
            f"""
            Your previous output is structurally valid but too generic.

            Original task:
            {task}

            Extracted structural signals:
            {signal_text}

            Required artifact:
            {artifact_name}

            Required fields:
            {field_text}

            Genericity failures:
            {failure_text}

            Fields requiring rewrite:
            {failed_field_text}

            Previous invalid output:
            {invalid_output}

            Reduce-genericity repair instructions:
            - Return only valid JSON with the same top-level schema.
            - Make minimal edits and rewrite only failed fields.
            - Replace generic filler with signal-anchored statements grounded in the original task.
            - Do not introduce any external domain nouns (technology, industry, solution, team, stakeholders, innovation, etc.).
            - Keep central_claim and organizing_idea distinct.
            - pressure_points must be frame-break tests, not execution concerns.
            - Do not explain your edits or use markdown fences.
            """
        ).strip()

    @staticmethod
    def _failed_fields(failures: List[str], fields: List[str]) -> List[str]:
        detected: List[str] = []
        for failure in failures:
            for field in fields:
                if failure.startswith(f"{field} ") and field not in detected:
                    detected.append(field)
        return detected

class OutputValidator:
    GENERIC_PHRASES = {
        "exploring",
        "assessing",
        "understanding",
        "navigating",
        "complexity",
        "various factors",
        "multiple perspectives",
        "deeper analysis",
        "careful consideration",
        "systemic issues",
        "abstract dynamics",
    }

    FORBIDDEN_GENERIC = {
        "technology",
        "stakeholders",
        "innovation",
        "solution",
        "industry",
        "team",
    }

    GENERIC_PRESSURE_WORDS = {
        "execute",
        "execution",
        "implement",
        "implementation",
        "roadmap",
        "timeline",
        "deliverable",
        "milestone",
        "resourcing",
        "coordination",
        "alignment",
        "rollout",
    }

    REQUIRED_SIGNAL_WORDS = {
        "expand",
        "define",
        "small",
        "spine",
        "fragment",
    }

    def validate(
        self,
        stage: Stage,
        raw_response: str,
        task: str = "",
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {
            "valid_json": False,
            "required_keys_present": False,
            "artifact_fields_present": False,
            "missing_keys": [],
            "missing_artifact_fields": [],
            "artifact_type_matches": False,
            "stage_matches": False,
            "semantic_valid": False,
            "semantic_failures": [],
            "parsed": None,
        }

        try:
            parsed = json.loads(raw_response)
            result["valid_json"] = True
            result["parsed"] = parsed
        except json.JSONDecodeError as e:
            result["error"] = f"JSON decode error: {e}"
            return result

        required = {"regime", "stage", "artifact_type", "artifact"}
        parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
        missing_keys = sorted(required - parsed_keys)
        result["missing_keys"] = missing_keys
        result["required_keys_present"] = len(missing_keys) == 0

        if not result["required_keys_present"]:
            return result

        artifact = parsed.get("artifact", {})
        if not isinstance(artifact, dict):
            result["error"] = "artifact must be a JSON object"
            return result

        required_artifact_fields = set(ARTIFACT_FIELDS[stage])
        artifact_keys = set(artifact.keys())
        missing_artifact_fields = sorted(required_artifact_fields - artifact_keys)
        result["missing_artifact_fields"] = missing_artifact_fields
        result["artifact_fields_present"] = len(missing_artifact_fields) == 0
        result["artifact_type_matches"] = parsed.get("artifact_type") == ARTIFACT_HINTS[stage]
        result["stage_matches"] = parsed.get("stage") == stage.value

        structural_valid = bool(
            result["valid_json"]
            and result["required_keys_present"]
            and result["artifact_fields_present"]
            and result["artifact_type_matches"]
            and result["stage_matches"]
        )

        if not structural_valid:
            result["is_valid"] = False
            return result

        semantic_failures = self._semantic_checks(
            stage,
            artifact,
            task,
            task_signals=task_signals,
            risk_profile=risk_profile,
        )
        result["semantic_failures"] = semantic_failures
        result["semantic_valid"] = len(semantic_failures) == 0
        result["is_valid"] = structural_valid and result["semantic_valid"]
        return result

    def _semantic_checks(
        self,
        stage: Stage,
        artifact: Dict[str, object],
        task: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
    ) -> List[str]:
        failures: List[str] = []

        flat_fields = {}
        for k, v in artifact.items():
            if isinstance(v, list):
                flat_fields[k] = " ".join(str(x) for x in v).strip()
            else:
                flat_fields[k] = str(v).strip()

        values = {k: self._normalize(v) for k, v in flat_fields.items() if v}

        # 1. Empty or ultra-short content
        for k, v in flat_fields.items():
            if len(v.split()) < 3:
                failures.append(f"{k} is too short to be meaningful")

        # 2. Repetition across fields
        keys = list(values.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if values[a] and values[a] == values[b]:
                    failures.append(f"{a} duplicates {b}")
                elif self._jaccard(values[a], values[b]) > 0.75:
                    failures.append(f"{a} is too similar to {b}")

        # 3. Generic filler detection
        for k, v in values.items():
            hits = [p for p in self.GENERIC_PHRASES if p in v]
            if hits:
                failures.append(f"{k} contains generic filler: {', '.join(hits[:3])}")

        # 4. Task grounding check
        task_tokens = self._tokenize(task)
        if task_tokens:
            grounded = False
            for v in values.values():
                overlap = task_tokens & self._tokenize(v)
                if len(overlap) >= 2:
                    grounded = True
                    break
            if not grounded:
                failures.append("artifact is not grounded in the task specifics")

        # 5. Task-specific grounding checks for abstract framing tasks
        all_text = " ".join(flat_fields.values()).lower()
        artifact_tokens = self._tokenize(all_text)
        task_text = task.lower()

        extracted_signals = set(extract_structural_signals(task))
        active_signals = extracted_signals | set(task_signals or [])
        if active_signals:
            forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
            if forbidden_hits:
                failures.append(
                    f"artifact introduces forbidden generic domain nouns: {', '.join(forbidden_hits)}"
                )

            token_expectations = {
                STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED: ["expand", "define"],
                STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL: ["concrete", "small"],
                STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED: ["fragment", "spine"],
            }
            matched = 0
            for signal in active_signals:
                tokens = token_expectations.get(signal, [])
                if all(tok in all_text for tok in tokens):
                    matched += 1
            if matched < 1:
                failures.append(
                    "artifact is not grounded in the task's core structural signals"
                )

        # 6. Stage-specific checks
        if stage == Stage.SYNTHESIS:
            if "central_claim" in values and "organizing_idea" in values:
                if self._jaccard(values["central_claim"], values["organizing_idea"]) > 0.65:
                    failures.append("organizing_idea restates central_claim instead of explaining it")

            if "supporting_structure" in values:
                token_count = len(self._tokenize(values["supporting_structure"]))
                if token_count < 6:
                    failures.append("supporting_structure is too thin")

            if active_signals:
                if "central_claim" in flat_fields and self._signal_overlap_count(flat_fields["central_claim"], list(active_signals)) < 1:
                    failures.append(
                        "central_claim is not anchored to the task's structural signals"
                    )
                if "organizing_idea" in flat_fields and self._signal_overlap_count(flat_fields["organizing_idea"], list(active_signals)) < 1:
                    failures.append(
                        "organizing_idea is not anchored to the task's structural signals"
                    )
                if "key_tensions" in flat_fields and self._signal_overlap_count(flat_fields["key_tensions"], list(active_signals)) < 1:
                    failures.append(
                        "key_tensions are not tied to the task's structural signals"
                    )

                if "pressure_points" in flat_fields:
                    pressure_text = flat_fields["pressure_points"].lower()
                    generic_pressure_hits = sorted(
                        word for word in self.GENERIC_PRESSURE_WORDS if word in pressure_text
                    )
                    if generic_pressure_hits:
                        failures.append(
                            "pressure_points use generic execution language instead of frame pressure tests: "
                            + ", ".join(generic_pressure_hits[:4])
                        )
                    if self._signal_overlap_count(flat_fields["pressure_points"], list(active_signals)) < 1:
                        failures.append(
                            "pressure_points do not test the frame against the task's original structural signals"
                        )

        if stage == Stage.EXPLORATION:
            forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
            if forbidden_hits:
                failures.append(
                    f"exploration artifact introduces ungrounded generic domain terms: {', '.join(forbidden_hits)}"
                )

            if active_signals and not any(tok in all_text for tok in ("expand", "define", "small", "spine", "fragment", "concrete")):
                failures.append(
                    "exploration artifact does not engage the task's structural signals"
                )

        return failures

    def _normalize(self, text: str) -> str:
        return " ".join(self._tokenize(text))

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 2}

    def _jaccard(self, a: str, b: str) -> float:
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _signal_overlap_count(self, text: str, signals: List[str]) -> int:
        signal_tokens = {
            STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED: {"expand", "define"},
            STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL: {"concrete", "small"},
            STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED: {"fragment", "spine"},
        }
        text_tokens = self._tokenize(text)
        overlap_count = 0
        for signal in signals:
            expected = signal_tokens.get(signal)
            if expected:
                if text_tokens & expected:
                    overlap_count += 1
                continue
            fallback_tokens = self._tokenize(signal.replace("_", " "))
            if fallback_tokens and text_tokens & fallback_tokens:
                overlap_count += 1
        return overlap_count
# ============================================================
# Evolution engine
# ============================================================

class EvolutionEngine:
    def propose_revision(self, regime: Regime, failure: FailureLog) -> RevisionProposal:
        if failure.recurrence_count >= 2 and failure.severity == Severity.HIGH:
            revision_type = "tighten"
        else:
            revision_type = "replace"

        target_failure = failure.observed_failure
        old_instruction = regime.dominant_line.text if failure.implicated_instruction_ids and regime.dominant_line.id in failure.implicated_instruction_ids else None
        new_instruction = None

        if regime.stage == Stage.SYNTHESIS and "coherence" in target_failure.lower():
            new_instruction = "If evidence directly weakens the central frame, revise the frame before integrating surrounding material."
            revision_type = "tighten"
            old_instruction = LIBRARY["SYN-P1"].text
        elif regime.stage == Stage.ADVERSARIAL and "weak objections" in target_failure.lower():
            new_instruction = "If a critique does not change the next action, omit it."
            revision_type = "add"
        elif regime.stage == Stage.OPERATOR and "forced closure" in target_failure.lower():
            new_instruction = "If two options are close, choose the one that preserves future flexibility."
            revision_type = "add"
        else:
            new_instruction = failure.missing_instruction or "Tighten the most ambiguous line into a decision-changing rule."

        return RevisionProposal(
            regime_name=regime.name,
            revision_type=revision_type,
            target_failure=target_failure,
            old_instruction=old_instruction,
            new_instruction=new_instruction,
            expected_increase=["regime stability", "failure containment"],
            expected_decrease=[target_failure],
            likely_side_effect=["reduced flexibility", "possible overcorrection if applied globally"],
            validation_test=(
                "Run the revised regime on 3 prompts that previously triggered the failure. "
                "Adopt only if the target failure decreases without destroying the regime's core behavior."
            ),
            adoption_recommendation="test_first",
        )


# ============================================================
# Runtime
# ============================================================

class CognitiveRuntime:
    def __init__(self, ollama_base_url: str = "http://localhost:11434") -> None:
        self.router = Router()
        self.composer = RegimeComposer()
        self.validator = OutputValidator()
        self.prompt_builder = PromptBuilder()
        self.evolver = EvolutionEngine()
        self.ollama = OllamaClient(base_url=ollama_base_url)

    def plan(
        self,
        bottleneck: str,
        risk_profile: Optional[Set[str]] = None,
        handoff_expected: bool = True,
        task_signals: Optional[List[str]] = None,
        risks_inferred: bool = False,
    ) -> Tuple[RoutingDecision, Regime, Handoff]:
        signals = task_signals if task_signals is not None else extract_structural_signals(bottleneck)
        risks = set(risk_profile or set()) if risks_inferred else infer_risk_profile(bottleneck, risk_profile)
        decision = self.router.route(bottleneck, task_signals=signals, risk_profile=risks)
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

# ============================================================
# JSON persistence
# ============================================================

def to_jsonable(obj: object) -> object:
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


class SessionStore:
    def __init__(self, root: str = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: SessionRecord, filename: Optional[str] = None) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = filename or f"run_{timestamp}.json"
        if not safe_name.endswith(".json"):
            safe_name += ".json"
        path = self.root / safe_name
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(record), f, indent=2, ensure_ascii=False)
        return path

    def load(self, filename: str) -> Dict[str, object]:
        path = self.root / filename
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def list_runs(self) -> List[str]:
        return sorted(p.name for p in self.root.glob("*.json"))


# ============================================================
# CLI helpers
# ============================================================

def print_routing(decision: RoutingDecision) -> None:
    print("ROUTING HEADER")
    print(f"- Current bottleneck: {decision.bottleneck}")
    print(f"- Primary regime: {decision.primary_regime.value}")
    print(f"- Runner-up regime: {decision.runner_up_regime.value if decision.runner_up_regime else 'none'}")
    print(f"- Why primary wins now: {decision.why_primary_wins_now}")
    print(f"- Switch trigger: {decision.switch_trigger}")
    print()


def print_handoff(handoff: Handoff) -> None:
    print("HANDOFF")
    print(f"- Current bottleneck: {handoff.current_bottleneck}")
    print(f"- Dominant frame: {handoff.dominant_frame}")
    print(f"- What is known: {', '.join(handoff.what_is_known)}")
    print(f"- What remains uncertain: {', '.join(handoff.what_remains_uncertain)}")
    print(f"- Active contradictions: {', '.join(handoff.active_contradictions)}")
    print(f"- Assumptions in play: {', '.join(handoff.assumptions_in_play)}")
    print(f"- Main risk if continue: {handoff.main_risk_if_continue}")
    print(f"- Recommended next regime: {handoff.recommended_next_regime.value if handoff.recommended_next_regime else 'none'}")
    print(f"- Minimum useful artifact: {handoff.minimum_useful_artifact}")
    print()


def print_validation(validation: Dict[str, object]) -> None:
    print("VALIDATION")
    for k, v in validation.items():
        if k == "parsed":
            continue
        print(f"- {k}: {v}")
    print()


def parse_risk_profile(raw: Optional[str]) -> Set[str]:
    if not raw:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


def make_record(
    task: str,
    risk_profile: Set[str],
    model: str,
    routing: RoutingDecision,
    regime: Regime,
    result: RegimeExecutionResult,
    handoff: Handoff,
) -> SessionRecord:
    return SessionRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        task=task,
        risk_profile=sorted(risk_profile),
        model=model,
        routing=to_jsonable(routing),
        regime=to_jsonable(regime),
        result=to_jsonable(result),
        handoff=to_jsonable(handoff),
    )


def cmd_run(args: argparse.Namespace) -> int:
    runtime = CognitiveRuntime(ollama_base_url=args.base_url)
    store = SessionStore(root=args.out_dir)
    risk_profile = parse_risk_profile(args.risks)

    decision, regime, result, handoff = runtime.execute(
        task=args.task,
        model=args.model,
        risk_profile=risk_profile,
        handoff_expected=not args.no_handoff,
    )

    print_routing(decision)
    print(regime.render())
    print()
    print("MODEL OUTPUT")
    print(result.raw_response)
    print()
    print_validation(result.validation)
    print_handoff(handoff)

    record = make_record(args.task, risk_profile, args.model, decision, regime, result, handoff)
    path = store.save(record, filename=args.save_as)
    print(f"Saved run to: {path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    runtime = CognitiveRuntime(ollama_base_url=args.base_url)
    decision, regime, handoff = runtime.plan(
        bottleneck=args.task,
        risk_profile=parse_risk_profile(args.risks),
        handoff_expected=not args.no_handoff,
    )
    print_routing(decision)
    print(regime.render())
    print()
    print_handoff(handoff)
    return 0


def cmd_list_runs(args: argparse.Namespace) -> int:
    store = SessionStore(root=args.out_dir)
    runs = store.list_runs()
    if not runs:
        print("No saved runs found.")
        return 0
    for run in runs:
        print(run)
    return 0


def cmd_show_run(args: argparse.Namespace) -> int:
    store = SessionStore(root=args.out_dir)
    data = store.load(args.filename)
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    runtime = CognitiveRuntime(ollama_base_url=args.base_url)
    models = runtime.ollama.list_models()
    print(json.dumps(models, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cognitive router prototype with Ollama-backed execution and JSON persistence."
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--out-dir", default="runs", help="Directory for saved JSON runs")

    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Route + compose + execute against Ollama + save JSON")
    run_p.add_argument("--task", required=True, help="Task or bottleneck description")
    run_p.add_argument("--model", default="llama3", help="Ollama model name")
    run_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    run_p.add_argument("--save-as", default=None, help="Optional output JSON filename")
    run_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    run_p.set_defaults(func=cmd_run)

    plan_p = sub.add_parser("plan", help="Route + compose without calling Ollama")
    plan_p.add_argument("--task", required=True, help="Task or bottleneck description")
    plan_p.add_argument("--risks", default="", help="Comma-separated risk profile tags")
    plan_p.add_argument("--no-handoff", action="store_true", help="Disable tail/transfer line where optional")
    plan_p.set_defaults(func=cmd_plan)

    list_p = sub.add_parser("list-runs", help="List saved run files")
    list_p.set_defaults(func=cmd_list_runs)

    show_p = sub.add_parser("show-run", help="Print a saved run JSON")
    show_p.add_argument("filename", help="Filename inside the runs directory")
    show_p.set_defaults(func=cmd_show_run)

    models_p = sub.add_parser("models", help="List models from local Ollama")
    models_p.set_defaults(func=cmd_models)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
