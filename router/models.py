from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypedDict

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


class ControlAuthority(str, Enum):
    HARD_VETO = "hard_veto"
    SOFT_GUARDRAIL = "soft_guardrail"
    ADVISORY_ONLY = "advisory_only"


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
    primary_regime: Optional[Stage]
    runner_up_regime: Optional[Stage]
    why_primary_wins_now: str
    switch_trigger: str
    pre_policy_primary_regime: Optional[Stage] = None
    pre_policy_runner_up_regime: Optional[Stage] = None
    likely_endpoint_regime: str = "operator"
    endpoint_confidence: float = 0.7
    confidence: "RegimeConfidenceResult" = field(default_factory=lambda: RegimeConfidenceResult.low_default())
    deterministic_stage_scores: Dict[Stage, int] = field(default_factory=dict)
    deterministic_score_summary: str = ""
    deterministic_score_contributions: Dict[Stage, List[str]] = field(default_factory=dict)
    analyzer_enabled: bool = False
    analyzer_used: bool = False
    analyzer_changed_primary: bool = False
    analyzer_changed_runner_up: bool = False
    analyzer_summary: Optional[str] = None
    inference_quality: str = "fallback"
    policy_warnings: List[str] = field(default_factory=list)
    policy_actions: List[str] = field(default_factory=list)
    policy_events: List["PolicyEvent"] = field(default_factory=list)


@dataclass(frozen=True)
class ReentryJustification:
    defect_class: str
    repair_target: str
    contract_delta: str
    state_delta: str


@dataclass(frozen=True)
class ReentryDecision:
    allowed: bool
    reason: str
    justification: Optional[ReentryJustification] = None


@dataclass(frozen=True)
class PolicyEvent:
    rule_name: str
    authority: str
    consumed_features: List[str]
    action: str
    detail: str


@dataclass
class RegimeConfidenceResult:
    level: str
    rationale: str
    top_stage_score: int
    runner_up_score: int
    score_gap: int
    nontrivial_stage_count: int
    weak_lexical_dependence: bool
    structural_feature_state: str

    @classmethod
    def low_default(cls) -> "RegimeConfidenceResult":
        return cls(
            level=Severity.LOW.value,
            rationale="Routing confidence defaults to low until scoring evidence is computed.",
            top_stage_score=0,
            runner_up_score=0,
            score_gap=0,
            nontrivial_stage_count=0,
            weak_lexical_dependence=True,
            structural_feature_state="sparse",
        )


@dataclass(frozen=True)
class TaskAnalyzerOutput:
    bottleneck_label: str
    candidate_regimes: List[Stage]
    stage_scores: Dict[Stage, float]
    structural_signals: List[str]
    decision_pressure: int
    evidence_quality: int
    recurrence_potential: int
    confidence: float
    rationale: str
    likely_endpoint_regime: Stage = Stage.OPERATOR
    endpoint_confidence: float = 0.7


@dataclass(frozen=True)
class EmbeddingScore:
    stage_scores: Dict[Stage, float]
    best_stage: Stage
    best_score: float
    below_threshold: bool


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


class RegimeOutputPayload(TypedDict):
    regime: str
    purpose: str
    artifact_type: str
    artifact: Dict[str, Any]
    completion_signal: str
    failure_signal: str
    recommended_next_regime: str


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

REGIME_PURPOSE_HINTS: Dict[Stage, str] = {
    Stage.EXPLORATION: "Generate and compare structurally distinct candidate frames.",
    Stage.SYNTHESIS: "Produce the strongest coherent interpretation from live signals.",
    Stage.EPISTEMIC: "Separate supported claims from uncertainty and gaps.",
    Stage.ADVERSARIAL: "Stress test the frame with destabilizers and break conditions.",
    Stage.OPERATOR: "Commit to a concrete decision with executable next moves.",
    Stage.BUILDER: "Convert insight into reusable architecture and modules.",
}

COMPLETION_SIGNAL_HINTS: Dict[Stage, str] = {
    Stage.EXPLORATION: "selection_criteria_ready",
    Stage.SYNTHESIS: "coherent_frame_stable",
    Stage.EPISTEMIC: "evidence_boundary_clear",
    Stage.ADVERSARIAL: "critical_breakpoints_mapped",
    Stage.OPERATOR: "decision_committed_with_actions",
    Stage.BUILDER: "blueprint_ready_for_build_sequence",
}

FAILURE_SIGNAL_HINTS: Dict[Stage, str] = {
    Stage.EXPLORATION: "frames_not_structurally_distinct",
    Stage.SYNTHESIS: "frame_collapses_under_pressure_points",
    Stage.EPISTEMIC: "insufficient_support_for_key_claims",
    Stage.ADVERSARIAL: "destabilizers_unresolved_or_redundant",
    Stage.OPERATOR: "decision_not_actionable_under_constraints",
    Stage.BUILDER: "architecture_not_modular_or_reusable",
}


DOMINANT_FAILURE_MAP: Dict[str, List[str]] = {
    "EXP-D1": ["sprawl", "fake_divergence", "no_transition"],
    "SYN-D1": ["premature_lock", "false_unification"],
    "SYN-D2": ["false_unification", "coherence_over_truth"],
    "EPI-D1": ["flat_output", "under_synthesis"],
    "EPI-D2": ["weak_gestalt", "decision_drag"],
    "ADV-D1": ["tunnel_vision", "critique_sludge", "strawman_attack"],
    "OPR-D1": ["forced_closure", "hidden_tradeoff"],
    "BLD-D1": ["overbuild", "premature_productization"],
}

FAILURE_SUPPRESSOR_MAP: Dict[str, List[str]] = {
    "sprawl": ["EXP-P1"],
    "fake_divergence": ["EXP-P1", "EXP-S1"],
    "premature_lock": ["SYN-P1"],
    "false_unification": ["SYN-P1", "SYN-P2"],
    "coherence_over_truth": ["SYN-P1", "SYN-P2"],
    "flat_output": ["EPI-P1", "EPI-S1"],
    "under_synthesis": ["EPI-P1", "EPI-S1"],
    "weak_gestalt": ["EPI-P2"],
    "decision_drag": ["EPI-P2", "EPI-S1"],
    "elegant_theory_drift": ["EPI-D2"],
    "tunnel_vision": ["ADV-P1", "ADV-S1"],
    "critique_sludge": ["ADV-P2"],
    "strawman_attack": ["ADV-P1"],
    "forced_closure": ["OPR-P1", "OPR-G1"],
    "hidden_tradeoff": ["OPR-P1", "OPR-S2"],
    "overbuild": ["BLD-P1"],
    "premature_productization": ["BLD-P1", "BLD-S2"],
}

DOMINANT_SELECTION_RULES: Dict[Stage, Dict[str, str]] = {
    Stage.SYNTHESIS: {
        "default": "SYN-D1",
        "sprawl": "SYN-D2",
    },
    Stage.EPISTEMIC: {
        "default": "EPI-D1",
        "elegant_theory_drift": "EPI-D2",
    },
}


# ============================================================
# Router
