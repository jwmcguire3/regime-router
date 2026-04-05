from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
import hashlib
import re

from ..models import (
    ARTIFACT_FIELDS,
    ARTIFACT_HINTS,
    CANONICAL_FAILURE_IF_OVERUSED,
    Regime,
    RegimeExecutionResult,
    RoutingDecision,
    Stage,
    TaskAnalyzerOutput,
)
from ..orchestration.canonical_status import canonical_status_from_validation
from ..state import Handoff, RouterState

if TYPE_CHECKING:
    from ..routing import RegimeComposer

HANDOFF_PRIORITY_FIELDS: Dict[Stage, List[str]] = {
    Stage.EXPLORATION: ["candidate_frames", "selection_criteria", "unresolved_axes"],
    Stage.SYNTHESIS: ["central_claim", "key_tensions", "pressure_points"],
    Stage.EPISTEMIC: ["supported_claims", "contradictions", "decision_relevant_conclusions"],
    Stage.ADVERSARIAL: ["top_destabilizers", "break_conditions", "survivable_revisions"],
    Stage.OPERATOR: ["decision", "tradeoff_accepted", "fallback_trigger", "next_actions"],
    Stage.BUILDER: ["reusable_pattern", "modules", "implementation_sequence"],
}
DOMINANT_FRAME_PRIORITY_FIELDS: Tuple[str, ...] = ("central_claim", "decision", "reusable_pattern")


def resolve_next_regime(state: RouterState, stage: Stage, composer: "RegimeComposer") -> Regime:
    return state.resolve_regime(stage, composer.compose)


def build_router_state(
    *,
    bottleneck: str,
    decision: RoutingDecision,
    regime: Regime,
    signals: List[str],
    risks: Set[str],
    features: object,
    composer: "RegimeComposer",
    analyzer_result: Optional[TaskAnalyzerOutput] = None,
) -> RouterState:
    task_hash = hashlib.sha1(bottleneck.encode("utf-8")).hexdigest()[:12]
    normalized_bottleneck = decision.bottleneck.strip() if isinstance(decision.bottleneck, str) and decision.bottleneck.strip() else bottleneck
    stage_goal = regime.tail_line.text if regime.tail_line else "Produce the minimum useful typed artifact for this regime."
    runner_up_regime = composer.compose(decision.runner_up_regime, risk_profile=risks) if decision.runner_up_regime else None
    primary_name = decision.primary_regime.value if decision.primary_regime else "direct"
    primary_failure = (
        CANONICAL_FAILURE_IF_OVERUSED[decision.primary_regime]
        if decision.primary_regime
        else "Bypassing routing can miss hidden reasoning bottlenecks."
    )
    return RouterState(
        task_id=f"task-{task_hash}",
        task_summary=bottleneck[:180],
        current_bottleneck=normalized_bottleneck,
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
        uncertainties=[],
        contradictions=[],
        assumptions=[],
        substantive_assumptions=[],
        risks=sorted(risks) + [primary_failure],
        stage_goal=stage_goal,
        planned_switch_condition=decision.switch_trigger,
        observed_switch_cause=None,
        switch_trigger=decision.switch_trigger,
        recommended_next_regime=runner_up_regime,
        decision_pressure=float(getattr(features, "decision_pressure", 0)),
        evidence_demand=float(getattr(features, "evidence_demand", 0)),
        evidence_quality=float(analyzer_result.evidence_quality) if analyzer_result is not None else 0.0,
        fragility_pressure=float(getattr(features, "fragility_pressure", 0)),
        possibility_space_need=float(getattr(features, "possibility_space_need", 0)),
        detected_markers=dict(getattr(features, "detected_markers", {})),
        structural_signals=list(getattr(features, "structural_signals", [])),
        recurrence_potential=float(getattr(features, "recurrence_potential", 0)),
        policy_events=list(decision.policy_events),
        last_reentry_justification=None,
        last_state_delta=None,
        last_contract_delta=None,
    )


def update_router_state_from_execution(
    state: Optional[RouterState],
    result: RegimeExecutionResult,
    *,
    reason_entered: str,
    composer: "RegimeComposer",
) -> None:
    if state is None:
        return
    prior_dominant_frame = state.dominant_frame
    prior_recommended_next = state.recommended_next_regime.stage if state.recommended_next_regime is not None else None
    prior_contradictions = list(state.contradictions)
    parsed = result.validation.get("parsed", {})
    structurally_trustworthy = bool(
        result.validation.get("valid_json", False)
        and result.validation.get("required_keys_present", False)
        and result.validation.get("artifact_fields_present", False)
        and result.validation.get("artifact_type_matches", False)
        and result.validation.get("contract_controls_valid", False)
    )
    if isinstance(parsed, dict):
        explicit_assumptions = _extract_assumptions(parsed)
        if explicit_assumptions:
            state.substantive_assumptions = list(explicit_assumptions)

        if structurally_trustworthy:
            state.apply_dominant_frame(
                _resolve_dominant_frame(
                    parsed=parsed,
                    fallback_dominant_frame=state.dominant_frame or "",
                    fallback_regime_name=state.current_regime.name,
                )
            )
            recommended_next = parsed.get("recommended_next_regime")
            if isinstance(recommended_next, str):
                normalized_stage = recommended_next.strip().lower()
                if normalized_stage in Stage._value2member_map_:
                    state.recommended_next_regime = resolve_next_regime(state, Stage(normalized_stage), composer)

    semantic_failures = [str(f) for f in result.validation.get("semantic_failures", [])]
    if semantic_failures:
        state.update_inference_state(
            contradictions=state.contradictions + semantic_failures,
            assumptions=state.assumptions + ["Validation semantic failures were observed and should shape next regime."],
            substantive_assumptions=list(state.substantive_assumptions),
        )

    artifact_mapping = parsed.get("artifact", {}) if isinstance(parsed, dict) else {}
    if not isinstance(artifact_mapping, dict):
        artifact_mapping = {}
    canonical = canonical_status_from_validation(
        validation_result=result.validation,
        current_stage=state.current_regime.stage,
        artifact=artifact_mapping,
        state=state,
    )
    artifact_complete = canonical.terminal_signal == "completion"
    failure_signal_seen = canonical.terminal_signal in ("failure", "contradictory")
    summary_chunks = [
        "Execution yielded a completed artifact."
        if artifact_complete
        else "Execution output was incomplete or invalid."
    ]
    if canonical.completion_signal:
        summary_chunks.append(f"completion_signal={canonical.completion_signal}")
    if canonical.failure_signal:
        summary_chunks.append(f"failure_signal={canonical.failure_signal}")
    if not canonical.is_valid:
        summary_chunks.append("validation_failed=true")
    state.record_regime_step(
        regime=state.current_regime,
        reason_entered=reason_entered,
        completion_signal_seen=artifact_complete,
        failure_signal_seen=failure_signal_seen,
        outcome_summary=" ".join(summary_chunks),
    )
    state.last_state_delta = _compute_state_delta(
        state,
        parsed,
        prior_dominant_frame=prior_dominant_frame,
        prior_recommended_next=prior_recommended_next,
    )
    if semantic_failures and set(state.contradictions) != set(prior_contradictions):
        state.last_state_delta = "semantic_failures_introduced"
    if semantic_failures:
        state.last_contract_delta = "contract_invalidated_by_semantic_failure"
    elif canonical.control_conflict:
        state.last_contract_delta = "contract_invalidated_by_control_conflict"
    else:
        current_recommended_next = state.recommended_next_regime.stage if state.recommended_next_regime is not None else None
        if current_recommended_next != prior_recommended_next:
            state.last_contract_delta = "next_stage_contract_changed"
        else:
            state.last_contract_delta = "artifact_contract_advanced"


def handoff_from_state(state: Optional[RouterState]) -> Handoff:
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
            prior_artifact_summary="",
            recommended_next_regime_full=None,
            source_stage=None,
            source_regime_name="",
            created_from="fallback",
            stable_elements=[],
            tentative_elements=[],
            broken_elements=[],
            do_not_relitigate=[],
        )
    if state.latest_forward_handoff is not None:
        return state.latest_forward_handoff
    return Handoff(
        current_bottleneck=state.current_bottleneck,
        dominant_frame=state.dominant_frame or "",
        what_is_known=state.knowns,
        what_remains_uncertain=state.uncertainties,
        active_contradictions=state.contradictions,
        assumptions_in_play=state.assumptions,
        main_risk_if_continue=_best_active_risk(state.risks),
        recommended_next_regime=state.recommended_next_regime.stage if state.recommended_next_regime else None,
        minimum_useful_artifact=state.stage_goal,
        prior_artifact_summary="",
        recommended_next_regime_full=state.recommended_next_regime,
        source_stage=state.current_regime.stage if state else None,
        source_regime_name=state.current_regime.name if state else "",
        created_from="fallback",
        stable_elements=[],
        tentative_elements=[],
        broken_elements=[],
        do_not_relitigate=[],
    )


def _unique_preserve(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _determine_next_stage(result: RegimeExecutionResult, state: RouterState) -> Optional[Stage]:
    parsed = result.validation.get("parsed", {})
    if isinstance(parsed, dict):
        candidate = parsed.get("recommended_next_regime")
        if isinstance(candidate, str):
            normalized = candidate.strip().lower()
            if normalized in Stage._value2member_map_:
                return Stage(normalized)
    if state.recommended_next_regime is not None:
        return state.recommended_next_regime.stage
    return None


def _compute_state_delta(
    state: RouterState,
    parsed: object,
    *,
    prior_dominant_frame: Optional[str],
    prior_recommended_next: Optional[Stage],
) -> str:
    if (state.dominant_frame or "") != (prior_dominant_frame or ""):
        return "dominant_frame_changed"
    parsed_contradictions = tuple(_extract_contradictions(parsed))
    if parsed_contradictions and parsed_contradictions != tuple(state.contradictions):
        return "contradictions_changed"
    parsed_uncertainties = tuple(_extract_uncertainties(parsed))
    if parsed_uncertainties and parsed_uncertainties != tuple(state.uncertainties):
        return "uncertainties_changed"
    current_recommended_next = state.recommended_next_regime.stage if state.recommended_next_regime else None
    if current_recommended_next != prior_recommended_next:
        return "recommended_next_stage_changed"
    return "no_material_state_delta"


def _first_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", compact, maxsplit=1)
    sentence = parts[0].strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def _normalize_field_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return "; ".join(parts)
    return ""


def _resolve_dominant_frame(
    *,
    parsed: object,
    fallback_dominant_frame: str,
    fallback_regime_name: str,
) -> str:
    if isinstance(parsed, dict):
        artifact = parsed.get("artifact", {})
        if isinstance(artifact, dict):
            for field in DOMINANT_FRAME_PRIORITY_FIELDS:
                value = artifact.get(field)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return fallback_dominant_frame or fallback_regime_name


def _finding_from_field(field: str, normalized: str) -> str:
    templates = {
        "decision": "The priority decision is {}",
        "central_claim": "The central claim is {}",
        "rationale": "{}",
        "tradeoff_accepted": "The accepted tradeoff is {}",
        "next_actions": "Immediate next actions are {}",
        "fallback_trigger": "A fallback is triggered if {}",
    }
    template = templates.get(field)
    text = template.format(normalized) if template else normalized
    return _first_sentence(text)


def _extract_key_findings(result: RegimeExecutionResult) -> Tuple[List[str], str]:
    parsed = result.validation.get("parsed", {})
    if not isinstance(parsed, dict):
        return [], ""
    artifact = parsed.get("artifact", {})
    if not isinstance(artifact, dict):
        return [], ""
    field_names = ARTIFACT_FIELDS.get(result.stage, [])
    top_fields = HANDOFF_PRIORITY_FIELDS.get(result.stage, field_names[:5])
    findings: List[str] = []
    summary_bits: List[str] = []
    for field in top_fields:
        normalized = _normalize_field_value(artifact.get(field))
        if not normalized:
            continue
        finding = _finding_from_field(field, normalized)
        if finding:
            findings.append(finding)
            summary_bits.append(finding)
        if len(findings) >= 5:
            break
    return findings, " ".join(summary_bits)


def _extract_uncertainties(parsed: object) -> List[str]:
    if not isinstance(parsed, dict):
        return []
    out: List[str] = []
    artifact = parsed.get("artifact", {})
    if isinstance(artifact, dict):
        for field in ("plausible_but_unproven", "unresolved_axes", "open_questions"):
            value = artifact.get(field)
            if isinstance(value, list):
                out.extend(_first_sentence(str(v)) for v in value if str(v).strip())
            elif isinstance(value, str) and value.strip():
                out.append(_first_sentence(value))
    failure_signal = parsed.get("failure_signal")
    if isinstance(failure_signal, str) and failure_signal.strip():
        out.append(_first_sentence(f"Potential failure signal: {failure_signal.strip()}"))
    return _unique_preserve([item for item in out if item])


def _extract_contradictions(parsed: object) -> List[str]:
    if not isinstance(parsed, dict):
        return []
    artifact = parsed.get("artifact", {})
    if not isinstance(artifact, dict):
        return []

    contradictions = artifact.get("contradictions")
    out: List[str] = []
    if isinstance(contradictions, list):
        out.extend(_first_sentence(str(v)) for v in contradictions if str(v).strip())
    elif isinstance(contradictions, str) and contradictions.strip():
        out.append(_first_sentence(contradictions.strip()))
    return _unique_preserve([item for item in out if item])


def _extract_assumptions(parsed: object) -> List[str]:
    if not isinstance(parsed, dict):
        return []
    artifact = parsed.get("artifact", {})
    if not isinstance(artifact, dict):
        return []

    out: List[str] = []
    for field in ("hidden_assumptions", "assumptions"):
        value = artifact.get(field)
        if isinstance(value, list):
            out.extend(_first_sentence(str(v)) for v in value if str(v).strip())
        elif isinstance(value, str) and value.strip():
            out.append(_first_sentence(value.strip()))
    return _unique_preserve([item for item in out if item])


def _build_artifact_summary(result: RegimeExecutionResult, findings_summary: str) -> str:
    parsed = result.validation.get("parsed", {})
    if not isinstance(parsed, dict):
        return ""
    artifact = parsed.get("artifact", {})
    if not isinstance(artifact, dict):
        return ""
    decision = _normalize_field_value(artifact.get("decision") or artifact.get("central_claim"))
    rationale = _normalize_field_value(artifact.get("rationale"))
    tradeoff = _normalize_field_value(artifact.get("tradeoff_accepted"))
    sentences: List[str] = []
    if decision:
        sentences.append(_first_sentence(f"The {result.stage.value} stage decided that {decision}"))
    elif findings_summary:
        sentences.append(_first_sentence(f"The {result.stage.value} stage established that {findings_summary}"))
    if rationale:
        sentences.append(_first_sentence(f"This was driven by {rationale}"))
    if tradeoff:
        sentences.append(_first_sentence(f"The accepted tradeoff is {tradeoff}"))
    summary = " ".join(sentences[:3]).strip()
    return summary[:500]


def _classify_elements(result: RegimeExecutionResult) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Classify handoff elements by confidence level based on validation state."""
    stable: List[str] = []
    tentative: List[str] = []
    broken: List[str] = []
    do_not_relitigate: List[str] = []

    is_valid = bool(result.validation.get("is_valid", False))
    semantic_failures = [str(f) for f in result.validation.get("semantic_failures", [])]
    failed_field_names = set()

    parsed = result.validation.get("parsed", {})
    artifact = parsed.get("artifact", {}) if isinstance(parsed, dict) else {}
    if not isinstance(artifact, dict):
        artifact = {}

    priority_fields = HANDOFF_PRIORITY_FIELDS.get(result.stage, [])

    for failure_msg in semantic_failures:
        for field_name in priority_fields:
            if failure_msg.startswith(f"{field_name} "):
                failed_field_names.add(field_name)

    for field_name in priority_fields:
        value = artifact.get(field_name)
        if not value:
            continue
        normalized = str(value).strip() if isinstance(value, str) else str(value)
        if not normalized:
            continue

        if field_name in failed_field_names:
            broken.append(f"{field_name}: {normalized[:120]}")
        elif is_valid:
            stable.append(f"{field_name}: {normalized[:120]}")
            do_not_relitigate.append(field_name)
        else:
            tentative.append(f"{field_name}: {normalized[:120]}")

    return stable, tentative, broken, do_not_relitigate


def _minimum_useful_artifact(next_stage: Optional[Stage], state: RouterState) -> str:
    next_artifact = ARTIFACT_HINTS.get(next_stage) if next_stage is not None else None
    endpoint = None
    if state.task_classification and isinstance(state.task_classification.get("likely_endpoint_regime"), str):
        endpoint = state.task_classification.get("likely_endpoint_regime")
    endpoint = endpoint or "operator"
    if next_artifact:
        return f"Progress toward endpoint '{endpoint}' by delivering a valid '{next_artifact}' artifact."
    return f"Progress toward endpoint '{endpoint}' with the minimum useful typed artifact."


def _best_active_risk(risks: List[str]) -> str:
    if not risks:
        return ""
    for risk in risks:
        lowered = risk.lower()
        if any(token in lowered for token in ("active", "current", "immediate", "now", "blocked", "invalid")):
            return risk
    return risks[-1]


def compute_forward_handoff(
    result: RegimeExecutionResult,
    router_state: RouterState,
    regime: Regime,
    composer: Optional["RegimeComposer"] = None,
) -> Handoff:
    parsed = result.validation.get("parsed", {})
    key_findings, findings_summary = _extract_key_findings(result)
    knowns = _unique_preserve(key_findings)[:5]
    uncertainties = _extract_uncertainties(parsed)
    explicit_contradictions = _extract_contradictions(parsed)
    explicit_assumptions = _extract_assumptions(parsed)
    contradictions = explicit_contradictions if explicit_contradictions else list(router_state.contradictions)
    assumptions = explicit_assumptions if explicit_assumptions else list(router_state.assumptions)
    if explicit_assumptions:
        router_state.substantive_assumptions = list(explicit_assumptions)
    next_stage = _determine_next_stage(result, router_state)
    if composer is None:
        from ..routing import RegimeComposer

        effective_composer = RegimeComposer()
    else:
        effective_composer = composer
    next_regime = resolve_next_regime(router_state, next_stage, effective_composer) if next_stage is not None else None
    dominant_frame = _resolve_dominant_frame(
        parsed=parsed,
        fallback_dominant_frame=router_state.dominant_frame or "",
        fallback_regime_name=regime.name,
    )
    artifact_summary = _build_artifact_summary(result, findings_summary)
    stable, tentative, broken, do_not_relitigate = _classify_elements(result)
    return Handoff(
        current_bottleneck=router_state.current_bottleneck,
        dominant_frame=dominant_frame,
        what_is_known=knowns,
        what_remains_uncertain=uncertainties,
        active_contradictions=contradictions,
        assumptions_in_play=assumptions,
        main_risk_if_continue=_best_active_risk(router_state.risks),
        recommended_next_regime=next_stage,
        minimum_useful_artifact=_minimum_useful_artifact(next_stage, router_state),
        prior_artifact_summary=artifact_summary,
        recommended_next_regime_full=next_regime,
        source_stage=regime.stage,
        source_regime_name=regime.name,
        created_from="switch" if router_state.switches_executed > 0 else "initial_run",
        stable_elements=stable,
        tentative_elements=tentative,
        broken_elements=broken,
        do_not_relitigate=do_not_relitigate,
    )
