from __future__ import annotations

from typing import List, Optional, Set
import hashlib

from ..models import CANONICAL_FAILURE_IF_OVERUSED, Regime, RegimeExecutionResult, RoutingDecision, Stage
from ..routing import RegimeComposer
from ..state import Handoff, RouterState


def resolve_next_regime(state: RouterState, stage: Stage, composer: RegimeComposer) -> Regime:
    return state.resolve_regime(stage, composer.compose)


def build_router_state(
    *,
    bottleneck: str,
    decision: RoutingDecision,
    regime: Regime,
    signals: List[str],
    risks: Set[str],
    features: object,
    composer: RegimeComposer,
) -> RouterState:
    task_hash = hashlib.sha1(bottleneck.encode("utf-8")).hexdigest()[:12]
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


def update_router_state_from_execution(
    state: Optional[RouterState],
    result: RegimeExecutionResult,
    *,
    reason_entered: str,
    composer: RegimeComposer,
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
                    state.recommended_next_regime = resolve_next_regime(state, Stage(normalized_stage), composer)

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
