# Cognitive Router — Control Surface Code Change Spec vNext

Status: implementation spec  
Depends on: `docs/CONTROL_SURFACE_POLICY_vNEXT.md`  
Purpose: translate policy into exact edits by file and function

---

## 0. Implementation goals

This spec implements four changes:

1. deterministic features stop acting as hard regime-legitimacy police
2. Builder stops being blocked by sparse lexical thresholds
3. reentry becomes qualified and defect-driven instead of history-banned
4. stop policy reconciles actual stage/artifact state before endpoint completion

This spec does **not** widen regime meanings, add new regimes, or redesign prompt language.

---

## 1. File-by-file change plan

### 1.1 `router/models.py`

#### A. Extend `RoutingDecision`

Add fields:

- `pre_policy_primary_regime: Optional[Stage] = None`
- `pre_policy_runner_up_regime: Optional[Stage] = None`
- `policy_warnings: List[str] = field(default_factory=list)`
- `policy_actions: List[str] = field(default_factory=list)`

Purpose:
- preserve analyzer choice before soft guardrails run
- make pre/post-policy routing visible in run records

#### B. Add control authority enum

Add:

```python
class ControlAuthority(str, Enum):
    HARD_VETO = "hard_veto"
    SOFT_GUARDRAIL = "soft_guardrail"
    ADVISORY_ONLY = "advisory_only"
```

Purpose:
- make rule authority explicit in code and logging

#### C. Add reentry-support dataclasses

Add:

```python
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
```

Purpose:
- make same-stage retry and prior-stage reentry policy machine-readable

#### D. Add policy event model for observability

Add:

```python
@dataclass(frozen=True)
class PolicyEvent:
    rule_name: str
    authority: str
    consumed_features: List[str]
    action: str
    detail: str
```

Purpose:
- serialize which rules actually exercised authority in a run

---

### 1.2 `router/state.py`

#### A. Extend `SwitchDecisionRecord`

Add fields:

- `defect_class: Optional[str] = None`
- `repair_target: Optional[str] = None`
- `contract_delta: Optional[str] = None`
- `state_delta: Optional[str] = None`
- `reentry_allowed: Optional[bool] = None`

Purpose:
- persist qualified reentry decisions in run history

#### B. Extend `RouterState`

Add fields:

- `policy_events: List[PolicyEvent] = field(default_factory=list)`
- `last_reentry_justification: Optional[ReentryJustification] = None`
- `last_state_delta: Optional[str] = None`
- `last_contract_delta: Optional[str] = None`

Purpose:
- keep policy-layer decisions visible and serializable

#### C. Update `record_switch_decision(...)`

Change signature to accept optional:

- `defect_class`
- `repair_target`
- `contract_delta`
- `state_delta`
- `reentry_allowed`

Populate the new `SwitchDecisionRecord` fields.

#### D. Add `record_policy_event(...)`

Add helper method:

```python
def record_policy_event(self, event: PolicyEvent) -> None:
    self.policy_events.append(event)
```

#### E. Update JSON restoration

Update `router_state_from_jsonable(...)` to restore:

- `policy_events`
- extended `SwitchDecisionRecord` fields
- `last_reentry_justification`
- `last_state_delta`
- `last_contract_delta`

---

### 1.3 `router/analyzer.py`

#### A. Replace hard demotion block inside `TaskAnalyzer.decision_from_analysis(...)`

Current logic hard-demotes:

- operator without decision evidence -> exploration
- builder without recurrence -> exploration
- adversarial without fragility -> exploration

Replace with a policy application helper.

Add helper:

```python
def _apply_routing_policy(
    self,
    *,
    primary: Stage,
    runner_up: Stage,
    analyzer_result: TaskAnalyzerOutput,
    routing_features: RoutingFeatures,
) -> tuple[Stage, Stage, list[str], list[str]]:
```

Return:
- post-policy primary
- post-policy runner_up
- warnings
- actions

#### B. New rule behavior

##### Rule: operator_without_decision_support
Authority: `soft_guardrail`

Behavior:
- do not rewrite `primary`
- if `primary == Stage.OPERATOR` and `decision_pressure == 0` and no decision marker family:
  - append warning
  - optionally bias `runner_up` to `Stage.EXPLORATION` if not already operator
  - reduce effective confidence one tier only if analyzer confidence < 0.8

No hard override.

##### Rule: builder_without_recurrence_support
Authority: `advisory_only` for primary, `soft_guardrail` for endpoint

Behavior:
- do not rewrite `primary`
- append warning when `primary == Stage.BUILDER` and `recurrence_potential == 0`
- for `likely_endpoint_regime == Stage.BUILDER` and `analyzer_result.recurrence_potential == 0`:
  - keep endpoint as builder only if analyzer confidence >= 0.8
  - otherwise soften endpoint to operator and append policy action `endpoint_softened_builder_to_operator`

##### Rule: adversarial_without_fragility_support
Authority: `advisory_only`

Behavior:
- do not rewrite `primary`
- append warning when `primary == Stage.ADVERSARIAL` and `fragility_pressure == 0`
- optionally reduce confidence one tier only if analyzer confidence < 0.5 and score gap is narrow

#### C. Preserve pre-policy analyzer choice

Before policy application, set:

- `pre_policy_primary_regime = primary`
- `pre_policy_runner_up_regime = runner_up`

Store warnings/actions in `RoutingDecision.policy_warnings` and `RoutingDecision.policy_actions`.

#### D. Analyzer summary

Append policy action details into `analyzer_summary` instead of old demotion strings.

Remove strings like:
- `operator proposed without decision evidence; demoted to exploration`
- `builder proposed without recurrence potential; demoted to exploration`
- `adversarial proposed without fragility pressure; demoted to exploration`

Replace with:
- `operator support weak; soft guardrail only`
- `builder support weak; advisory only`
- `adversarial support weak; advisory only`
- `builder endpoint softened to operator` when applicable

---

### 1.4 `router/routing/feature_extraction.py`

#### A. Do not widen feature taxonomy in this patch

No new pressures in this implementation pass.

#### B. Add explicit marker-family serialization helper

Add:

```python
def explain_feature_matches(features: RoutingFeatures) -> dict[str, list[str]]:
    return dict(features.detected_markers)
```

Purpose:
- make marker-family matches easy to carry into run records and tests

#### C. Keep feature computation unchanged for now

Reason:
- this patch downgrades authority before changing signal surface

---

### 1.5 `router/runtime/state_updater.py`

#### A. Update `build_router_state(...)`

Add optional parameter:

- `analyzer_result: Optional[TaskAnalyzerOutput] = None`

Behavior changes:
- set `evidence_quality` from analyzer result when present, else from feature extraction
- keep `decision_pressure` and `recurrence_potential` from feature extraction for now
- initialize `policy_events` empty
- set `last_state_delta = None`
- set `last_contract_delta = None`

Reason:
- current field name is `evidence_quality` in state but the feature source is `evidence_demand`; use analyzer output when available to avoid semantic mismatch

#### B. Add helper to compute state delta after execution

Add:

```python
def _compute_state_delta(
    state: RouterState,
    parsed: object,
    *,
    prior_dominant_frame: Optional[str],
    prior_recommended_next: Optional[Stage],
) -> str:
```

Return one concise text summary of what materially changed.

Suggested priority:
- dominant frame changed
- contradiction set changed
- uncertainty set changed
- recommended next stage changed
- semantic failures introduced
- none

#### C. Update `update_router_state_from_execution(...)`

Before mutation, capture:
- prior dominant frame
- prior recommended next stage

After mutation, set:
- `state.last_state_delta`
- `state.last_contract_delta`

Contract delta rules:
- if semantic failures exist -> `contract_invalidated_by_semantic_failure`
- if completion/failure signals conflict -> `contract_invalidated_by_control_conflict`
- if recommended next regime changed -> `next_stage_contract_changed`
- else `artifact_contract_advanced`

#### D. Update `compute_forward_handoff(...)`

Populate handoff additions:
- `stable_elements`
- `tentative_elements`
- `broken_elements`
- `do_not_relitigate`
- include `main_risk_if_continue` from current active risk, not stale tail only when possible

No schema change needed here because those fields already exist.

---

### 1.6 `router/orchestration/transition_rules.py`

#### A. Replace `ALLOWED_PATHWAYS`

Current pathways are monotonic and too narrow.

Replace with two structures:

```python
DEFAULT_FORWARD_PATHWAYS = {
    Stage.EXPLORATION: {Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.OPERATOR},
    Stage.SYNTHESIS: {Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR},
    Stage.EPISTEMIC: {Stage.ADVERSARIAL, Stage.OPERATOR},
    Stage.ADVERSARIAL: {Stage.OPERATOR},
    Stage.OPERATOR: set(),
    Stage.BUILDER: set(),
}

CONDITIONAL_REENTRY_PATHWAYS = {
    Stage.EXPLORATION: {Stage.EXPLORATION, Stage.ADVERSARIAL, Stage.BUILDER},
    Stage.SYNTHESIS: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.BUILDER},
    Stage.EPISTEMIC: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.BUILDER},
    Stage.ADVERSARIAL: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.BUILDER},
    Stage.OPERATOR: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR, Stage.BUILDER},
    Stage.BUILDER: {Stage.EXPLORATION, Stage.SYNTHESIS, Stage.EPISTEMIC, Stage.ADVERSARIAL, Stage.OPERATOR, Stage.BUILDER},
}
```

#### B. Add reentry-qualification helpers

Add:

```python
def defect_class_from_context(... ) -> Optional[str]
def repair_target_for_stage(target_stage: Stage, defect_class: str) -> str
def build_reentry_justification(... ) -> Optional[ReentryJustification]
def transition_requires_justification(current_stage: Stage, next_stage: Stage, state: RouterState) -> bool
```

#### C. Update `next_stage(...)`

New behavior:
- keep default forward transitions for normal completion/failure flows
- allow conditional backward or same-stage transitions only when a `ReentryJustification` can be built
- stop using `state.recurrence_potential >= 2.0` as the sole Builder transition trigger from operator

Replace:

```python
if current_stage == Stage.OPERATOR and completion_signal and state.recurrence_potential >= 2.0:
    return Stage.BUILDER
```

With:
- prefer Builder only when detection or recommended next stage suggests Builder and the handoff/task shape indicates repeated structure, not one-off closure
- otherwise return `None`

#### D. Keep `control_failure_regime_mismatch(...)`

But treat it as a defect source for possible reentry rather than automatic monotonic move only.

---

### 1.7 `router/orchestration/switch_orchestrator.py`

#### A. Capture reentry justification

After `resolved_next_stage = next_stage(...)`, compute:

```python
reentry_justification = build_reentry_justification(
    state=state,
    current_stage=current_stage,
    next_stage=resolved_next_stage,
    completion_signal=completion_signal,
    failure_signal=failure_signal,
    detection=detection,
    output=output,
)
```

Store in:
- `state.last_reentry_justification`

#### B. Set observed cause more precisely

Current code sets `observed_switch_cause` from a narrow priority order.

Change to prefer:
- defect class when reentry justification exists
- otherwise current existing cause selection

#### C. Do not treat absence of structured switching signal as a blanket no-switch if a valid reentry justification exists from output mismatch or semantic invalidation.

---

### 1.8 `router/runtime/session_runtime.py`

#### A. Remove blanket same-stage denial block

Current block:
- denies any same-stage recommendation and stops with `loop_prevented_same_stage`

Replace with:
- call `_evaluate_reentry(...)`
- allow same-stage retry only if qualified justification exists and `state.last_state_delta` is non-empty and non-trivial
- otherwise deny

#### B. Remove blanket prior-stage denial block

Current block:
- denies any previously executed stage unless collapse reentry is allowed

Replace with:
- prior-stage revisitation allowed when `_evaluate_reentry(...)` returns allowed
- collapse is one `defect_class`, not the sole exception

#### C. Add helper methods

Add:

```python
def _evaluate_reentry(
    self,
    *,
    state: RouterState,
    next_stage: Stage,
    reason_for_switch: str,
) -> ReentryDecision:
```

Rules:
- same-stage with unchanged brief/cause -> deny
- previously visited stage with empty `last_state_delta` -> deny
- repeated oscillation with same cause and same target -> deny
- otherwise allow only if `state.last_reentry_justification` is present and complete

Add:

```python
def _is_ping_pong(self, state: RouterState, next_stage: Stage) -> bool:
```

Suggested behavior:
- inspect last 2 executed transitions
- deny if returning to same pair for same observed cause with no new state delta

#### D. Update `record_switch_decision(...)` calls

Pass the new fields from the reentry decision:
- `defect_class`
- `repair_target`
- `contract_delta`
- `state_delta`
- `reentry_allowed`

#### E. Remove `_allow_collapse_reentry(...)`

Delete helper entirely after reentry qualification is in place.

---

### 1.9 `router/orchestration/stop_policy.py`

#### A. Remove `BUILDER_RECURRENCE_THRESHOLD`

Delete:

```python
BUILDER_RECURRENCE_THRESHOLD = 7
```

#### B. Remove `_builder_blocked(...)`

Delete the helper and the early stop branch:

```python
if self._builder_blocked(router_state):
    ...
```

Builder should no longer be blocked by a single lexical threshold.

#### C. Tighten endpoint completion

Add helper:

```python
def _artifact_matches_current_stage(self, validation_result: Mapping[str, object], current_stage: Stage) -> bool:
```

Behavior:
- parse payload
- compare `artifact_type` to `ARTIFACT_HINTS[current_stage]`
- compare `regime` field to `current_stage.value`
- require both to match

Then modify `should_stop(...)`:
- require `_artifact_matches_current_stage(...)` in addition to `_artifact_complete(...)`
- if current stage is at/past endpoint but artifact/regime mismatch exists, do not stop

#### D. Forward recommendation deferral should allow justified reentry

Current `_should_defer_stop_for_forward_recommendation(...)` only defers for forward stage rank.

Change signature to accept `router_state` only and inspect:
- higher-rank forward move, or
- same/lower-rank move with valid `last_reentry_justification`

New rule:
- justified reentry can defer stop
- unjustified same/lower-rank recommendation cannot

---

### 1.10 `router/runtime/planner.py`

#### A. Thread analyzer result into state build

If planner currently has analyzer result available before building state, pass it to `build_router_state(...)`.

Required edit:
- update call site to include `analyzer_result=...`

Purpose:
- let state use analyzer evidence quality rather than feature-extracted evidence demand

If planner does not currently keep analyzer result separate from `RoutingDecision`, add a small local variable before converting to `RoutingDecision`.

---

## 2. Observability requirements

### 2.1 Run serialization

After these edits, every run record must expose:

- pre-policy analyzer primary and runner-up
- post-policy primary and runner-up
- policy warnings/actions
- feature marker families used by policy
- policy events with authority level
- reentry decision fields in switch history
- last state delta and last contract delta

### 2.2 Minimum implementation location

Use `make_record(...)` / `to_jsonable(...)` path already present in `router/state.py`; do not add a parallel logger.

---

## 3. Tests to add or update

### 3.1 Analyzer policy tests

File target:
- existing analyzer tests or new `tests/test_analyzer_policy.py`

Required tests:

1. `test_operator_primary_not_hard_demoted_when_decision_support_absent`
2. `test_adversarial_primary_not_hard_demoted_when_fragility_support_absent`
3. `test_builder_primary_not_hard_demoted_when_recurrence_support_absent`
4. `test_builder_endpoint_softened_when_support_absent_and_confidence_not_high`
5. `test_pre_policy_and_post_policy_routing_are_serialized`

### 3.2 Stop policy tests

File target:
- existing stop policy tests or new `tests/test_stop_policy_control_surface.py`

Required tests:

1. `test_stop_policy_does_not_block_builder_by_recurrence_threshold`
2. `test_stop_policy_refuses_endpoint_completion_on_artifact_stage_mismatch`
3. `test_stop_policy_allows_justified_reentry_to_defer_stop`

### 3.3 Reentry/session runtime tests

File target:
- existing orchestration tests or new `tests/test_reentry_policy.py`

Required tests:

1. `test_same_stage_retry_denied_without_state_delta`
2. `test_same_stage_retry_allowed_with_contract_and_state_delta`
3. `test_prior_stage_reentry_allowed_when_defect_class_present`
4. `test_prior_stage_reentry_denied_without_justification`
5. `test_ping_pong_denied_for_repeated_same_cause`
6. `test_collapse_is_one_reentry_class_not_the_only_one`

### 3.4 Serialization tests

Required tests:

1. `test_policy_events_serialize_in_router_state`
2. `test_switch_history_serializes_reentry_fields`
3. `test_record_contains_pre_and_post_policy_routing`

---

## 4. Suggested implementation order

1. `router/models.py`
2. `router/state.py`
3. `router/analyzer.py`
4. `router/runtime/state_updater.py`
5. `router/orchestration/transition_rules.py`
6. `router/orchestration/switch_orchestrator.py`
7. `router/runtime/session_runtime.py`
8. `router/orchestration/stop_policy.py`
9. `router/runtime/planner.py`
10. tests

Reason:
- data model first
- routing authority second
- reentry mechanics third
- stop policy after the new state and reentry semantics exist

---

## 5. Explicit non-goals for this patch

Do not do these in the same patch:

- widen the feature taxonomy
- add synthesis pressure or dominant tension fields
- redesign Builder semantics beyond the policy already chosen
- add a new construction regime
- rewrite regime prompts
- retune validation semantics unrelated to stop-policy artifact alignment

---

## 6. Definition of done

This spec is complete when all of the following are true:

1. analyzer-led routing can no longer be hard-overridden by absence-only lexical support for operator, adversarial, or builder
2. Builder is no longer blocked by a sparse recurrence threshold in stop policy
3. same-stage retry and prior-stage reentry are governed by qualified justification, not blanket bans
4. run records expose policy actions, pre/post-policy stage choice, and reentry justification state
5. endpoint completion cannot fire on an artifact/stage mismatch
6. tests cover the new policy behavior
