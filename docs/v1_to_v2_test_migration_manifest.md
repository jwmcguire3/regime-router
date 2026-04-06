# v1→v2 Test Migration Manifest (Plan-First, Pre-Code-Change)

## Intent and constraints

This manifest is a **pre-implementation** inventory of test impact for the routing refactor.

- No production code was modified.
- No tests were modified.
- Classifications below are based on repo discovery commands and current test names.
- Any uncertainty is explicitly labeled.

## Discovery log (authoritative commands used)

### 1) Locate test-command sources

```bash
rg --files -g 'pyproject.toml' -g 'tox.ini' -g 'Makefile' -g '.github/workflows/*' -g 'README*'
rg -n "pytest|tox|make test|unittest" AGENTS.MD README* pyproject.toml tox.ini Makefile .github/workflows 2>/dev/null
```

Result summary:
- No `pyproject.toml`, `tox.ini`, `Makefile`, top-level `README`, or workflow file was found.
- The only in-repo authoritative test guidance found is in `AGENTS.MD`.

### 2) Symbol impact sweep (tests + production)

```bash
rg -n "TaskClassifier|TaskClassification|classify\(|infer_risk_profile|risk_inference|RoutingFeatures|build_router_state|update_router_state_from_execution|\.plan\(|\.evaluate\(|\.route\(|\.analyze\(|\.propose_route\(|run_orchestration_loop|plan_direct|_plan_direct|execute_direct_task" router tests
```

### 3) Exact test-name enumeration in impacted test files

```bash
pytest --collect-only -q tests
for f in tests/test_classifier.py tests/test_classifier_demotion_runtime.py tests/test_routing_analyzer_feature_regression.py tests/test_runtime_planner_determinism.py tests/test_router_integration.py tests/test_router_state.py tests/test_router_state_integration.py tests/test_regression_routing_reform.py tests/test_switch_orchestrator.py tests/test_stop_policy.py tests/test_fallback_regressions.py tests/test_v2_stage_continuity.py tests/test_orchestrator_harness.py tests/test_orchestration_fixes.py tests/test_misrouting_harness.py tests/test_misrouting_detector.py tests/test_transition_rules_completion_escalation.py tests/test_transition_rules_escalation_preferred_forward.py tests/test_storage_serialization.py tests/test_proposer.py tests/test_analyzer_policy.py tests/test_escalation_harness.py tests/test_reentry_policy.py ; do rg -n "^def test_" "$f"; done
```

## Canonical test command for this repo

Given the absence of CI/Makefile/pyproject/tox/README test instructions, this manifest treats the AGENTS guidance as canonical:

- Broad suite command: `pytest tests/ -x -v`
- Use focused subsets first (per AGENTS examples).

## Refactor milestones and linked test migration work

## Milestone A — analyzer contract changes

Expected code-surface change:
- Analyzer output becomes authoritative for structural signals/risk tags in router path.

Primary impacted tests:
- `tests/test_proposer.py`
  - `test_propose_route_returns_exploration_fallback_when_analyzer_fails`
  - `test_propose_route_ranks_stages_from_analyzer_output`
  - `test_analyzer_summary_is_populated`
  - `test_classifier_signal_in_analyzer_prompt` **(likely semantic/assertion rewrite; classifier demotion)**
  - `test_endpoint_in_routing_decision`
  - `test_endpoint_cannot_precede_primary`
- `tests/test_analyzer_policy.py`
  - all policy/endpoint tests, especially confidence + support demotion behaviors.
- `tests/test_routing_analyzer_feature_regression.py`
  - analyzer-led tests likely survive with assertion updates where risk/support sources move.

Migration action:
- Mostly **migrate assertions because semantics changed**.
- Potential **call-site-only** changes for analyzer payload construction.

Validation subset for packet:
```bash
pytest tests/test_proposer.py tests/test_analyzer_policy.py tests/test_routing_analyzer_feature_regression.py -x -v
```

## Milestone B — RouterState changes

Expected code-surface change:
- RouterState carries structural/risk/analyzer-derived fields directly; updater logic changes.

Primary impacted tests:
- `tests/test_router_state.py::test_plan_populates_router_state_core_fields`
- `tests/test_router_state_integration.py::test_handoff_fields_are_router_state_backed_after_plan`
- `tests/test_v2_stage_continuity.py`
  - `test_state_dominant_frame_updates_from_synthesis_central_claim`
  - `test_state_dominant_frame_updates_from_operator_decision`
  - `test_state_dominant_frame_updates_from_builder_reusable_pattern`
  - plus handoff continuity tests may require field-source updates.
- `tests/test_fallback_regressions.py::test_collapse_assumption_guard_is_noop_after_state_build`
- `tests/test_reentry_policy.py::test_last_state_and_contract_delta_storage`
- `tests/test_storage_serialization.py`
  - `test_router_state_round_trip_preserves_planned_and_observed_switch_fields`
  - `test_policy_events_serialize_in_router_state`
  - `test_switch_history_serializes_reentry_fields`
  - `test_record_contains_pre_and_post_policy_routing`

Migration action:
- Mix of **call-site migration** and **assertion migration**.

Validation subset for packet:
```bash
pytest tests/test_router_state.py tests/test_router_state_integration.py tests/test_v2_stage_continuity.py -k "state or handoff" -x -v
pytest tests/test_storage_serialization.py tests/test_reentry_policy.py tests/test_fallback_regressions.py -k "state or router_state" -x -v
```

## Milestone C — switch routing changes (state-driven)

Expected code-surface change:
- Switch decisions are state-driven (`route_switch` path) instead of feature/classifier coupling.

Primary impacted tests:
- `tests/test_switch_orchestrator.py` (all switch transition and bounded-loop tests)
- `tests/test_orchestration_fixes.py`
- `tests/test_orchestrator_harness.py`
- `tests/test_misrouting_harness.py`
- `tests/test_misrouting_detector.py`
- `tests/test_transition_rules_completion_escalation.py`
- `tests/test_transition_rules_escalation_preferred_forward.py`
- `tests/test_stop_policy.py::test_max_switches_still_hard_ceiling`

Migration action:
- Mostly **integration setup update** (mocked sequence/state scaffolding).
- Some **assertion migration** where switch cause/fields are renamed or re-sourced.

Validation subset for packet:
```bash
pytest tests/test_switch_orchestrator.py tests/test_orchestrator_harness.py tests/test_orchestration_fixes.py -x -v
pytest tests/test_misrouting_detector.py tests/test_misrouting_harness.py tests/test_transition_rules_completion_escalation.py tests/test_transition_rules_escalation_preferred_forward.py -x -v
```

## Milestone D — escalation/orchestration changes

Expected code-surface change:
- Remove orchestration-path dependency on `RoutingFeatures`; rebalance escalation inputs.

Primary impacted tests:
- `tests/test_escalation_harness.py` (all tests)
- `tests/test_stop_policy.py` (builder gating + endpoint checks)
- `tests/test_fallback_regressions.py`
- `tests/test_regression_routing_reform.py`
  - `test_stop_at_operator_integration`
  - `test_builder_only_when_justified_integration`
  - `test_builder_entered_when_justified`
  - `test_no_independent_stage_reruns`

Migration action:
- **migrate assertions because semantics changed**; some **integration setup update**.

Validation subset for packet:
```bash
pytest tests/test_escalation_harness.py tests/test_stop_policy.py tests/test_fallback_regressions.py tests/test_regression_routing_reform.py -x -v
```

## Milestone E — runtime/planner simplification

Expected code-surface change:
- Remove fastpath gate and possibly direct-planning path (`plan_direct`, `_plan_direct`, `execute_direct_task`) from critical routing.

Primary impacted tests:
- `tests/test_classifier_demotion_runtime.py`
  - `test_direct_fastpath_requires_all_three` **(deletion candidate)**
  - `test_direct_pattern_task_with_structural_tension_not_fastpathed_even_when_classifier_direct` **(deletion or full semantic rewrite)**
  - `test_analyzer_decision_consumes_supplied_analysis` (likely survives with call-site updates)
- `tests/test_routing_analyzer_feature_regression.py::test_fastpath_blocks_direct_when_fragility_pressure_is_positive` **(deletion candidate)**
- `tests/test_runtime_planner_determinism.py` (planner API/signature updates)
- `tests/test_router_integration.py::test_outcome_routes_match_expected_stage` (if planning semantics changed)

Migration action:
- **delete removed-behavior tests** + **call-site migration** for planner invocation.

Validation subset for packet:
```bash
pytest tests/test_classifier_demotion_runtime.py tests/test_routing_analyzer_feature_regression.py tests/test_runtime_planner_determinism.py -x -v
```

## Milestone F — dead-code deletion

Expected code-surface change:
- Delete classifier/risk-inference/fastpath/direct-path code that is no longer in critical path.

Primary impacted tests (deletion first-pass):
- `tests/test_classifier.py` (all tests; pure classifier unit tests)
- `tests/test_classifier_demotion_runtime.py::test_direct_fastpath_requires_all_three`
- `tests/test_classifier_demotion_runtime.py::test_direct_pattern_task_with_structural_tension_not_fastpathed_even_when_classifier_direct`
- `tests/test_routing_analyzer_feature_regression.py::test_fastpath_blocks_direct_when_fragility_pressure_is_positive`

Potentially impacted-by-signature (not immediate delete):
- `tests/test_router_integration.py`
  - `test_synthesis_prompt_contains_anchor_contract_and_role_guidance`
  - `test_validator_accepts_grounded_synthesis_artifact`
  - `test_validator_rejects_polished_but_generic_synthesis_artifact`
  - `test_validator_rejects_pressure_points_as_execution_risks`
  (these currently call `infer_risk_profile`; likely call-site migration unless risk behavior is removed from test objective).

Validation subset for packet:
```bash
pytest tests/test_classifier_demotion_runtime.py tests/test_routing_analyzer_feature_regression.py tests/test_router_integration.py -x -v
```

## Action-based classification manifest

## Category 1 — delete with removed code

### Safe delete candidates (high confidence)
- `tests/test_classifier.py`
  - `test_classifies_write_breakout_game_code_as_direct`
  - `test_classifies_build_rest_api_as_direct`
  - `test_classifies_draft_q3_email_as_direct`
  - `test_classifies_fragments_spine_statement_as_regime`
  - `test_classifies_stress_test_frame_as_regime`
  - `test_classifies_pricing_decision_question_as_regime`
  - `test_classifies_explore_interpretations_as_regime`
  - `test_classifies_fix_bug_as_direct`
  - `test_classifies_architecture_fragility_why_question_as_regime`
- `tests/test_classifier_demotion_runtime.py::test_direct_fastpath_requires_all_three`
- `tests/test_routing_analyzer_feature_regression.py::test_fastpath_blocks_direct_when_fragility_pressure_is_positive`

### Delete candidates (explicit uncertainty)
- `tests/test_classifier_demotion_runtime.py::test_direct_pattern_task_with_structural_tension_not_fastpathed_even_when_classifier_direct`
  - **Uncertain**: delete if direct path removed; otherwise rewrite to new state-driven non-direct assertion.

## Category 2 — migrate call site only

- `tests/test_runtime_planner_determinism.py`
  - `test_planner_deterministic`
  - `test_planner_no_model_calls`
  - `test_planner_threads_analyzer_evidence_quality_into_router_state`
- `tests/test_router_state.py::test_plan_populates_router_state_core_fields`
- `tests/test_router_state_integration.py::test_handoff_fields_are_router_state_backed_after_plan`
- `tests/test_regression_routing_reform.py::test_every_task_reaches_analyzer`

Notes:
- These are expected to retain core intent while adapting planner/runtime signatures and fixture construction.

## Category 3 — migrate assertions (semantic changes)

- `tests/test_proposer.py` (all tests, especially classifier/risk signal-in-prompt expectations)
- `tests/test_analyzer_policy.py` (all tests)
- `tests/test_escalation_harness.py` (all tests)
- `tests/test_stop_policy.py`
  - builder/endpoint/switch cap tests likely need updated pressure source assertions.
- `tests/test_fallback_regressions.py`
  - state-collapse and truthful-failure bookkeeping tied to state-update semantics.
- `tests/test_v2_stage_continuity.py`
  - state-derived handoff/knowns/assumptions/contradictions tests may need source-field assertion updates.

## Category 4 — pass unchanged (initial expectation)

Likely unchanged or minimally touched:
- `tests/test_misrouting_detector.py` (detector logic is mostly regime-output semantic)
- `tests/test_transition_rules_completion_escalation.py` (if transition rules are preserved)
- `tests/test_transition_rules_escalation_preferred_forward.py` (if thresholds/stage legality unchanged)

**Uncertain flag:** these remain sensitive to orchestration-field renames; treat as canary tests after each milestone.

## Category 5 — integration setup update (mock sequence / runtime wiring)

- `tests/test_switch_orchestrator.py` (mocked planner/executor sequencing)
- `tests/test_orchestrator_harness.py`
- `tests/test_orchestration_fixes.py`
- `tests/test_misrouting_harness.py`
- `tests/test_regression_routing_reform.py`
  - `test_handoff_continuity_across_switch`
  - `test_stop_at_operator_integration`
  - `test_builder_only_when_justified_integration`
  - `test_builder_entered_when_justified`
  - `test_no_independent_stage_reruns`
- `tests/test_storage_serialization.py` (if orchestration event payloads evolve)

## Category 6 — new coverage required

Required by target v2 behavior (not present as exact tests today):

1. `_update_pressures_from_execution` behavior
   - Add focused unit tests (recommended file: `tests/test_runtime_state_updater.py` or `tests/test_fallback_regressions.py` extension).
   - Cases:
     - pressure increments from explicit execution failures
     - pressure reset/decay rules across successful completion
     - interaction with recurrence and contradiction tracking

2. `route_switch` state-driven routing function
   - Add dedicated tests in `tests/test_switch_orchestrator.py` or new `tests/test_route_switch.py`.
   - Cases:
     - deterministic tie-breaking
     - bounded switching under repeated triggers
     - precedence between stop policy and switch routing in v2 ordering

3. Analyzer audit behavior
   - Add/extend tests in `tests/test_proposer.py` or new `tests/test_analyzer_audit.py`.
   - Cases:
     - analyzer structural signals and risk tags copied into RouterState/audit fields
     - analyzer contract fallback behavior when payload is partial
     - serialization/logging of analyzer evidence quality markers

## Production-symbol coverage check (required)

Each requested symbol has either affected tests or explicit no-direct-tests note:

- `TaskClassifier` / `TaskClassification` / `classify(`: direct tests in `tests/test_classifier.py`, `tests/test_classifier_demotion_runtime.py`, `tests/test_regression_routing_reform.py`.
- `infer_risk_profile` / `risk_inference`: direct references in `tests/test_router_integration.py`.
- `RoutingFeatures`: direct references across proposer/analyzer/planner/escalation tests (`tests/test_proposer.py`, `tests/test_analyzer_policy.py`, `tests/test_runtime_planner_determinism.py`, `tests/test_escalation_harness.py`, others).
- `build_router_state` / `update_router_state_from_execution`: direct references in `tests/test_fallback_regressions.py`, `tests/test_reentry_policy.py`, `tests/test_v2_stage_continuity.py`.
- `.plan(`: broad direct references in planner/runtime integration tests.
- `.evaluate(`: direct coverage via escalation and session runtime orchestration tests.
- `.route(`: direct references in router/CLI/integration tests.
- `.analyze(` / `.propose_route(`: direct references in proposer/routing correctness tests.
- `run_orchestration_loop`: direct references in stop/fallback tests.
- `plan_direct` / `_plan_direct`: **no direct test-name references found**; covered indirectly through planner/runtime tests touching fastpath behavior.
- `execute_direct_task`: **no direct test-name references found**; indirect only via runtime plan/run paths.

## Shared fixture work recommended before migrations

Create a common fixture module (or extend existing helpers) for `TaskAnalyzerOutput` v2 contract data to avoid repetitive per-test rewrites:

- canonical analyzer output fixture with:
  - structural signals
  - risk tags
  - endpoint proposal
  - evidence quality markers
- variant fixtures for partial/low-confidence/error payloads.

Suggested placement:
- extend `tests/test_proposer.py` local helpers **or** introduce `tests/conftest.py` fixture set if reused across 4+ files.

## Recommended order for test migration packets

1. **Delete dead tests first** (classifier + fastpath-specific).
2. **Analyzer contract packet** (proposer/analyzer policy/regression).
3. **Planner + RouterState call-site packet** (determinism/state tests).
4. **Switch routing packet** (switch/harness/misrouting integration setup).
5. **Escalation/orchestration packet** (stop policy + fallback + reform regressions).
6. **Serialization + continuity stabilization packet**.
7. **Add new required v2 tests** (`_update_pressures_from_execution`, `route_switch`, analyzer audit).

This order minimizes cascading fixture churn and keeps failing-surface size bounded.

## Later-packet minimal validation matrix

- Packet 1 (deletions):
  ```bash
  pytest tests/test_classifier_demotion_runtime.py tests/test_routing_analyzer_feature_regression.py -x -v
  ```
- Packet 2 (analyzer):
  ```bash
  pytest tests/test_proposer.py tests/test_analyzer_policy.py -x -v
  ```
- Packet 3 (planner/state):
  ```bash
  pytest tests/test_runtime_planner_determinism.py tests/test_router_state.py tests/test_router_state_integration.py -x -v
  ```
- Packet 4 (switch routing):
  ```bash
  pytest tests/test_switch_orchestrator.py tests/test_orchestrator_harness.py tests/test_orchestration_fixes.py -x -v
  ```
- Packet 5 (orchestration semantics):
  ```bash
  pytest tests/test_stop_policy.py tests/test_fallback_regressions.py tests/test_regression_routing_reform.py -x -v
  ```
- Packet 6 (continuity/serialization):
  ```bash
  pytest tests/test_v2_stage_continuity.py tests/test_storage_serialization.py -x -v
  ```

---

Status: complete pre-change migration manifest; ready for implementation packets.
