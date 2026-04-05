# Control Surface vNext Alignment Report (April 5, 2026)

## Scope

Compared the current repository implementation against:

- `docs/CONTROL_SURFACE_POLICY_vNEXT.md`
- `docs/CONTROL_SURFACE_CODE_CHANGE_SPEC_vNEXT.md`

I also verified the local copies of those two docs are byte-identical to `main` on GitHub.

## Verdict

The repo is **mostly aligned, but not fully aligned**.

- Major architectural changes from vNext are present (softened analyzer demotion, qualified reentry model, expanded transition pathways, stop-policy improvements).
- A few implementation deltas remain versus the exact code-change spec.

## Aligned areas

1. **Routing policy datamodel additions are present**
   - `ControlAuthority`, `ReentryJustification`, `ReentryDecision`, `PolicyEvent` exist.
   - `RoutingDecision` now carries pre-policy regimes plus policy warnings/actions.

2. **State persistence surface is updated**
   - `SwitchDecisionRecord` includes reentry and defect metadata.
   - `RouterState` includes policy event storage and last reentry/context deltas.
   - JSON restoration supports these added fields.

3. **Analyzer no longer hard-demotes operator/builder/adversarial by missing lexical support alone**
   - `_apply_routing_policy(...)` issues warnings and limited runner-up softening instead of hard overrides.
   - Endpoint softening from builder to operator is implemented when support is weak.

4. **Transition system is materially updated for qualified reentry**
   - `DEFAULT_FORWARD_PATHWAYS` and `CONDITIONAL_REENTRY_PATHWAYS` exist.
   - Reentry helper functions and justification building are implemented.

5. **Session runtime now uses explicit reentry evaluation**
   - `_evaluate_reentry(...)` and `_is_ping_pong(...)` are present.
   - Same-stage/prior-stage moves are qualification-gated instead of blanket-banned.

6. **Stop policy no longer contains hard builder recurrence threshold blocking**
   - No `BUILDER_RECURRENCE_THRESHOLD` constant.
   - No `_builder_blocked(...)` helper.

## Remaining deltas vs code-change spec

1. **Operator→Builder still partially keys on a recurrence threshold signal**
   - `transition_rules._looks_like_reusable_structure(...)` still allows builder based on `state.recurrence_potential >= 2.0` when misrouting recommends builder.
   - The spec says stop using this threshold as sole transition trigger and prefer semantic evidence.

2. **Policy event observability model is defined but not wired into runtime rule execution**
   - `RouterState.record_policy_event(...)` exists, but no live call sites currently emit `PolicyEvent` records during analyzer/transition/stop decisions.

3. **Legacy/dead branch remains in session runtime**
   - `SessionRuntime.run_orchestration_loop(...)` still checks for `reason.startswith("Builder blocked:")`, but current stop policy no longer emits that branch.
   - Not a functional blocker, but it is drift from the cleaned-up spec intent.

## Validation checks run

- `pytest -q tests/test_analyzer_policy.py tests/test_reentry_policy.py tests/test_stop_policy_control_surface.py tests/test_switch_orchestrator.py -q`
- Local-vs-remote hash equality check for both vNext docs.

