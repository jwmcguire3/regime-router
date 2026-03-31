# AGENTS.md

## Project purpose

This repository is a cognitive routing runtime for composable behavioral control over LLMs.

The system is being refactored to preserve future extension paths for:
- grammar-driven regime composition
- workspace board
- parallel execution
- reconciliation protocol
- advanced orchestration patterns

Current refactor work is explicitly **not** implementing those features yet. The goal is to cut the right seams now so those later additions are easy and low-risk.

---

## General working rules

1. Prefer **minimal, extraction-focused diffs** over opportunistic cleanup.
2. Preserve **current public behavior** unless the task explicitly requests a behavior change.
3. Keep **existing CLI behavior and command surface stable** unless the task explicitly requests CLI work.
4. Avoid broad renames unless they are directly required by the task.
5. Do not silently redesign architecture beyond the requested seam extraction.
6. When in doubt, preserve behavior over elegance.

---

## Architectural invariants

These must be preserved during refactors:

### 1. External control over model behavior
The LLM is the constrained system. The control/evaluation system is external.
Do not move evaluation/compliance judgment into the model itself.

### 2. Typed, inspectable control
Composition, routing, validation, repair, switching, and state updates must remain inspectable.
Do not hide important logic in opaque helper flows.

### 3. RouterState is control-state, not artifact-store
Keep `RouterState` focused on control/orchestration truth.
Do **not** deepen artifact-content conflation.
Do **not** stuff more work-product structure into RouterState just because it is convenient.

Future `WorkspaceBoard` support should be easy to add as a nested/adjacent concern later.

### 4. Validation stays pure and per-output
Validation should remain:
- one output in
- one validation result out

Do not make validation depend on runtime globals, orchestration topology, board state, or hidden mutable context.

### 5. Prompt construction should remain role-extensible
Prompt construction should stay cleanly separable by execution role.
Preserve the ability to add later:
- normal regime prompts
- repair prompts
- consolidator prompts
- verifier prompts
- challenger prompts

Do not hard-specialize prompt building around only the current single execution flow.

### 6. Provider abstraction must not leak
Provider/client abstraction belongs in runtime/execution transport boundaries.
Do not leak provider-specific naming or logic into:
- routing
- control/orchestration policy
- RouterState
- validation

### 7. Execution and state mutation must stay separable
Execution should be an independent unit.
State update / handoff / restore logic should remain separate from raw execution.
Future `execute_many(...)` should feel like an extension, not a rewrite.

### 8. Orchestration policy should not assume one forever-topology
When extracting control/orchestration logic:
- avoid names that imply only one linear progression forever
- preserve boundedness and inspectability
- keep logic topology-agnostic where feasible

---

## Refactor priorities

These are the future-shaped moves worth paying for now:

1. provider-neutral `ModelClient` seam in runtime
2. composer isolated and replaceable
3. routing split into explicit policy modules
4. misrouting / escalation / switching isolated into orchestration modules
5. runtime split into planner / executor / repair / updater
6. thin facade objects instead of giant mixed-policy files

Anything beyond that should be treated skeptically unless the task explicitly requires it.

---

## Explicit "do not do now" list

Unless the task explicitly asks for them, do **not** implement:

- OpenAI provider support
- provider switching
- provider-aware CLI/settings
- response normalization redesign
- grammar-driven composer
- failure-cost-driven composition
- workspace board
- board contribution tracking
- parallel execution
- reconciliation protocol
- consolidator flows
- verifier flows
- builder/challenger orchestration
- circle-of-experts orchestration
- topology registries
- advanced orchestration engines
- multi-artifact validation
- board-aware routing
- reconciliation-aware routing

Do not add placeholder implementations for these just to “prepare” the codebase.

---

## File-specific guidance

### `state.py`
- Keep RouterState focused on control/orchestration truth.
- Leave room for a future optional `workspace_board`.
- Keep serialization easy to extend.
- Keep handoff as a projection, not source of truth.

### `validation.py`
- Preserve pure per-output validation.
- Keep it reusable for direct, regime, verifier, challenger, and future consolidator outputs.

### `prompts.py`
- Preserve clean separation between prompt roles.
- Organize so future prompt roles can be added without rewriting current logic.

### `models.py`
- Do not turn this into a dumping ground for future architecture nouns.
- Add only types needed by the current task.
- Keep execution, state, and future board concerns conceptually separable.

### `routing/*`
- Keep routing evidence explicit and inspectable.
- Keep routing independent from provider/runtime details.
- Do not absorb orchestration-pattern logic into routing modules.

### `orchestration/*`
- Keep stage rules and transition policy pure where possible.
- Avoid baking in assumptions that only one linear progression exists forever.

### `runtime/*` and `execution/*`
- Runtime facade should stay thin.
- Execution units should stay independently testable.
- Keep provider-neutral naming in runtime internals.
- Keep collected execution results easy to validate independently.

---

## How to approach a refactor task

Before editing:
1. Inspect the smallest dependency surface needed.
2. Identify circular-import risk.
3. Plan the narrowest extraction that preserves behavior.

While editing:
1. Move logic first.
2. Change imports second.
3. Rename only when required by the task.
4. Avoid unrelated cleanup.

After editing:
1. Re-check imports.
2. Re-check preserved public method signatures.
3. Re-check behavior assumptions named in the task.
4. Provide a short manual test checklist.

---

## Output expectations for refactor tasks

When finishing a task, report:
1. files changed
2. what seam/module split was introduced
3. what behavior was intentionally kept unchanged
4. deferred cleanup or follow-up work
5. a short manual test checklist

Keep the report concise and concrete.

---

## Review heuristic

For any refactor, ask these three questions:

1. Did this make later WorkspaceBoard insertion easier or harder?
2. Did this make later parallel execution easier or harder?
3. Did this make later reconciliation easier or harder?

If any answer is “harder,” the refactor is probably cutting the wrong seam.

## Test guidance

- Preserve existing tests unless the task requires updating them.
- Prefer reusing current tests for behavior-preservation refactors.
- Add or update targeted tests when a task introduces a new seam, wrapper, extracted module boundary, or explicit contract that is not already directly covered.
- Do not add speculative tests for future architecture that is not implemented yet.
- For major shared-boundary refactors (runtime, routing coordination, orchestration policy), run the full test suite before marking the task complete.
