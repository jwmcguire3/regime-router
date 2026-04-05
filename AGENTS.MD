# AGENTS.md

## Purpose

This repository is a regime-routing runtime. It plans a starting regime, executes stage-specific artifacts, validates outputs, records handoffs, and may switch or stop based on orchestration logic.

The current workstream is not generic feature work. It is a targeted control-layer improvement program focused on:

1. truthful routing
2. fallback and recovery behavior
3. artifact-aware completion
4. stronger handoff propagation
5. only then broader feature extraction

The active task set includes fallback-to-exploration tests and implementation, analyzer-led routing diagnostics and replacement of placeholder routing, artifact-aware completion tests and implementation, invalid-output recovery tests and implementation, and handoff-strength tests and implementation, as outlined in the user-provided task plan :contentReference[oaicite:0]{index=0}

---

## Core architectural intent

The intended model is:

- analyzer identifies the bottleneck
- analyzer/planner chooses primary regime and runner-up
- likely endpoint regime is meaningful
- runtime mostly follows deterministic forward progression when the path remains valid
- fallback to Exploration is the explicit escape path when the active frame becomes invalid
- downstream stages should carry forward accumulated handoff state
- completion should be judged against the requested final deliverable, not merely against whether an intermediate regime artifact validated

Do not silently replace this with a different architecture.

---

## Primary goals of this work

### 1. Truthful routing
Top-level routing must stop using placeholder defaults and produce a real `RoutingDecision` grounded in planning/analyzer inputs.

### 2. Explicit fallback semantics
Fallback to Exploration is not generic failure noise. It is a first-class recovery path when:
- the current frame collapses
- assumptions fail
- constraints materially change
- all live regimes are solving the wrong problem

### 3. Artifact-aware completion
A valid intermediate regime artifact is not automatically task completion.
If the prompt explicitly requires a worksheet, framework document, memo, spec, or other named final artifact, the runtime must not stop early just because a `candidate_frame_set`, `dominant_frame`, or `decision_packet` validated.

### 4. Invalid-output recovery
If a stage returns empty, invalid, or unrecoverable output, the runtime must fail truthfully and recover coherently where appropriate. It must not mark false progress.

### 5. Stronger handoff propagation
Later stages should not behave like lightly augmented reruns of the original task. Stable elements, broken elements, do-not-relitigate elements, and minimum useful artifact pressure should materially shape downstream execution.

### 6. Narrow, high-value feature extraction only after the above
Feature extraction should improve start-position quality and fallback sensitivity, but should not become a keyword swamp or compensate for missing control logic.

---

## Task ordering

Follow this order unless the current task explicitly says otherwise:

1. fallback tests
2. fallback implementation
3. planner-path diagnostics
4. replace placeholder routing
5. artifact-completion tests
6. artifact-aware completion implementation
7. invalid-output recovery tests
8. invalid-output recovery implementation
9. handoff propagation tests
10. handoff propagation implementation
11. only then broader feature extraction

Do not jump ahead to extraction work if control-layer failures remain unresolved.

---

## What must not happen

Do not:
- fix control failures only by rewriting prompts
- paper over missing logic with richer prose
- mark tasks complete because a regime artifact validated if the requested final artifact is still missing
- silently degrade to placeholder routing behavior
- invent fake certainty, fake switch triggers, or fake reasons
- broaden scope into a repo-wide redesign
- turn feature extraction into an overgrown ontology or keyword pile
- weaken loop-prevention protections casually
- create same-stage churn or oscillation
- blur the distinction between:
  - planned switch condition
  - observed runtime cause
  - fallback cause
  - invalid-output failure
  - normal forward progression

---

## Testing discipline

Every behavior change must come with tests in the same task unless the path is impossible to test directly.

For each task:

1. inspect the relevant existing tests first
2. add or update focused tests for the intended behavior
3. make the implementation change
4. run the smallest relevant test subset
5. run adjacent tests that could regress
6. report exactly what was run and whether it passed

Do not defer all testing to a final cleanup pass.

### Minimum validation rule
For any touched behavior:
- run focused tests first
- then directly related modules
- run the full suite only after focused runs pass or when the task explicitly requires it

### Reporting rule
When finishing a task, report:
- files changed
- tests added or updated
- commands run
- pass/fail result
- any remaining untested risk

---

## Scope discipline by task type

### Tests-first tasks
For tasks that introduce regression tests:
- make the missing behavior explicit
- prefer deterministic fixtures/fakes over real model calls
- do not rewrite production behavior except for minimal harness support
- keep tests narrow and named around the exact semantic gap

### Control-layer implementation tasks
For tasks implementing fallback, completion, recovery, routing, or handoff:
- prefer explicit control logic over prose heuristics
- keep changes small and local
- preserve current compatibility expectations unless a change is clearly necessary
- record stop/switch/failure causes truthfully and specifically

### Routing tasks
When working on routing:
- route by bottleneck, not by domain
- keep runner-up regime meaningful
- keep endpoint logic meaningful
- degrade honestly and deterministically when analyzer inputs are thin
- never fabricate confidence or rationale

### Completion tasks
When working on completion/stopping:
- treat explicit final deliverables as real completion pressure
- do not count intermediate artifacts as completion unless they actually satisfy the requested deliverable semantics
- preserve truthful failure behavior when the requested deliverable cannot safely be produced

### Recovery tasks
When working on invalid-output recovery:
- empty output, invalid JSON after repair, and semantically unusable artifacts are first-class runtime failure states
- failed validation is not progress
- failed repair is not progress
- recovery behavior should be explicit, inspectable, and loop-safe

### Handoff tasks
When working on handoff propagation:
- preserve the original task in downstream prompts
- make stable elements materially carry forward
- reopen broken elements explicitly
- make do-not-relitigate fields meaningful
- strengthen minimum useful artifact pressure without turning the system into handoff-only execution
- avoid bloated or brittle prompt payloads

### Feature extraction tasks
When working on feature extraction:
- add only the next few highest-value signals
- improve start-position quality and fallback sensitivity
- do not compensate for missing routing/control logic by piling on markers
- do not create phrase-bound brittle behavior if a more structural signal is possible

---

## Truthfulness requirements

Always preserve the distinction between what the code actually does and what it is intended to do.

Valid outputs are:
- a direct factual answer
- a precise failure
- a precise description of what remains missing

Invalid outputs include:
- fake confidence
- fake completion
- fake recovery
- fake “why this wins now”
- fake switch semantics
- fake endpoint pressure

If analyzer inputs are thin, say so in code behavior and tests. Do not invent coverage through explanation.

---

## Prompt-related guidance

This repo uses regime-specific prompts, but prompts are not the primary fix surface for the current task set.

Prompt edits are allowed only when they:
- support truthful handoff propagation
- support clearer downstream discipline
- reinforce control semantics already implemented elsewhere

Do not use prompt wording as the sole implementation of:
- fallback
- stopping
- recovery
- endpoint selection
- artifact completion

---

## Runtime-state and bookkeeping expectations

State, handoff, switch history, and stop reasons must remain inspectable.

When changing orchestration or recovery logic:
- keep switch recording specific
- preserve truthful stop reasons
- avoid ambiguous catch-all failure labels when a more precise cause exists
- keep handoff/state integrity improvements intact

If a new normalized vocabulary is needed for fallback or failure causes:
- keep it small
- keep it explicit
- do not invent many strings

---

## Definition of done for this workstream

This workstream is only done when all of the following are true:

1. placeholder top-level routing is gone
2. fallback-to-exploration semantics exist and are tested
3. artifact-aware completion prevents premature stopping on intermediate artifacts
4. invalid-output recovery exists and is tested
5. handoff propagation more strongly shapes downstream stages
6. feature extraction improvements remain narrow and justified
7. focused tests pass for each change
8. directly related existing tests still pass
9. final behavior is more truthful, not merely more persuasive

---

## Preferred engineering style

- surgical changes
- deterministic tests
- explicit control logic
- small helper vocabularies if needed
- minimal new state
- no broad refactors unless the task truly requires them
- preserve compatibility where reasonable
- keep behavior inspectable in saved runs

---

## If you are unsure

When uncertain about a change:
- prefer the narrower control-layer fix
- prefer adding a regression test first
- prefer a truthful failure over a plausible-looking fake success
- prefer preserving inspectability over hiding complexity

The bar is not “looks smart.”
The bar is “routes truthfully, recovers honestly, and finishes the right artifact.”
