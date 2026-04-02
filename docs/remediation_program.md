# Cognitive Router Remediation Program

## Purpose

This document consolidates:

1. The full scope of weaknesses identified in `docs/system_assessment.md`.
2. The five-task execution sequence already defined for implementation.
3. A phased plan with explicit reasoning requirements per task.

The goal is to move from assessment to controlled delivery without blurring diagnostics, control-layer fixes, and front-door extraction improvements.

---

## Consolidated problem scope

### Core control-layer gaps (from the assessment)

- Transition pathways are too rigid for reliable `Any -> Exploration` recovery.
- Collapse/failure detection is too text-fragile.
- Re-entry protection is currently too blunt for valid corrective loops.
- Switching still depends too much on free-text semantics.
- Orchestration lacks explicit execution-time containment budgets.
- Escalation influence is narrower than intended.

### Secondary routing/planner gaps

- Top-level route entry still needs deterministic analyzer/planner grounding.
- Planner behavior (start regime, runner-up, endpoint pressure, progression) needs diagnostics locked before change.
- Feature extraction remains narrower than needed for mixed-tension tasks and robust fallback sensitivity.

### Additional non-covered areas (Phase 4 hardening backlog)

- Provider reliability and failover behavior.
- Cost/latency operating envelopes.
- Security and abuse boundaries.
- Observability and route-quality instrumentation depth.
- Data retention and memory-governance controls.
- Evaluation methodology and benchmark discipline.

---

## Program principles

- **Diagnostics precede behavior changes** for high-risk routing semantics.
- **One architectural concern per task** to avoid cross-contamination.
- **Deterministic control logic over prose inference** where routing decisions are made.
- **Smallest coherent change** that satisfies tests and preserves compatibility.
- **Stop/go gates between phases** to prevent stacking uncertainty.

---

## Phase plan

## Phase 0 — Baseline and test harness confidence

### Task P0-T1: Baseline verification and inventory

**Objective**
Confirm current behavior baseline and identify tests/modules to extend.

**Deliverables**
- Baseline run record for relevant routing/orchestration/planner suites.
- Inventory of existing tests to extend in Phases 1 and 2.

**What must be reasoned about**
- Which tests are true behavior contracts vs incidental implementation checks.
- How to isolate fallback/planner assertions from unrelated variability.
- Minimal fixture strategy to avoid real model calls in new diagnostics.

**Exit criteria**
- Baseline green run captured.
- Candidate test modules and fixture reuse plan documented.

---

## Phase 1 — Fallback semantics safety valve

### Task P1-T1 (Diagnostic): Regression tests for `Any -> Exploration`

**Objective**
Write failing/targeted tests for required fallback semantics from non-Exploration stages.

**Required cases**
- Assumptions fail.
- Frame collapses.
- Active path is solving the wrong problem.

**What must be reasoned about**
- Exact fallback-trigger contract: what counts as “collapse” vs normal failure.
- Precedence when a forward switch and fallback are both plausible.
- Assertion granularity: verify chosen stage and recorded cause, not only final stop reason.

**Exit criteria**
- Tests explicitly encode fallback expectations and fail on current missing behavior where applicable.

### Task P1-T2 (Implementation): First-class fallback behavior

**Objective**
Implement explicit fallback-to-Exploration behavior in orchestration/transition logic.

**What must be reasoned about**
- Trigger vocabulary normalization (small finite set of causes).
- Separation of planned switch condition vs observed runtime cause.
- Anti-churn controls (avoid oscillation and same-stage loops).
- Compatibility with existing loop-prevention and stop-policy behavior.

**Exit criteria**
- P1-T1 diagnostics pass.
- Switch history truthfully records fallback reason and stage.
- No regression in related orchestration suites.

---

## Phase 2 — Planner-path determinism and real entry routing

### Task P2-T1 (Diagnostic): Planner-path routing behavior tests

**Objective**
Lock intended analyzer-led deterministic routing behavior in tests before replacing placeholder top-level routing.

**Required assertions**
- Start regime comes from planning/analyzer signal, not canned defaults.
- Endpoint pressure shapes progression/stopping.
- Runtime stays on valid planned path unless justified fallback occurs.

**What must be reasoned about**
- Which fields are contract-critical (`bottleneck`, `primary`, `runner_up`, endpoint confidence/intent).
- What constitutes justified drift vs misrouting.
- How to represent deterministic progression in tests without overfitting implementation details.

**Exit criteria**
- Diagnostics express intended planning model and expose current placeholder limitations.

### Task P2-T2 (Implementation): Replace placeholder `Router.route()`

**Objective**
Replace canned top-level routing with analyzer/planner-driven deterministic decision output.

**What must be reasoned about**
- Fallback behavior when analyzer output is absent/thin (honest deterministic degradation).
- Truthfulness of routing rationale and switch-trigger fields.
- Preservation of compatibility for existing `RoutingDecision` consumers.
- Runner-up and endpoint semantics as meaningful outputs, not placeholders.

**Exit criteria**
- P2-T1 diagnostics pass.
- Placeholder defaults removed from top-level route path.
- Routing/planner integration tests remain stable.

---

## Phase 3 — Feature extraction broadening (targeted, not expansive)

### Task P3-T1 (Implementation + tests): Expand extraction for start quality and fallback sensitivity

**Objective**
Broaden feature extraction only enough to improve regime start-position quality and fallback sensitivity.

**Target improvements**
- Epistemic vs Operator separation.
- Synthesis vs Epistemic separation.
- Builder eligibility (recurrence/productization evidence).
- Exploration recovery sensitivity when frame validity degrades.

**What must be reasoned about**
- Signal specificity: high-value families that do not overfire.
- Marker composability: avoid keyword pile and maintain legibility.
- Calibration tradeoffs: precision vs recall by regime boundary.
- Regression risk: where broadened features could distort existing routing tests.

**Exit criteria**
- New/updated feature tests pass.
- Routing correctness tests show improved mixed-tension handling.
- No uncontrolled expansion of signal taxonomy.

---

## Phase 4 — Post-core hardening backlog (optional but recommended)

### Task P4-T1: Reliability and containment hardening

**What must be reasoned about**
- Timeout/retry/failover interaction with orchestration stop reasons.
- Provider-specific degradation modes and deterministic handling.

### Task P4-T2: Observability and evaluation depth

**What must be reasoned about**
- Metric contracts for route quality and switch correctness.
- Benchmark design for longitudinal regression detection.

### Task P4-T3: Governance and data controls

**What must be reasoned about**
- Memory retention boundaries, audit requirements, and privacy constraints.
- Human override points for high-stakes paths.

---

## Execution order and gating

1. **Phase 0** (baseline confidence)
2. **Phase 1** (fallback diagnostics then implementation)
3. **Phase 2** (planner diagnostics then route implementation)
4. **Phase 3** (feature extraction broadening)
5. **Phase 4** (hardening backlog)

**Hard gate rule**
Do not start a later phase until the current phase’s diagnostics/exit criteria are met.

---

## Canonical 5-task mapping (requested sequence)

- **Task 1** → P1-T1 (diagnostic fallback semantics)
- **Task 2** → P1-T2 (implement fallback semantics)
- **Task 3** → P2-T1 (diagnostic planner/start-end behavior)
- **Task 4** → P2-T2 (replace placeholder top-level routing)
- **Task 5** → P3-T1 (broaden feature extraction + tests)

This preserves the intended order: **escape path first, real planner path second, better front-door sensitivity third**.
