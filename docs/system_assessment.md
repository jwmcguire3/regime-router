# Cognitive Router System Assessment (April 2, 2026)

## Executive summary

The current implementation is **stable and heavily tested** (886 tests passing), but it still has several architectural weak points where behavior can drift from the original governance-layer intent.

## What looks strong

- Comprehensive automated test coverage exists and currently passes.
- Core routing stages and orchestration components are cleanly separated.
- State and handoff objects are typed and serializable.

## High-risk weak points

### 1) The transition graph cannot satisfy the spec's `Any -> Exploration` fallback in many real failures

The original spec says any regime should be able to switch back to Exploration when assumptions fail, constraints change, or the frame collapses.

In code, transitions are constrained by `ALLOWED_PATHWAYS`, and only a narrow `assumption_or_frame_collapse` branch can force Exploration. This creates a mismatch: many failures that should reopen exploration are blocked by stage-local pathways.

**Risk:** Router can continue in a locally valid but globally wrong basin, especially after late-stage invalidation.

**Fix:** Add a global override transition policy that permits `-> Exploration` when collapse criteria are met from any stage, not only when text-based collapse heuristics fire.

### 2) Collapse detection is overly brittle and tied to exact phrase patterns + state preconditions

`assumption_or_frame_collapse` relies on string matching (`"assumption"` + `"collapse"`, etc.) and additionally requires `state.assumptions` to be non-empty.

**Risk:** Real collapse signals with equivalent language (e.g., “core premise failed”, “invalidated foundation”) will not trigger. If assumptions were not populated earlier, fallback can be missed entirely.

**Fix:** Move collapse detection to structured fields in the output contract (enum or boolean flags), and treat missing assumptions as uncertainty rather than hard veto.

### 3) Loop prevention is too strict and may block legitimate recovery sequences

`SessionRuntime` blocks switching to **any previously executed stage**.

**Risk:** Valid iterative flows like `Synthesis -> Epistemic -> Synthesis` (after evidence correction) or `Operator -> Epistemic -> Operator` can be impossible, producing premature stop instead of controlled re-entry.

**Fix:** Replace hard ban with bounded re-entry (e.g., max visits per stage + monotonic progress checks).

### 4) Switch signaling depends on free-text completion/failure strings

`signal_from_output` extracts strings from parsed JSON; switching logic then keys on presence/shape of those strings.

**Risk:** Small wording changes can alter orchestration behavior even if semantic state is the same.

**Fix:** Introduce typed switch fields (e.g., `completion: boolean`, `failure_mode: enum`, `switch_hint: enum`) and only use free text for rationale.

### 5) No execution-time circuit breaker around regime execution calls

The orchestration loop repeatedly calls `execute_regime_once(...)` but has no local timeout/latency guardrails in the loop itself.

**Risk:** If downstream execution hangs or degrades, router-level containment is weak.

**Fix:** Add per-step timeout budget + cumulative orchestration budget, and emit explicit stop reasons (`executor_timeout`, `budget_exhausted`).

### 6) Escalation influence is narrow compared to its intended governance role

Escalation currently affects routing only in selected branches (e.g., stricter/looser pressure checks), while many main paths remain completion/failure driven.

**Risk:** Escalation policy becomes mostly advisory in borderline cases where it should arbitrate.

**Fix:** Integrate escalation as a weighted decision layer in `next_stage` for all eligible transitions, with deterministic tie-break rules.

## Medium-risk weak points

### 7) Analyzer fallback defaults to Exploration too aggressively on analyzer failure

When analyzer output fails, router falls back to low-confidence Exploration.

**Risk:** In urgent decision tasks, this may introduce unnecessary detours.

**Fix:** Use deterministic feature-based fallback prior to Exploration default (e.g., if high decision pressure + low uncertainty, fallback to Operator).

### 8) Builder entry is primarily recurrence-threshold based

Builder switch from Operator depends heavily on `recurrence_potential >= 2.0` plus completion signal.

**Risk:** Recurrence may be over/underestimated by scalar heuristics; can cause premature productization or delayed systemization.

**Fix:** Gate Builder entry with a compound predicate: recurrence evidence + stable interface candidate + at least one repeated use case.

### 9) Cross-stage mismatch handling can over-trust output self-labeling

Misrouting detector can switch based on output claiming a different regime text.

**Risk:** Model self-label drift can induce unnecessary switches.

**Fix:** Require mismatch corroboration from at least one structural misrouting signal before stage re-assignment.

## Prioritized remediation plan

1. Implement typed switch contract fields and migrate away from free-text trigger dependence.
2. Add global `Any -> Exploration` override based on structured collapse conditions.
3. Replace strict prior-stage ban with bounded, audited re-entry policy.
4. Add orchestration execution budgets/timeouts.
5. Expand escalation from advisory to first-class arbitration input.


## What must be reasoned through in each remediation step

### Step 1) Typed switch contract fields
Reason about:
- **Semantic invariants:** Which switch meanings must be machine-stable (`completed`, `failed`, `collapse`, `handoff_ready`) regardless of phrasing.
- **Schema boundary:** What belongs in typed fields vs free-text rationale so control logic never parses prose.
- **Backward compatibility:** How legacy responses without typed fields are interpreted during migration.
- **Validation ownership:** Where schema checks run (output validator vs orchestrator) and which failures are fatal vs recoverable.

### Step 2) Global `Any -> Exploration` override
Reason about:
- **Override threshold:** Which conditions are severe enough to justify global reset (assumption invalidation, frame contradiction, constraint shock).
- **False-positive cost:** How to avoid thrashing into exploration on minor inconsistencies.
- **State preservation:** What survives reset (knowns, contradictions, assumptions provenance) so exploration does not restart from zero.
- **Precedence rules:** When override outranks local stage pathways and escalation advice.

### Step 3) Bounded, audited re-entry policy
Reason about:
- **Progress definition:** How to prove a re-entry is corrective (new evidence, changed constraints, unresolved contradiction narrowed).
- **Loop bounds:** Max visits per stage and stop conditions for pathological cycles.
- **Monotonic telemetry:** Which counters/metrics must strictly improve across revisits.
- **Auditability:** How to store rationale so humans can inspect why a prior stage was re-entered.

### Step 4) Execution budgets/timeouts
Reason about:
- **Budget model:** Per-step timeout, per-orchestration wall-clock budget, and retry budget.
- **Failure semantics:** Difference between timeout, cancellation, transport failure, and invalid output.
- **Degradation path:** What minimal artifact/handoff is emitted on budget exhaustion.
- **Operational tuning:** How model/provider latency variance changes safe defaults.

### Step 5) Escalation as first-class arbitration
Reason about:
- **Decision fusion:** How escalation scores combine with misrouting, completion, and failure signals.
- **Conflict resolution:** Deterministic tie-breakers when signals disagree.
- **Calibration:** Thresholds for stricter vs looser escalation under different risk profiles.
- **Explainability:** What debug trace is required so operators can verify why escalation changed a route.

## Acceptance criteria for the plan

- Each step has explicit invariants, failure modes, and rollback behavior.
- Routing outcomes are deterministic under equivalent typed inputs.
- New tests cover edge cases: semantic-equivalent collapse language, legal stage re-entry, timeout containment, and escalation/signal conflicts.

## Confidence in this assessment

- **High** on structural mismatches with the original spec intent.
- **Medium** on runtime failure severity (because current test suite is strong and passing).
- **Next best validation:** add scenario tests for late-stage collapse, legal re-entry, and semantically equivalent collapse phrasing.
