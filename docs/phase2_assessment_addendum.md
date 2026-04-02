# Phase 2 Assessment Addendum

## Scope

This addendum covers the areas intentionally listed as non-core in the initial remediation plan and prioritizes them by **Impact × Implementation Effort**.

Scoring model used:
- **Impact:** 1 (low) to 5 (very high)
- **Effort:** 1 (low) to 5 (very high)
- **Priority score:** `Impact / Effort` (higher means better near-term leverage)

---

## Priority matrix (Impact × Effort)

| Rank | Area | Impact | Effort | Priority score | Why this priority now |
|---|---|---:|---:|---:|---|
| 1 | Observability depth | 5 | 2 | 2.50 | Improves debugging, routing confidence, and incident response across all other initiatives. |
| 2 | Change-management / migration plan | 5 | 2 | 2.50 | Reduces rollout risk for typed contracts and routing behavior changes already planned. |
| 3 | Evaluation methodology quality | 5 | 3 | 1.67 | Needed to prove “better routing” rather than just changing behavior. |
| 4 | Model/provider reliability layer | 5 | 3 | 1.67 | Directly affects production stability of orchestration and fallback paths. |
| 5 | Human-in-the-loop and override UX | 4 | 3 | 1.33 | Critical for high-stakes confidence while autonomous routing matures. |
| 6 | Cost/latency economics | 4 | 3 | 1.33 | Enables sustainable operation and regime-level budget controls. |
| 7 | Org/process risks | 4 | 3 | 1.33 | Prevents policy/extraction ownership drift and regression blind spots. |
| 8 | Security + abuse handling | 5 | 4 | 1.25 | High impact but broader surface area; should begin early with focused guardrails. |
| 9 | Data/memory lifecycle governance | 5 | 4 | 1.25 | High compliance/quality impact; requires cross-cutting policy and implementation. |
| 10 | Domain-specific correctness risk | 4 | 4 | 1.00 | Important, but should be staged after baseline reliability/eval controls are in place. |

---

## Work packages and reasoning requirements

## WP-A (Immediate, highest leverage)

### 1) Observability depth

**Includes**
- Metric taxonomy
- Alert thresholds
- Route-quality dashboards
- Incident-debug ergonomics beyond current debug fields

**Reasoning required**
- Which metrics are **decision-useful**, not just easy to log.
- Which thresholds should page humans vs create low-priority anomalies.
- How to preserve deterministic replay (same inputs -> same route decisions in debug traces).
- Which trace fields are contract-level (must be stable across versions).

**Suggested deliverables**
- Routing metrics spec v1
- Switch/fallback event schema
- Dashboard and alert playbook

### 2) Change-management / migration plan

**Includes**
- Versioned contract rollout strategy
- Compatibility window
- Canarying
- Rollback criteria
- Deprecation schedule

**Reasoning required**
- What must be backwards-compatible during typed switch contract adoption.
- How to define objective rollback triggers (error budget, switch anomaly rate, latency/cost drift).
- How long dual-read/dual-write must run before removal.
- How to avoid mixed semantics in partially migrated sessions.

**Suggested deliverables**
- Migration RFC with versioning matrix
- Canary + rollback runbook
- Deprecation calendar

---

## WP-B (Near-term quality and stability)

### 3) Evaluation methodology quality

**Includes**
- Gold datasets
- Adversarial benchmarks
- Inter-rater protocol for “right regime first”
- Statistical confidence over time

**Reasoning required**
- What “correct route” means in ambiguous tasks.
- How to separate routing quality from model fluency quality.
- Which sample sizes and confidence intervals are required for release decisions.
- How to prevent benchmark overfitting and silent regressions.

**Suggested deliverables**
- Eval rubric and annotation guide
- Curated benchmark suite
- Release gate policy tied to statistical thresholds

### 4) Model/provider reliability layer

**Includes**
- Retries
- Backoff policy quality
- Rate-limit behavior
- Provider failover strategy
- Partial-response handling

**Reasoning required**
- When to retry vs fail fast (idempotency and semantic risk).
- How provider failover affects determinism and comparability.
- How partial responses are normalized or rejected at contract boundaries.
- Which reliability events should trigger fallback to safe regimes or stop.

**Suggested deliverables**
- Provider reliability policy
- Retry/failover decision table
- Partial-response validator behavior spec

---

## WP-C (Control and operating discipline)

### 5) Human-in-the-loop and override UX

**Includes**
- Operator override mechanisms
- Approval gates for high-stakes transitions
- Audit replay workflows

**Reasoning required**
- Which transitions require approval and by whom.
- How override actions are logged to preserve accountability.
- How to avoid override fatigue while retaining safety.
- How replay supports post-incident learning.

**Suggested deliverables**
- Override policy matrix
- High-stakes gate definitions
- Replay checklist

### 6) Cost/latency economics

**Includes**
- Token/latency budgets per regime
- Cost-aware routing
- Orchestration overhead vs value

**Reasoning required**
- Budget ceilings by regime and risk class.
- How to evaluate if extra switches justify cost/latency.
- When to degrade to cheaper/shorter paths without violating quality gates.
- How to expose cost/latency tradeoffs in routing telemetry.

**Suggested deliverables**
- Regime budget policy
- Cost-per-outcome scorecard
- Degradation ladder

### 7) Org/process risks

**Includes**
- Ownership boundaries (routing policy vs extraction)
- Release cadence
- Regression triage process

**Reasoning required**
- Clear RACI for routing logic, feature extraction, evaluation, and incident response.
- How release cadence aligns with benchmark refresh and migration windows.
- How regressions are triaged and who has authority to rollback.

**Suggested deliverables**
- Ownership charter
- Release and triage SOP

---

## WP-D (Risk-intensive hardening)

### 8) Security + abuse handling

**Includes**
- Prompt-injection resilience
- Untrusted input boundaries
- Output sanitization for downstream tool execution

**Reasoning required**
- Trust boundaries between user input, model output, and tool calls.
- Sanitization and policy checks that are mandatory before execution.
- How to detect and quarantine suspected injection attempts.
- Which security signals should hard-stop orchestration.

**Suggested deliverables**
- Threat model
- Tool-execution policy gate
- Security incident response playbook

### 9) Data/memory lifecycle governance

**Includes**
- Retention policy
- Redaction/PII handling
- Session isolation guarantees
- Long-horizon memory drift controls

**Reasoning required**
- Data minimization constraints for stored routing artifacts.
- Which fields are redacted, encrypted, or forbidden at rest.
- How memory updates are validated to avoid drift and contamination.
- How deletion/audit requirements are verified.

**Suggested deliverables**
- Data governance policy
- Redaction and retention controls
- Memory drift monitoring spec

### 10) Domain-specific correctness risk

**Includes**
- Performance in regulated/high-stakes domains (medical/legal/finance)

**Reasoning required**
- Domain-specific evidence thresholds and prohibited inference types.
- Escalation/approval requirements for high-consequence outputs.
- How domain benchmarks differ from general routing benchmarks.
- What safe fallback behavior means per domain.

**Suggested deliverables**
- Domain risk profile matrix
- Domain evaluation suites
- High-stakes routing policy extensions

---

## Recommended rollout order

1. WP-A (observability + migration)
2. WP-B (evaluation + provider reliability)
3. WP-C (human override + economics + org process)
4. WP-D (security/data/domain hardening)

This order maximizes near-term leverage and reduces uncertainty for later, higher-effort controls.
