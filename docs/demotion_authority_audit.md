# Demotion Authority Audit: `router/analyzer.py`

## Scope and goal

This audit examines whether current **demotion rules** are appropriately strong relative to their evidence, independent of any weakness in feature extraction itself. Focus is on whether rules treat **absence of deterministic support** as if it were negative counter-evidence against analyzer judgment.

Primary code under review:

- `TaskAnalyzer.decision_from_analysis` demotion logic in `router/analyzer.py`.
- Supporting signal semantics in `router/routing/feature_extraction.py`.

---

## Judgment table

| demotion rule | intended purpose | actual trigger | analyzer can be vetoed? | uses positive evidence or missing evidence? | likely false negative rate | likely damage when wrong | recommendation |
|---|---|---|---|---|---|---|---|
| operator without decision evidence → exploration | Prevent premature operational closure when no decision pressure exists | Fires when analyzer primary is `operator` AND `routing_features.decision_pressure == 0` AND no `decision_tradeoff_commitment` marker | **Yes** (hard override of analyzer primary) | Mostly **missing evidence** (absence of lexical marker/score), not positive anti-operator evidence | **Medium–High** for implicit decision tasks where decision language is indirect | **High**: forced exploration can add latency and dilute direct-answer tasks | **Soften to soft penalty**; keep as guardrail, not veto |
| builder without recurrence → exploration | Prevent over-escalation to builder when reuse/systemization is not evidenced | Fires when analyzer primary is `builder` AND `routing_features.recurrence_potential == 0` | **Yes** | Mostly **missing evidence**; recurrence score relies on narrow strong cues and excludes generic recurrence from score | **High** when recurring intent is implied but not explicitly phrased | **High**: suppresses infrastructure/system design routes and biases toward shallow outputs | **Remove hard veto**; convert to warning-only or confidence-gated soft penalty |
| adversarial without fragility → exploration | Avoid adversarial stress-testing mode when no fragility pressure is present | Fires when analyzer primary is `adversarial` AND `routing_features.fragility_pressure == 0` | **Yes** | Mostly **missing evidence** of fragility terms, not positive evidence that adversarial mode is wrong | **Medium**; depends on domain where risk is implicit rather than explicit | **Medium–High**: can skip needed failure-mode analysis in high-consequence tasks | **Soften to soft penalty** with risk-sensitive exceptions |

---

## Rule-by-rule assessment

### 1) Operator demotion

**Current behavior**

- Rule: if analyzer primary is operator and deterministic decision evidence is zero/missing marker, force exploration.
- This is a hard override: analyzer stage score ranking and confidence do not protect operator from demotion.

**Why this is too strong**

- Trigger is mostly absence-based. Lack of detected decision tokens is not equivalent to evidence that operator is wrong.
- The rule does not check analyzer confidence, analyzer score gap, or whether task wording is imperative/decision-like but lexically indirect.

**Recommended policy**

- Keep as a **soft penalty** (e.g., subtract from operator score or lower confidence level), not hard override.
- Escalate to hard demotion only when multiple independent negatives align (e.g., no decision markers + analyzer confidence low + small score gap + exploration/synthesis close competitor).

---

### 2) Builder demotion

**Current behavior**

- Rule: if analyzer primary is builder and recurrence potential is zero, force exploration.

**Why this is too strong**

- Recurrence trigger in extraction is strict and score uses strong recurrence cues only.
- Generic recurrence hints are captured as markers but not included in `recurrence_potential` score used by this demotion path.
- Therefore, hard demotion can occur due to sparse lexical coverage rather than true anti-builder evidence.

**Recommended policy**

- Hard veto should be **removed** for primary-stage demotion.
- Replace with **warning-only** when analyzer confidence is high or score gap is large.
- At most, apply a **soft penalty** when confidence is moderate/low and no recurrence indicators appear across multiple channels.

---

### 3) Adversarial demotion

**Current behavior**

- Rule: if analyzer primary is adversarial and fragility pressure is zero, force exploration.

**Why this is too strong**

- Again, trigger is absence of deterministic fragility evidence.
- No use of analyzer confidence, no score-gap threshold, no task-severity/risk wording checks in this demotion gate.

**Recommended policy**

- Keep as **soft penalty**, not hard override.
- Allow hard demotion only with combined conditions: low analyzer confidence + low analyzer margin + no fragility/risk indicators + non-critical domain cues.

---

## Cross-cutting findings

1. **Absence of deterministic support is currently treated as negative evidence.**
   - This is the strongest issue and applies to all three demotion rules.

2. **Demotions are hard vetoes of analyzer primary regime.**
   - The analyzer can be overridden even if it expresses high confidence.

3. **Analyzer confidence is computed but not used to gate demotion authority.**
   - Confidence affects reporting, not override policy.

4. **Analyzer stage score gap is ignored by demotion rules.**
   - A large model margin should increase resistance to veto by sparse deterministic misses.

5. **Task wording nuance is not incorporated into demotion authority.**
   - Deterministic checks are marker-centric and brittle to paraphrase/implicit intent.

6. **Fallback to exploration is aggressive and one-directional.**
   - All three rules demote to exploration, which can over-broaden tasks better served by operator/adversarial/builder.

---

## Recommended policy framework

Use a tiered authority model for demotions:

1. **Warning-only signal** (default for missing-evidence conditions)
   - Trigger when deterministic evidence is absent but there is no explicit contradictory evidence.

2. **Soft penalty** (score/priority adjustment)
   - Apply when missing-evidence condition persists and analyzer confidence is not high.

3. **Hard override** (rare)
   - Require multiple independent signals:
     - missing deterministic support,
     - **and** low analyzer confidence,
     - **and** small analyzer score gap,
     - **and** no corroborating risk/task wording signals.

4. **No-demotion path for strong analyzer certainty**
   - If analyzer confidence high + margin large, preserve analyzer primary and attach warning note.

---

## Concrete per-rule disposition

- **operator without decision evidence**: **Soften** (hard → soft penalty).
- **builder without recurrence**: **Remove as hard veto** (becomes warning-only by default; optional soft penalty when analyzer uncertainty is high).
- **adversarial without fragility**: **Soften** (hard → soft penalty, risk-aware).

---

## Optional implementation sketch (policy, not extractor changes)

Within `decision_from_analysis`, evaluate demotion authority as:

- Compute `analyzer_margin = top_score - runner_up_score` from `analyzer_result.stage_scores`.
- Define demotion levels (`warning`, `soft_penalty`, `hard_override`) instead of immediate reassignment.
- Promote to hard override only when:
  - deterministic-missing condition true,
  - analyzer confidence below threshold,
  - analyzer margin below threshold,
  - and no reinforcing task/risk indicators.
- Otherwise retain analyzer primary and append warning note to summary.

This separates “extractor may miss a cue” from “analyzer is wrong.”
