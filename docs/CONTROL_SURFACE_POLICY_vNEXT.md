# Cognitive Router — Control Surface Policy vNext

Status: proposed policy artifact  
Purpose: define how deterministic features, demotion, reentry, and Builder entry should behave going forward  
Scope: routing control surface only, not prompt wording or regime library changes

## Grounding

- v1 spec establishes a bottleneck-routed, typed-handoff system, not a monotonic stage ladder.
- v2 transition establishes analyzer-led routing as the primary path and deterministic features as lightweight evidence and sanity checks, not primary scoring authority.
- Live code currently gives sparse deterministic features hard demotion authority in `router/analyzer.py` and hard Builder blocking authority in `router/orchestration/stop_policy.py`.
- Live runtime blocks same-stage recurrence and prior-stage revisitation by default, except for narrow collapse reentry.

## 1. Governing principle

Deterministic features may bias analyzer-led routing, but they may not police regime legitimacy unless they are proven reliable enough for that role.

Default rule:

- analyzer-led routing is primary control
- deterministic features are secondary control
- absence of deterministic support is not negative evidence by itself
- hard overrides require stronger proof than missing lexical matches

## 2. Control authority classes

Every deterministic signal or policy rule must be assigned one of three authority levels.

### A. `hard_veto`

Definition:
A rule may block, demote, or deny progression directly.

Use only when all three are true:

1. the signal has demonstrated high precision across runs
2. false-negative cost is low
3. the rule is not triggered by mere absence of lexical evidence

Current recommendation:
Use almost nowhere.

### B. `soft_guardrail`

Definition:
A rule may lower confidence, bias runner-up selection, add warnings, or require stronger justification, but does not directly rewrite the chosen regime by itself.

Use when:

- signal has some value
- false-negative cost is moderate or high
- analyzer-led routing should remain primary

Current recommendation:
This should be the main role of deterministic features.

### C. `advisory_only`

Definition:
A rule may influence logging, explanation, debugging, and post-run inspection, but does not directly alter stage choice or legality.

Use when:

- the signal is sparse
- the signal is brittle
- the signal is under-audited
- the signal is not serialized clearly enough to inspect per run

Current recommendation:
This should be the default for most weak lexical features.

## 3. Feature authority policy

### 3.1 `decision_pressure`

Current live role:
Used indirectly as an operator legitimacy gate in analyzer demotion logic.

New authority:
`soft_guardrail`

Policy:

- may support operator routing
- may reduce operator confidence if absent
- may make exploration or synthesis a stronger runner-up
- may not hard-demote operator by itself

Reason:
This feature is useful but still lexical and incomplete. False negatives are too costly for hard veto.

### 3.2 `recurrence_potential`

Current live role:
Used for Builder demotion and Builder blocking.

New authority:
`advisory_only` for primary routing  
`soft_guardrail` for endpoint speculation only

Policy:

- may support Builder as a hint
- may weaken Builder endpoint confidence
- may not hard-demote Builder primary routing by itself
- may not hard-block Builder progression using a single sparse threshold

Reason:
The current extractor is too sparse and too lexical for legality policing.

### 3.3 `fragility_pressure`

Current live role:
Used to hard-demote adversarial routing if absent.

New authority:
`advisory_only`, possibly very weak `soft_guardrail` if analyzer confidence is already low

Policy:

- may support adversarial confidence
- may enrich rationale and diagnostics
- may not deny adversarial routing by itself

Reason:
Underfires on obvious adversarial tasks. False-negative cost is high.

### 3.4 `evidence_demand`

Current live role:
Passed into analyzer context; not currently one of the main hard demoters.

New authority:
`advisory_only`

Policy:

- keep as analyzer context
- do not use for direct demotion, blocking, or stage denial until broadened and revalidated

### 3.5 `possibility_space_need`

Current live role:
Feature context only.

New authority:
`advisory_only`

Policy:

- keep as analyzer context
- do not use as direct control until it proves operational value

### 3.6 `structural_signals`

Current live role:
Contextual signal extraction with narrow conjunction logic.

New authority:
`advisory_only`

Policy:

- use as descriptive context for analyzer and diagnostics
- do not treat as regime legitimacy substrate

### 3.7 detected marker families

Current live role:
Affect demotion and legitimacy checks through hidden family matches.

New authority:
`advisory_only` until fully serialized per run

Policy:

- if marker families are not run-visible, they may not carry veto or blocking power
- every consuming rule must expose which marker family was used

## 4. Demotion policy

### 4.1 General rule

Demotion is no longer a default safety reflex.

A proposed stage may be:

- annotated
- softened
- confidence-reduced
- runner-up-shifted

Hard demotion is rare and must satisfy all conditions below:

1. analyzer confidence is below configured threshold
2. analyzer score gap is small
3. deterministic contradiction is positive, not merely absent
4. false-negative cost for the stage is low
5. at least one additional signal supports the demotion

If any of these are missing, use `soft_guardrail` or `advisory_only` behavior instead.

### 4.2 operator-without-decision

Current live behavior:
hard demotion to exploration.

New behavior:
`soft_guardrail`

Allowed effects:

- reduce operator confidence
- strengthen exploration or synthesis as runner-up
- annotate routing summary

Disallowed:

- rewriting primary stage solely because decision markers are absent

### 4.3 builder-without-recurrence

Current live behavior:
hard demotion to exploration.

New behavior:
`advisory_only` for primary route  
`soft_guardrail` for endpoint inference only

Allowed effects:

- note weak recurrence support
- soften Builder endpoint confidence
- request stronger justification if Builder is selected repeatedly

Disallowed:

- hard exploration override from recurrence absence alone

### 4.4 adversarial-without-fragility

Current live behavior:
hard demotion to exploration.

New behavior:
`advisory_only`

Allowed effects:

- annotate weak deterministic fragility support
- slightly lower adversarial confidence only when analyzer confidence is already weak

Disallowed:

- hard demotion based on missing fragility matches

## 5. Builder policy

### 5.1 Builder semantic definition

Builder means reusable infrastructure only.

Builder is for:

- playbooks
- templates
- repeatable systems
- modules and interfaces
- reusable workflows
- compounding operating structure

Builder is not for:

- generic final artifact construction
- one-off documents, specs, or outputs just because they are substantial
- decisions that need to be made now
- polished deliverables without repeatability pressure

This stays aligned with v1 and the current live regime definition, while avoiding conceptual sprawl.

### 5.2 Builder entry rule

Builder remains semantically recurrence-linked, but not lexically threshold-policed.

New rule:
Builder may be selected when repeatability is demonstrated or strongly implied by task shape, analyzer judgment, or prior-stage artifact.

Builder may not be denied solely because `recurrence_potential` is low or zero.

### 5.3 Builder blocking rule

Current live behavior:
Operator to Builder is blocked when `recurrence_potential < 7`.

New behavior:
remove current threshold block.

Replacement:
Builder progression may be deferred only when:

- analyzer justification for recurrence is weak
- no repeated pattern is evidenced in task or handoff
- current artifact is still solving a one-off bottleneck

This is a semantic defer, not a lexical threshold ban.

### 5.4 One-off construction rule

One-off construction remains in Operator unless a separate future construction concept is introduced.

## 6. Reentry policy

### 6.1 Governing principle

The router is bottleneck-cyclic with monotonic default.

Meaning:

- forward progression is default
- revisitation is allowed when a downstream stage surfaces an upstream defect
- loops are blocked when nothing materially changed

This replaces the live anti-history policy that blocks same-stage recurrence and prior-stage revisitation by default.

### 6.2 Blanket bans removed

Remove as default policy:

- same-stage automatic denial
- prior-stage automatic denial
- collapse-only as the sole valid reentry class

### 6.3 Reentry qualification rule

A same-stage retry or prior-stage reentry is allowed only when all four are present:

1. `defect_class`  
One of:
- `frame_failure`
- `evidence_failure`
- `break_condition_discovery`
- `decision_non_actionable`
- `abstraction_overshot`
- `contract_invalidated`
- `new_constraint`

2. `repair_target`  
Why that target stage is the natural repair site

3. `contract_delta`  
What the next visit is supposed to do differently

4. `state_delta`  
What materially changed since the last visit to that stage

If any of these are missing, deny reentry.

### 6.4 Same-stage retry rule

Same-stage retry is conditional, not banned.

Allow only if:

- the purpose of the retry is materially different, or
- the prior failure mode changed, or
- new evidence or constraints changed the brief

Deny if:

- same brief
- same cause
- same expected artifact
- no state delta

### 6.5 Prior-stage reentry rule

Prior-stage reentry is conditional, not banned.

Allow only when a downstream stage exposes a real upstream defect.

Examples:

- synthesis -> epistemic -> synthesis
- epistemic -> adversarial -> epistemic
- operator -> epistemic
- builder -> operator
- adversarial -> synthesis

### 6.6 Anti-ping-pong rule

Disallow:

- repeating the same stage pair for the same cause without stronger state delta
- revisiting a prior stage when the reentry justification is only stylistic preference
- oscillating between stages with unchanged artifact target

## 7. Transition policy matrix

| from | to | allowed | default | only_if_justified | never | required_justification |
|---|---|---:|---:|---:|---:|---|
| exploration | exploration | yes | no | yes | no | new evidence or tighter contract |
| exploration | synthesis | yes | yes | no | no | none |
| exploration | epistemic | yes | yes | no | no | none |
| exploration | adversarial | yes | no | yes | no | stress-test needed |
| exploration | operator | yes | yes | no | no | none |
| exploration | builder | yes | no | yes | no | recurrence proven or strongly implied |
| synthesis | exploration | yes | no | yes | no | frame failure; new frame needed |
| synthesis | synthesis | yes | no | yes | no | integration repair or tighter contract |
| synthesis | epistemic | yes | yes | no | no | none |
| synthesis | adversarial | yes | yes | no | no | none |
| synthesis | operator | yes | yes | no | no | none |
| synthesis | builder | yes | no | yes | no | recurrence proven or strongly implied |
| epistemic | exploration | yes | no | yes | no | question decomposition wrong |
| epistemic | synthesis | yes | no | yes | no | integration repair after evidence clarified |
| epistemic | epistemic | yes | no | yes | no | new evidence or sharper evidentiary question |
| epistemic | adversarial | yes | yes | no | no | none |
| epistemic | operator | yes | yes | no | no | none |
| epistemic | builder | yes | no | yes | no | recurrence proven or strongly implied |
| adversarial | exploration | yes | no | yes | no | critique exposed missing frame alternatives |
| adversarial | synthesis | yes | no | yes | no | survivable revision needs reintegration |
| adversarial | epistemic | yes | no | yes | no | evidence boundary recalibration needed |
| adversarial | adversarial | yes | no | yes | no | new attack surface or stronger break condition |
| adversarial | operator | yes | yes | no | no | none |
| adversarial | builder | yes | no | yes | no | recurrence proven after stress-tested survival |
| operator | exploration | yes | no | yes | no | premise collapse or missing decomposition |
| operator | synthesis | yes | no | yes | no | available frames do not support decision |
| operator | epistemic | yes | no | yes | no | unresolved support or contradiction blocks action |
| operator | adversarial | yes | no | yes | no | risk may change the decision |
| operator | operator | yes | no | yes | no | new constraints, new evidence, or invalid prior packet |
| operator | builder | yes | no | yes | no | recurrence proven or strongly implied |
| builder | exploration | yes | no | yes | no | abstraction overshot; framing wrong |
| builder | synthesis | yes | no | yes | no | architecture lacks coherent center |
| builder | epistemic | yes | no | yes | no | recurrence, assumption, or interface evidence weak |
| builder | adversarial | yes | no | yes | no | architecture needs stress test |
| builder | operator | yes | no | yes | no | abstraction overshot; concrete action is bottleneck |
| builder | builder | yes | no | yes | no | tighter build target or new recurrence evidence |

Structural never-rules:

| from | to | allowed | default | only_if_justified | never | required_justification |
|---|---|---:|---:|---:|---:|---|
| any | same stage, same brief, same cause | no | no | no | yes | never |
| any | previously visited stage with no state delta | no | no | no | yes | never |
| any | repeated oscillation with same cause and same target | no | no | no | yes | never |
| any | builder based only on hypothetical recurrence | no | no | no | yes | never |

## 8. Stop policy alignment

### 8.1 Stop-policy principle

Stopping must reconcile:

- actual executed stage
- actual artifact type
- inferred endpoint
- recommended next regime
- current state deltas

Current live stop policy can stop on `artifact_complete_at_or_past_endpoint` and also hard-block Builder with a recurrence threshold.

New policy:

- remove lexical Builder threshold block
- do not treat endpoint completion as valid unless actual artifact and stage alignment is satisfied
- same-stage or prior-stage recommendations are not auto-illegal; they must go through reentry qualification
- forward recommendation deferral should respect justified reentry, not just forward-only progression

### 8.2 Endpoint completion rule

A run may stop as endpoint-complete only if:

1. current artifact is valid
2. current artifact type matches executed stage contract
3. current executed stage is endpoint stage, or an explicitly allowed endpoint-equivalent
4. no justified reentry or forward move is still active

Absence of these conditions means no endpoint-complete stop.

## 9. Serialization and observability requirements

No hidden lexical authority.

Every run record must serialize:

- matched feature families
- per-family matches
- which policy rule consumed which feature
- whether the feature caused warning, soft guardrail, or hard action
- pre-policy analyzer choice
- post-policy chosen stage
- reason any reentry was allowed or denied
- `defect_class`, `contract_delta`, and `state_delta` for reentry decisions

If a feature cannot be inspected in run output, it may not hold regime-legitimacy authority.

## 10. Immediate implementation consequences

### keep

- analyzer-led routing as primary
- decision_pressure as weak prior
- recurrence_potential as hint
- anti-ping-pong goal in principle

### soften

- operator-without-decision demotion
- Builder endpoint softening
- any lexical legitimacy checks that currently behave like silent vetoes

### remove or redesign

- builder-without-recurrence hard veto
- adversarial-without-fragility hard veto
- Builder recurrence threshold block in stop policy
- same-stage automatic denial
- prior-stage automatic denial
- collapse-only as sole legitimate reentry class
- opaque marker-family authority without serialization

## 11. Repo-facing change targets

Primary files implicated by this policy:

- `router/analyzer.py` — demotion authority rewrite
- `router/routing/feature_extraction.py` — feature status remains secondary; improve observability first
- `router/runtime/session_runtime.py` — replace anti-history bans with qualified reentry policy
- `router/orchestration/stop_policy.py` — remove lexical Builder threshold block; tighten endpoint completion reconciliation
