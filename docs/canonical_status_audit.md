# Canonical status audit (current behavior)

This audit captures what each consumer currently **decides** versus merely **consumes**.

## Decision map by file

### `router/validation.py`
- **Role:** decides.
- **Decisions made:**
  - Whether output is parseable JSON (`valid_json`).
  - Whether required top-level keys are present (`required_keys_present`).
  - Whether required artifact fields are present (`artifact_fields_present`).
  - Whether `artifact_type` matches the stage (`artifact_type_matches`).
  - Whether control fields are valid (`contract_controls_valid`, with `control_failures`).
  - Whether semantics pass (`semantic_valid`, with `semantic_failures`).
  - Final validity (`is_valid = structural_valid && semantic_valid`).
- **What it does not decide:** stop/switch actions.

### `router/orchestration/output_contract.py`
- **Role:** consumes only.
- **Decisions made:** none.
- **Purpose:** container for `stage`, raw output text, and `validation` payload.

### `router/runtime/state_updater.py`
- **Role:** both decides and consumes.
- **Consumes:** validator output booleans/parsed fields.
- **Decides:**
  - `structurally_trustworthy` using the structural booleans.
  - Local `artifact_complete` as `completion_signal present && is_valid && completion_signal != failure_signal`.
  - `failure_signal_seen` as `bool(failure_signal) || !is_valid`.
  - Contract-delta labels:
    - semantic failures => `contract_invalidated_by_semantic_failure`.
    - both completion and failure present and unequal => `contract_invalidated_by_control_conflict`.
    - otherwise either `next_stage_contract_changed` or `artifact_contract_advanced`.

### `router/orchestration/stop_policy.py`
- **Role:** both decides and consumes.
- **Consumes:** validation result, router state, routing decision.
- **Decides:**
  - `artifact_complete` locally (`is_valid && completion_signal present && completion != failure`).
  - Stage artifact match requirement (`artifact_type` + `regime`).
  - Collapse override behavior.
  - Whether to defer stop for forward recommendations.
  - Final stop outcome (`StopDecision.should_stop`) and reason codes.

### `router/orchestration/transition_rules.py`
- **Role:** both decides and consumes.
- **Consumes:** validation payload via `RegimeOutputContract`.
- **Decides:**
  - Helper predicates for route semantics:
    - `operator_semantic_failure`
    - `control_failure_regime_mismatch`
    - assumption/frame collapse interpretation.
  - Defect classification (`defect_class_from_context`).
  - Reentry justification requirement + payload.
  - Next stage (`next_stage`) from completion/failure/escalation/misrouting context.

## Duplication currently present

The following concepts are currently evaluated in more than one place:

- **Artifact complete**
  - `state_updater` and `stop_policy` both compute their own forms.
- **Failure seen**
  - `state_updater` computes via `failure_signal || !is_valid`.
  - `transition_rules` and `stop_policy` infer from failure/control/semantic conditions.
- **Control conflict / contradictory controls**
  - `state_updater` tags control conflict when completion and failure are both present and unequal.
  - `stop_policy` treats completion==failure as incomplete.
- **Switch-worthy / repair-worthy posture**
  - `transition_rules` computes switching/reentry logic.
  - `stop_policy` computes stopping posture.
  - `state_updater` sets contract delta labels used by reentry logic.

## Canonical shape (defined before integration patches)

```python
@dataclass(frozen=True)
class CanonicalStatus:
    terminal_signal: Literal["completion", "failure", "contradictory", "neither"]
    artifact_status: Literal["valid_complete", "valid_blocked", "invalid", "repairable"]
    switch_posture: Literal["stay", "repair", "stop", "switch"]
    completion_signal: str
    failure_signal: str
    is_valid: bool
    structurally_valid: bool
    semantic_valid: bool
    control_conflict: bool
    recommended_next_stage: Optional[Stage]
```

Derived by:
- `canonical_status_from_validation(validation_result, current_stage=None, should_stop=False)`.

Interpretation:
- `terminal_signal`: one source of truth for completion/failure/contradiction.
- `artifact_status`: one source of truth for valid-complete vs blocked vs invalid vs repairable.
- `switch_posture`: shared posture that downstream consumers can read instead of re-deriving.
