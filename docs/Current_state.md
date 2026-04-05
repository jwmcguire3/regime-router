# Cognitive Router — Current State Detailed Standalone

Last updated: April 5, 2026 (repo-alignment pass against current code)

## 1. What this system is

The Cognitive Router is a runtime for controlling how an LLM approaches a task under an external control plane.

It is a constrained staged-execution system. The runtime does not simply send a prompt and accept whatever comes back. It classifies the task, analyzes the task shape, selects a regime, composes that regime from typed primitives, executes under a structured output contract, validates the result outside the model, computes forward state, and may continue through a bounded orchestration loop.

The current runtime combines five layers of control:

1. **task-shape inspection** through deterministic classification and structural feature extraction,
2. **route proposal** through analyzer-led assessment when enabled, with deterministic feature-led fallback when analyzer use is disabled or unavailable,
3. **behavioral regime composition** through typed line primitives,
4. **external validation and bounded repair** of model outputs,
5. **stateful orchestration** across one or more regime steps with handoff continuity and stop-policy control.

The system operates over six stage families:

- exploration
- synthesis
- epistemic
- adversarial
- operator
- builder

These are behavior families with stage-specific contracts, failure tendencies, completion signals, artifact types, and switching logic.

The runtime owns the control plane. The model is used for analysis proposals and staged artifact generation, but the runtime outside the model determines the active regime, validates outputs, chooses repair mode, computes handoffs, records state, and decides whether execution should stop or continue.

This document describes the implementation and operating behavior that currently exists.

### 1.1 Recent commit-backed updates (last 15 commits)

A review of commits `5d5eca6` through `528b34b` confirms the following are now true in the current codebase:

- **Analyzer-led planning remains the primary path**, with analyzer/feature signals integrated into routing quality markers and planner behavior.
- **Runtime model clients are lazily initialized** in `CognitiveRouterRuntime`, so construction does not immediately require provider credentials until a model-backed operation is invoked.
- **Stop policy now includes artifact-aware completion semantics** with additional regression coverage.
- **Executor/runtime behavior includes explicit invalid-output recovery fallback handling** in orchestration and state progression paths.
- **Downstream handoff discipline was strengthened in prompts**, and continuity expectations are now regression-tested.

This section is intentionally commit-derived so that the rest of the document can be read with the correct operational assumptions.

---

## 2. Core operating model

A normal run currently follows this shape:

1. accept a task,
2. classify the task to extract a direct-vs-regime signal,
3. extract deterministic structural features from the task,
4. call the analyzer with task text, structural features, task signals, risk profile, and classifier signal,
5. convert analyzer output into a `RoutingDecision`, including start-stage and likely endpoint inference,
6. decide whether direct fast-path planning is allowed,
7. if direct fast-path planning is allowed, compose a direct passthrough regime,
8. execute the selected regime through the regime executor,
9. build system and user prompts for that regime,
10. execute the model call,
11. validate the output against the stage contract,
12. if first-pass validation fails, select one bounded repair mode and perform one repair attempt,
13. update canonical router state from the validated result,
14. compute a forward handoff for any subsequent stage,
15. if orchestration is enabled, evaluate stop policy before asking whether to switch,
16. if stop policy allows continuation, run misrouting detection and switch orchestration,
17. if a switch is approved, execute the next stage with the prior handoff threaded into the prompt,
18. continue until stop policy ends the run, a hard orchestration ceiling is reached, or no switch is justified,
19. serialize session output for inspection or persistence.

The codebase currently has both:

- a direct execution helper (`router/execution/direct_execution.py`), and
- the staged regime execution path (`router/execution/executor.py`).

At runtime, `CognitiveRouterRuntime.execute(...)` currently routes execution through the regime executor path (including direct passthrough regimes) rather than calling the direct helper from the main execute loop.

---

## 3. Main components

The codebase is organized around a small number of concrete subsystems.

### 3.1 Runtime

`router/runtime/__init__.py` exposes `CognitiveRouterRuntime`, the main runtime entry point.

The runtime constructs and wires together:

- `Router()`
- `RegimeComposer()`
- `OutputValidator()`
- `PromptBuilder()`
- `EvolutionEngine()`
- `MisroutingDetector()`
- `EscalationPolicy()`
- `SwitchOrchestrator()`
- `StopPolicy()`
- lazy model client plumbing via `create_model_client(...)` + `_ensure_model_client(...)`
- `TaskAnalyzer(...)`
- `TaskClassifier()`
- `RuntimePlanner(...)`
- `RegimeExecutor(...)`
- `SessionRuntime(...)`

The runtime exposes user-facing methods such as:

- `plan(...)`
- `execute(...)`
- `list_models()`
- `restore_router_state(...)`

### 3.2 Planner

`router/runtime/planner.py` contains `RuntimePlanner`.

The planner converts analysis inputs into planning state. It produces:

- a `RoutingDecision`,
- a composed `Regime`,
- a `Handoff`,
- a fully initialized `RouterState`,
- a `TaskClassification`.

The planner is deterministic. Given identical inputs, it produces identical planning outputs. It does not call the model client.

### 3.3 Task classifier

`router/classifier.py` contains `TaskClassifier`.

The classifier extracts a deterministic direct-vs-regime signal from the task text. Its output is used as structured input to planning and analysis. It does not independently determine whether analysis is skipped.

### 3.4 Task analyzer

`router/analyzer.py` contains `TaskAnalyzer`.

When enabled and available, this is the primary route proposer for staged tasks. It receives the task, deterministic features, task signals, risk profile, and the classifier signal, and returns structured route analysis in strict JSON.

If analyzer use is disabled (or analyzer output is unavailable), planning falls back to deterministic feature-led routing through `Router.route(...)`.

### 3.5 Regime composition

`router/routing/composer.py` and `router/routing/grammar_composer.py` contain regime composition logic.

The active regime is composed from typed primitives rather than from a single canned stage prompt.

### 3.6 Prompt construction

`router/prompts.py` contains `PromptBuilder`.

This builds regime-aware system prompts, user prompts, and repair prompts. It also threads prior-stage handoff context into later-stage user prompts when orchestration continues.

### 3.7 Execution

`router/execution/executor.py` contains `RegimeExecutor`.

`router/execution/direct_execution.py` contains the direct execution path.

`router/execution/repair_policy.py` contains repair-mode selection.

### 3.8 Validation

`router/validation.py` contains `OutputValidator`.

This validates JSON structure, stage-specific artifact fields, control fields, grounding, and semantic quality.

### 3.9 Orchestration

The control plane for multi-step continuation lives in `router/orchestration/`.

This includes:

- `misrouting_detector.py`
- `switch_orchestrator.py`
- `transition_rules.py`
- `escalation_policy.py`
- `escalation_rules.py`
- `output_contract.py`
- `stop_policy.py`

### 3.10 Canonical state and serialization

`router/state.py` defines the canonical state objects and JSON conversion helpers.

`router/storage.py` handles persistence.

---

## 4. Providers and model clients

The runtime is provider-aware.

The active model client is created lazily by `create_model_client(...)` (via `_ensure_model_client(...)`) inside the runtime. The currently supported providers are:

- `ollama`
- `openai`
- `deepseek`

### 4.1 Ollama provider

When `provider="ollama"`, the runtime constructs `OllamaModelClient` using the configured Ollama base URL.

### 4.2 OpenAI-compatible providers

When `provider="openai"` or `provider="deepseek"`, the runtime constructs `OpenAIModelClient` using:

- `openai_base_url`
- `openai_api_key_env`

The runtime reads the API key from the configured environment variable and raises a runtime error if that variable is missing or empty.

### 4.3 Provider transition behavior

The CLI/settings layer now includes provider-transition logic for OpenAI-compatible endpoints. When the provider changes between `openai` and `deepseek`, endpoint defaults are automatically refreshed unless explicit overrides are provided.

This keeps `openai_base_url` and `openai_api_key_env` aligned with the selected provider while still allowing custom values when intentionally set.

### 4.4 Settings defaults

The settings and runtime layers currently define:

- default provider: `deepseek`
- default Ollama model: `dolphin29:latest`
- default OpenAI model: `gpt-5.4-mini`
- default OpenAI base URL: `https://api.openai.com/v1`
- default OpenAI API key environment variable: `OPENAI_API_KEY`
- default DeepSeek model: `deepseek-reasoner`
- default DeepSeek base URL: `https://api.deepseek.com`
- default DeepSeek API key environment variable: `DEEPSEEK_API_KEY`
- default `use_task_analyzer`: `True`
- default `bounded_orchestration`: `True`
- default `max_switches`: `2`
- default model-control profile: `strict`

The system supports provider-aware model listing and provider-aware model defaulting through runtime settings and the CLI surface.

Note on defaults: CLI/user settings default `bounded_orchestration=True`, and `CognitiveRouterRuntime.execute(...)` also currently defaults `bounded_orchestration=True`.

---

## 5. Task classification

Every task currently passes through `TaskClassifier.classify()` before analysis.

### 5.1 Purpose

The classifier extracts a deterministic routing signal from the task text. The signal is included in analyzer context and state, and participates in the runtime’s decision about whether the direct path is allowed.

### 5.2 Current rule

The classifier looks for an imperative action-plus-artifact pattern near the start of the task text.

It tokenizes the task and inspects the first ten words for:

- an action verb, followed by
- an artifact noun later in that early segment.

### 5.3 Action verbs

The built-in action verbs are:

- write
- create
- build
- make
- generate
- draft
- code
- implement
- design
- draw
- compose
- translate
- convert
- fix
- refactor
- debug
- test
- deploy
- install
- set up
- configure

### 5.4 Artifact nouns

The built-in artifact nouns are:

- code
- script
- function
- class
- module
- app
- application
- page
- component
- email
- letter
- document
- report
- presentation
- spreadsheet
- database
- api
- endpoint
- test
- game
- website
- dashboard
- chart
- diagram
- file
- bug
- login
- flow
- environment

### 5.5 Classification outputs

If the early action-plus-artifact pattern is found, classification returns:

- `route_type="direct"`
- `confidence=0.92`
- `classification_source="pattern"`

Otherwise it returns:

- `route_type="regime"`
- `confidence=0.85`
- `classification_source="fallback"`

### 5.6 How classification is used in runtime control

The classifier output is used as an advisory control signal.

The runtime can allow direct fast-path execution only when all of the following are true:

1. the analyzer confidence is greater than `0.9`,
2. the analyzer proposes a single regime with no structural tension,
3. the classifier returned `route_type="direct"`.

If any of those conditions fail, the task proceeds through staged regime execution.

---

## 6. Routing features and task signals

Before analyzer-led route proposal, the runtime extracts deterministic structural features from the task.

### 6.1 RoutingFeatures object

The core feature object is `RoutingFeatures`, with fields:

- `structural_signals`
- `decision_pressure`
- `evidence_demand`
- `fragility_pressure`
- `recurrence_potential`
- `possibility_space_need`
- `detected_markers`

### 6.2 Named structural signals

The system currently defines three named structural signals:

- `expansion_when_defined`
- `concrete_versions_feel_too_small`
- `fragments_understood_spine_missed`

### 6.3 Purpose of extracted features

These features do not directly execute the task. They serve three roles:

1. they provide a deterministic read of task shape,
2. they are fed into analyzer prompts as structured evidence,
3. they are reused later in validation, handoff construction, and orchestration.

---

## 7. Task analyzer and route proposal

`TaskAnalyzer` is the primary route proposer for routed work.

### 7.1 Analyzer input

The analyzer receives:

- the raw task text,
- the extracted `RoutingFeatures`,
- task signals,
- a risk profile,
- the classifier signal.

### 7.2 Analyzer prompt contract

The analyzer asks the model for strict JSON only.

The required analyzer output fields currently include:

- `bottleneck_label`
- `candidate_regimes`
- `stage_scores`
- `structural_signals`
- `decision_pressure`
- `evidence_quality`
- `recurrence_potential`
- `confidence`
- `rationale`
- `likely_endpoint_regime`
- `endpoint_confidence`

The stage scores must include numeric entries for all six stages.

The analyzer prompt also includes a structured classifier assessment, such as:

- classifier route type,
- classifier confidence,
- classification source.

It also includes endpoint-inference instruction. The analyzer estimates which regime is likely to produce the minimum useful artifact for the task. Operator is treated as the default endpoint when builder-specific recurrence is not strongly supported.

### 7.3 Analyzer behavior

`analyze(...)` sends the request to the active model client with:

- `temperature=0.0`
- `num_predict=500`
- streaming disabled

The analyzer then attempts to parse and validate the response.

### 7.4 Analyzer parsing and repair behavior

The analyzer has a layered recovery path:

1. try direct JSON parsing,
2. strip markdown code fences if present,
3. extract the first JSON object if extra text exists,
4. if still invalid, make a JSON-repair call,
5. if the JSON is present but missing required fields, make a missing-field repair call.

If all parsing or repair attempts fail, the analyzer returns `None`.

### 7.5 Analyzer validation

The analyzer validates that:

- the payload is a dictionary,
- all required top-level fields exist,
- `bottleneck_label` is a non-empty string,
- `rationale` is a string,
- candidate regimes are valid stage values,
- stage scores exist for all stages and are numeric,
- structural signals are a list of strings,
- decision/evidence/recurrence fields are integers from 0 to 10,
- `confidence` is numeric from 0 to 1,
- `likely_endpoint_regime` is a valid stage value,
- `endpoint_confidence` is numeric from 0 to 1.

### 7.6 Propose-route behavior

`propose_route(...)` converts valid analyzer output into a `RoutingDecision`.

It:

1. sorts stage scores,
2. chooses the highest-scoring stage as primary,
3. chooses the highest different stage as runner-up,
4. applies structural demotion rules,
5. maps confidence to low/medium/high,
6. generates `why_primary_wins_now`,
7. generates `switch_trigger`,
8. assembles an analyzer summary string,
9. carries endpoint inference into the routing decision,
10. applies endpoint demotion and endpoint clamping rules.

### 7.7 Structural demotion rules

Even when the analyzer proposes a stage, the result can be demoted if the task does not show the right feature evidence.

Current demotion rules are:

- operator is demoted to exploration if decision pressure is zero and no decision markers are present,
- builder is demoted to exploration if recurrence potential is zero,
- adversarial is demoted to exploration if fragility pressure is zero.

### 7.8 Endpoint rules

Endpoint inference is carried separately from primary-stage selection.

Current endpoint rules include:

- if `likely_endpoint_regime` is `builder` but `recurrence_potential == 0`, the endpoint is demoted to `operator`,
- if the inferred endpoint would fall earlier than the selected primary stage in the stage progression, the endpoint is clamped forward to at least the primary stage,
- if endpoint inference is absent, the runtime defaults the endpoint to `operator` with moderate confidence.

### 7.9 Fallback behavior

If the analyzer fails completely, `propose_route(...)` returns a conservative fallback routing decision:

- primary regime: exploration
- runner-up regime: synthesis
- confidence level: low
- rationale: analyzer unavailable, exploration is the safest fallback
- likely endpoint: operator
- endpoint confidence: moderate default

### 7.10 Confidence mapping

Analyzer confidence is mapped as follows:

- `>= 0.8` → high
- `>= 0.5` → medium
- `< 0.5` → low

The resulting `RegimeConfidenceResult` includes:

- level
- rationale
- top stage score
- runner-up score
- score gap
- nontrivial stage count
- weak lexical dependence flag
- structural feature state

The `weak_lexical_dependence` field remains part of the object model even though routing is analyzer-led.

---

## 8. RoutingDecision object

A `RoutingDecision` records the structured route proposal.

Its fields include:

- `bottleneck`
- `primary_regime`
- `runner_up_regime`
- `why_primary_wins_now`
- `switch_trigger`
- `confidence`
- `deterministic_stage_scores`
- `deterministic_score_summary`
- `deterministic_score_contributions`
- `analyzer_enabled`
- `analyzer_used`
- `analyzer_changed_primary`
- `analyzer_changed_runner_up`
- `analyzer_summary`
- `likely_endpoint_regime`
- `endpoint_confidence`

The decision records both the chosen entry regime and the currently inferred terminal regime.

---

## 9. Stage system

The router currently operates over six stage families.

### 9.1 Exploration

Purpose: generate and compare structurally distinct candidate frames.

Canonical artifact type: `candidate_frame_set`

Required artifact fields:

- `candidate_frames`
- `selection_criteria`
- `unresolved_axes`

Completion signal hint: `selection_criteria_ready`

Failure signal hint: `frames_not_structurally_distinct`

Canonical overuse failure: `branch sprawl`

### 9.2 Synthesis

Purpose: produce the strongest coherent interpretation from live signals.

Canonical artifact type: `dominant_frame`

Required artifact fields:

- `central_claim`
- `organizing_idea`
- `key_tensions`
- `supporting_structure`
- `pressure_points`

Completion signal hint: `coherent_frame_stable`

Failure signal hint: `frame_collapses_under_pressure_points`

Canonical overuse failure: `false unification`

### 9.3 Epistemic

Purpose: separate supported claims from uncertainty and gaps.

Canonical artifact type: `evidence_map`

Required artifact fields:

- `supported_claims`
- `plausible_but_unproven`
- `contradictions`
- `omitted_due_to_insufficient_support`
- `decision_relevant_conclusions`

Completion signal hint: `evidence_boundary_clear`

Failure signal hint: `insufficient_support_for_key_claims`

Canonical overuse failure: `under-synthesis / decision drag`

### 9.4 Adversarial

Purpose: stress test the frame with destabilizers and break conditions.

Canonical artifact type: `stress_report`

Required artifact fields:

- `top_destabilizers`
- `hidden_assumptions`
- `break_conditions`
- `survivable_revisions`
- `residual_risks`

Completion signal hint: `critical_breakpoints_mapped`

Failure signal hint: `destabilizers_unresolved_or_redundant`

Canonical overuse failure: `nihilistic or repetitive critique`

### 9.5 Operator

Purpose: commit to a concrete decision with executable next moves.

Canonical artifact type: `decision_packet`

Required artifact fields:

- `decision`
- `rationale`
- `tradeoff_accepted`
- `next_actions`
- `fallback_trigger`
- `review_point`

Completion signal hint: `decision_committed_with_actions`

Failure signal hint: `decision_not_actionable_under_constraints`

Canonical overuse failure: `forced closure`

### 9.6 Builder

Purpose: convert insight into reusable architecture and modules.

Canonical artifact type: `system_blueprint`

Required artifact fields:

- `reusable_pattern`
- `modules`
- `interfaces`
- `required_inputs`
- `produced_outputs`
- `implementation_sequence`
- `compounding_path`

Completion signal hint: `blueprint_ready_for_build_sequence`

Failure signal hint: `architecture_not_modular_or_reusable`

Canonical overuse failure: `over-engineering`

---

## 10. Primitive library

The regime system is built from a typed line-primitive library stored in `router/models.py`.

### 10.1 Primitive schema

Each `LinePrimitive` contains:

- `id`
- `text`
- `stage`
- `function`
- `attractor`
- `suppresses`
- `tension`
- `risks`
- `compatible_with`
- `incompatible_with`

### 10.2 Function types

The current function types are:

- dominance
- suppression
- shape
- gate
- transfer

### 10.3 Primitive count

The current primitive count is 32.

Breakdown by stage:

- exploration: 5
- synthesis: 5
- epistemic: 6
- adversarial: 6
- operator: 5
- builder: 5

### 10.4 Primitive IDs by stage

Exploration:

- `EXP-D1`
- `EXP-S1`
- `EXP-P1`
- `EXP-S2`
- `EXP-T1`

Synthesis:

- `SYN-D1`
- `SYN-D2`
- `SYN-S1`
- `SYN-P1`
- `SYN-P2`

Epistemic:

- `EPI-D1`
- `EPI-D2`
- `EPI-P1`
- `EPI-P2`
- `EPI-G1`
- `EPI-S1`

Adversarial:

- `ADV-D1`
- `ADV-P1`
- `ADV-P2`
- `ADV-S1`
- `ADV-S2`
- `ADV-T1`

Operator:

- `OPR-D1`
- `OPR-S1`
- `OPR-P1`
- `OPR-S2`
- `OPR-G1`

Builder:

- `BLD-D1`
- `BLD-S1`
- `BLD-P1`
- `BLD-S2`
- `BLD-T1`

### 10.5 Canonical dominant map

The current canonical dominant map is:

- exploration → `EXP-D1`
- synthesis → `SYN-D1`, `SYN-D2`
- epistemic → `EPI-D1`, `EPI-D2`
- adversarial → `ADV-D1`
- operator → `OPR-D1`
- builder → `BLD-D1`

### 10.6 Per-stage support maps

The model layer also defines:

- `SUPPRESSION_BY_STAGE`
- `SHAPES_BY_STAGE`
- `TAILS_BY_STAGE`
- `CANONICAL_FAILURE_IF_OVERUSED`
- `ARTIFACT_HINTS`
- `ARTIFACT_FIELDS`
- `REGIME_PURPOSE_HINTS`
- `COMPLETION_SIGNAL_HINTS`
- `FAILURE_SIGNAL_HINTS`
- `DOMINANT_FAILURE_MAP`
- `FAILURE_SUPPRESSOR_MAP`
- `DOMINANT_SELECTION_RULES`

These maps influence composition, validation, and output-contract enforcement.

---

## 11. Regime composition

The system does not execute directly from a stage label. It composes a `Regime` object.

### 11.1 Regime object

A `Regime` contains:

- `name`
- `stage`
- `dominant_line`
- `suppression_lines`
- `shape_lines`
- `tail_line`
- `rejected_lines`
- `rejection_reasons`
- `likely_failure_if_overused`

It also exposes:

- `all_lines`
- `instruction_block()`
- `render()`

### 11.2 Current composer structure

`RegimeComposer` wraps `GrammarComposer`.

When `RegimeComposer.compose(...)` is called, it first tries `GrammarComposer.compose(...)`. If that fails with an exception, it logs a warning and falls back to internal legacy composition logic contained in the same file.

### 11.3 Grammar composer inputs

The grammar composer receives:

- `stage`
- `risk_profile`
- `handoff_expected`

### 11.4 Grammar composer steps

The grammar composer currently performs the following sequence:

1. choose the effective stage,
2. choose a dominant line,
3. rank likely failures by cost,
4. choose suppression lines,
5. optionally apply extra synthesis break-condition pressure,
6. choose shape lines,
7. choose a tail line,
8. remove hard conflicts,
9. deduplicate,
10. preserve a tail slot under the five-line cap when possible,
11. validate the regime grammar,
12. if invalid, fall back to a minimal regime,
13. build the final `Regime` object.

### 11.5 Line-cap discipline

The active regime is bounded. The composition logic enforces a five-line cap.

### 11.6 Tail-slot behavior

If a requested tail line would exceed the cap, the grammar composer attempts to remove a shape line to preserve the tail. If no removable shape slot exists, the tail is rejected.

### 11.7 Synthesis break-condition pressure

The composer has an explicit synthesis rule that can force `SYN-P2` into selected suppressions when certain risk conditions are present.

Those conditions are:

- `coherence_over_truth`
- `false_unification`
- `high_stakes`
- `abstract_structural_task`

### 11.8 Hard conflict handling

The grammar composer removes hard-conflicting lines before final regime assembly.

### 11.9 Minimal fallback regime

If grammar validation fails after normal selection, the system falls back to a minimal regime made from:

- the dominant line,
- the first compatible suppression line it can find.

### 11.10 Emergency fallback composition

The wrapper `RegimeComposer` also contains internal direct fallback behavior for dominant, suppression, shape, and tail choice in case the grammar composer fails outright.

That fallback logic includes stage-specific choices such as:

- synthesis selecting `SYN-D2` when `sprawl` is in the risk profile,
- epistemic selecting `EPI-D2` when `elegant_theory_drift` is present,
- exploration adding `EXP-S2` for `need_reframing`,
- synthesis suppressing shape lines for `high_stakes`,
- adversarial omitting `ADV-S1` when `single_point_failure` is in the risk profile,
- operator adding `OPR-G1` when `optionality` is in the risk profile,
- epistemic adding `EPI-G1` when `strict` is in the risk profile.

---

## 12. Prompt construction

The executor sends prompts built from regime and contract state rather than a raw task alone.

### 12.1 System prompt contents

The system prompt includes regime-level control information such as:

- active regime name and stage,
- the active instruction lines from the regime,
- output contract shape,
- required artifact type,
- required artifact fields,
- control fields,
- stage purpose,
- completion and failure signaling expectations.

### 12.2 User prompt contents

The user prompt includes execution context such as:

- the task text,
- structural signals,
- risk profile,
- contract reminders.

When a later orchestration step exists and a prior handoff is available, the user prompt also includes a structured **Prior Stage Context** section containing:

- previous-stage dominant frame,
- what is known,
- what remains uncertain,
- active contradictions,
- assumptions in play,
- main risk if continuing,
- minimum useful artifact,
- prior artifact summary.

This section is omitted on the first stage when no prior handoff exists.

### 12.3 Repair prompts

Repair prompts are stage-aware correction prompts built after validation failure.

The prompt builder supports at least three repair modes:

- schema repair,
- semantic repair,
- reduce-genericity repair.

---

## 13. Execution paths

There are two execution paths.

### 13.1 Direct execution path

If runtime fast-path conditions are satisfied, the runtime calls `execute_direct_task(...)`.

This path:

- executes through a direct passthrough regime,
- uses a simpler direct system prompt,
- treats validation as effectively valid for passthrough purposes,
- still updates router state,
- still produces a handoff projection.

The direct path exists inside the same analysis-first planning model as staged execution.

### 13.2 Regime execution path

For staged routed tasks, the runtime calls `RegimeExecutor.execute_once(...)`.

The single-step regime execution cycle is:

1. build the system prompt,
2. build the user prompt,
3. include prior handoff if one exists,
4. call the active model client,
5. validate the raw response,
6. if first-pass validation is invalid, choose a repair mode,
7. build the repair prompt,
8. call the model once more,
9. revalidate,
10. return a `RegimeExecutionResult`.

### 13.3 RegimeExecutionResult fields

A regime execution result contains:

- `task`
- `model`
- `regime_name`
- `stage`
- `system_prompt`
- `user_prompt`
- `raw_response`
- `artifact_text`
- `validation`
- `ollama_meta`

The `ollama_meta` field remains the generic metadata carrier even when the active provider is not Ollama.

### 13.4 Repair trigger behavior

Repair is currently bounded to one retry and is only attempted when first-pass validation is invalid.

If first-pass validation is already valid, repair is skipped.

---

## 14. Output contract and validation

Validation is one of the core control boundaries in the runtime.

The model is not trusted to declare its own output valid. The runtime validates externally.

### 14.1 Structural requirements

Every regime response must parse into JSON and include these top-level keys:

- `regime`
- `purpose`
- `artifact_type`
- `artifact`
- `completion_signal`
- `failure_signal`
- `recommended_next_regime`

### 14.2 Artifact requirements

The `artifact` field must be a JSON object.

It must contain the required fields for the active stage.

The `artifact_type` must match the canonical artifact hint for the active stage.

### 14.3 Control-field validation

The validator checks:

- `purpose` is a non-empty string,
- `completion_signal` is present and stage-appropriate,
- `failure_signal` is present and stage-appropriate,
- `recommended_next_regime` is present and is a valid stage value,
- the `regime` field text matches the active stage.

If the returned regime field does not match the active stage, validation records a control failure describing the mismatch.

### 14.4 Validation profiles

The validator supports four model-control profiles:

- `strict`
- `balanced`
- `lenient`
- `off`

#### strict

- minimum words per field: 3
- Jaccard similarity limit: 0.75
- minimum task-token overlap: 2
- generic filler checks: on
- forbidden generic checks: on
- stage-specific checks: on

#### balanced

- minimum words per field: 2
- Jaccard similarity limit: 0.85
- minimum task-token overlap: 1
- generic filler checks: on
- forbidden generic checks: on
- stage-specific checks: on

#### lenient

- minimum words per field: 1
- Jaccard similarity limit: 0.93
- minimum task-token overlap: 1
- generic filler checks: off
- forbidden generic checks: off
- stage-specific checks: off

#### off

- minimum words per field: 0
- Jaccard similarity limit: 1.01
- minimum task-token overlap: 0
- generic filler checks: off
- forbidden generic checks: off
- stage-specific checks: off

### 14.5 Generic filler detection

The validator flags a defined set of generic filler phrases, including terms such as:

- exploring
- assessing
- understanding
- navigating
- complexity
- various factors
- multiple perspectives
- deeper analysis
- careful consideration
- systemic issues
- abstract dynamics

### 14.6 Forbidden generic nouns

The validator also flags generic domain nouns such as:

- technology
- stakeholders
- innovation
- solution
- industry
- team

### 14.7 Task grounding

The validator checks whether the artifact overlaps with the task text sufficiently to count as grounded.

### 14.8 Structural-signal grounding

The validator checks whether the artifact engages the task’s structural signals, especially in stricter profiles.

### 14.9 Stage-specific semantic checks

The validator includes stage-specific semantic rules.

#### Synthesis checks

For synthesis, it checks things like:

- `organizing_idea` not merely restating `central_claim`,
- `supporting_structure` not being too thin,
- `central_claim` being anchored to the task’s structural signals,
- `organizing_idea` being anchored to the task’s structural signals,
- `key_tensions` being tied to the task’s structural signals,
- `pressure_points` avoiding generic execution language,
- `pressure_points` testing the frame against the original structural signals.

#### Exploration checks

For exploration, it checks things like:

- forbidden generic domain terms,
- whether the artifact visibly engages the task’s structural signals.

### 14.10 Validation output structure

The validator returns a dictionary including fields such as:

- `model_profile`
- `valid_json`
- `is_valid`
- `required_keys_present`
- `artifact_fields_present`
- `missing_keys`
- `missing_artifact_fields`
- `artifact_type_matches`
- `contract_controls_valid`
- `semantic_valid`
- `semantic_failures`
- `parsed`
- `control_failures`

---

## 15. Repair policy

When validation fails, the system selects one repair mode externally.

### 15.1 Repair modes

The current repair modes are:

- schema repair
- semantic repair
- reduce-genericity repair

### 15.2 Selection logic

Repair mode is selected from the validation output, not by asking the model what went wrong.

In practical terms:

- structural or missing-field problems trigger schema repair,
- semantic-quality problems trigger semantic repair,
- generic filler and drift problems trigger reduce-genericity repair.

### 15.3 Repair bounds

The repair pass is bounded to one additional model call after an invalid first pass.

---

## 16. RouterState, handoff, and session state

The system keeps canonical internal state rather than recomputing everything from scratch after execution.

### 16.1 RouterState

`RouterState` is the main internal truth object.

It stores:

- `task_id`
- `task_summary`
- `current_bottleneck`
- `current_regime`
- `runner_up_regime`
- `regime_confidence`
- `dominant_frame`
- `knowns`
- `uncertainties`
- `contradictions`
- `assumptions`
- `risks`
- `stage_goal`
- `switch_trigger`
- `recommended_next_regime`
- `decision_pressure`
- `evidence_quality`
- `recurrence_potential`
- `prior_regimes`
- `orchestration_enabled`
- `max_switches`
- `switches_attempted`
- `switches_executed`
- `orchestration_stop_reason`
- `executed_regime_stages`
- `switch_history`
- `escalation_debug`
- `task_classification`

### 16.2 RouterState methods

RouterState currently supports methods to:

- record a completed regime step,
- record a switch decision,
- apply a dominant frame,
- update inference state,
- resolve a regime for a requested stage.

### 16.3 RegimeStep

Every executed regime step can be recorded as a `RegimeStep` containing:

- regime
- reason entered
- completion-signal seen flag
- failure-signal seen flag
- outcome summary

### 16.4 SwitchDecisionRecord

Every switch decision can be recorded as a `SwitchDecisionRecord` containing:

- switch index
- from stage
- to stage
- switch recommended flag
- switch executed flag
- reason
- switch trigger

### 16.5 Handoff

A `Handoff` is the structured summary passed forward to later execution steps.

It contains:

- `current_bottleneck`
- `dominant_frame`
- `what_is_known`
- `what_remains_uncertain`
- `active_contradictions`
- `assumptions_in_play`
- `main_risk_if_continue`
- `recommended_next_regime`
- `minimum_useful_artifact`
- `recommended_next_regime_full`
- `prior_artifact_summary`

### 16.6 Forward handoff computation

After execution, the runtime computes a forward handoff for subsequent stages.

This computation is deterministic and does not call the model.

The forward handoff is built from:

- the parsed validated artifact,
- the routing decision,
- the current router state,
- the current regime stage,
- completion and failure signals,
- the recommended next regime produced by the model.

### 16.7 Handoff content rules

The handoff content is task-derived rather than router-derived.

Current behavior includes:

- `current_bottleneck` comes from the routing decision’s bottleneck label,
- `what_is_known` contains concise extracted findings rather than raw field dumps,
- `what_remains_uncertain` is extracted from artifact limitations, deferrals, or open questions when present,
- `active_contradictions` reflects task-level tensions or tradeoffs when present,
- `assumptions_in_play` reflects domain assumptions implicit in the artifact,
- `prior_artifact_summary` is a short human-readable summary rather than a JSON dump.

The handoff summary is intended to support stage-to-stage continuity without filling later prompts with router boilerplate.

### 16.8 SessionRecord

A saved session record contains:

- `timestamp_utc`
- `task`
- `risk_profile`
- `model`
- `routing`
- `regime`
- `result`
- `handoff`
- `orchestration`
- `router_state`

---

## 17. State restoration and JSON conversion

The system supports reconstruction from serialized state.

### 17.1 JSON conversion helpers

`state.py` includes helpers such as:

- `to_jsonable(...)`
- `make_record(...)`
- `_stage_from_value(...)`
- `_line_from_payload(...)`
- `_regime_from_payload(...)`
- `_regime_confidence_from_payload(...)`
- `router_state_from_jsonable(...)`

### 17.2 Restoration behavior

The restoration layer can rebuild:

- stages,
- line primitives,
- regimes,
- confidence objects,
- prior regime history,
- switch history,
- the main `RouterState` object.

This allows runs to be saved, inspected, and later resumed or reloaded into runtime state.

---

## 18. Misrouting detection

After execution, the system can inspect whether the current regime is still productive.

The misrouting detector is deterministic and external to the model.

### 18.1 Purpose

The detector answers four questions:

1. is the current regime still productive,
2. is its dominant failure mode active,
3. is switching justified,
4. if so, what next stage is most plausible.

### 18.2 Inputs

It works from:

- the output contract,
- the parsed artifact,
- the validation state,
- the active regime stage.

### 18.3 Cross-stage mismatch detection

If the parsed output claims a different regime than the active stage, that mismatch can be treated as misrouting evidence.

Transition rules also include regime-field mismatch among the switch triggers used for operator-stage control logic.

### 18.4 Stage-specific failure and completion logic

The detector uses stage-specific rules to identify whether a regime has completed useful work or is stuck in its dominant failure pattern.

The exact rule implementations live in `misrouting_rules.py` and are consumed by the misrouting detector and switch orchestrator.

---

## 19. Escalation policy

The escalation policy is a biasing layer that affects routing and switching pressure.

### 19.1 Purpose

It changes how strict or loose the control plane should be under current conditions.

### 19.2 Inputs

It uses information such as:

- router state,
- routing features,
- current regime,
- regime confidence,
- task text,
- misrouting result.

### 19.3 Output

Its output includes fields used in state debugging such as:

- escalation direction,
- justification,
- preferred regime biases,
- switch-pressure adjustment,
- signal-level debug details.

### 19.4 State recording

The planner records escalation debug information into `RouterState.escalation_debug`.

---

## 20. Stop policy

The orchestration loop includes an explicit stop policy.

### 20.1 Purpose

The stop policy determines whether the current run should terminate before switch logic is consulted.

It uses several control gates:

- collapse-override detection (continue if collapse is actively detected),
- whether a valid artifact has been produced,
- whether explicit deliverable pressure means an intermediate artifact is not yet sufficient,
- whether the current stage is at or past the inferred endpoint,
- whether a forward recommendation should defer stopping,
- whether builder entry is justified by recurrence.

### 20.2 StopDecision

`StopPolicy.should_stop(...)` returns a `StopDecision` with:

- `should_stop: bool`
- `reason: str`

### 20.3 Standard stopping rules

The stop policy currently checks conditions equivalent to:

1. **collapse override**: if collapse is detected, do not stop yet (`collapse_override_active`),
2. **builder gate**: if operator is trying to move to builder but `recurrence_potential < 7`, stop with a builder-block reason,
3. **artifact complete**: validation must be valid and include completion/failure control fields in a non-contradictory way,
4. **deliverable pressure gate**: continue when the task asks for explicit final/concrete deliverables but the current stage artifact is still intermediate,
5. **endpoint check**: stop when artifact is complete and stage rank is at or past endpoint,
6. **forward recommendation deferral**: continue if there is a valid forward regime recommendation that still advances toward endpoint.

The loop stops when completion + endpoint logic wins and no defer gate applies.

### 20.4 Builder gate

Builder entry is separately gated.

Builder is only entered when `recurrence_potential >= 7`.

If switch logic recommends builder but recurrence potential is below threshold, the stop policy blocks builder entry and records a reason such as:

- `Builder blocked: recurrence_potential {n} < 7`

### 20.5 Relation to hard orchestration ceiling

The stop policy can terminate the loop early.

`max_switches` remains the hard ceiling.

---

## 21. Switch orchestration

When bounded orchestration is enabled, the system can move from one regime to another through a bounded loop.

### 21.1 Orchestration loop

`CognitiveRouterRuntime.execute(...)` performs one stage first, updates state/handoff, and then enters `SessionRuntime.run_orchestration_loop(...)`.

So the older three-step phrasing:

1. inspect current result,
2. update state from current step,
3. compute forward handoff,

still happens, but it is now split across two locations:

- **before** the loop, right after first-stage execution in `execute(...)`,
- **inside** the loop only after each newly executed switched stage.

Inside the orchestration loop, the runtime currently:

1. evaluates stop policy before switch logic,
2. enforces `max_switches` as a hard limit,
3. runs misrouting detection and escalation evaluation,
4. asks the switch orchestrator whether to switch,
5. applies builder gate checks before executing a recommended switch,
6. applies unrecoverable-invalid-output fallback behavior (attempt exploration fallback unless already in exploration, in which case stop),
7. blocks same-stage and prior-stage re-entry loops (with a bounded collapse-reentry exception),
8. records switch decisions in state history,
9. executes the next regime with prior handoff context,
10. updates state and recomputes forward handoff,
11. repeats until stop-policy stop, switch-limit stop, loop prevention stop, unrecoverable-invalid-output stop, or switch-not-recommended stop.

### 21.2 State limits

The bounded-run state tracks:

- whether orchestration is enabled,
- `max_switches`,
- `switches_attempted`,
- `switches_executed`,
- `orchestration_stop_reason`,
- executed regime stages,
- switch history.

### 21.3 Current defaults

The default maximum number of switches is 2.

### 21.4 Step-local continuity

Regression coverage currently protects stage continuity across multi-step runs.

In practice this means:

- first-stage prompts do not include prior handoff context,
- later-stage prompts do include prior handoff context,
- later stages build on prior work instead of restarting from only the raw task text.

---

## 22. CLI and settings surface

The system includes a CLI and a PowerShell wrapper layer.

### 22.1 Persisted settings

Persisted settings currently cover two groups:

- user settings
- model-control settings

### 22.2 User settings

Current `UserSettings` fields are:

- `provider`
- `model`
- `openai_base_url`
- `openai_api_key_env`
- `use_task_analyzer`
- `task_analyzer_model`
- `debug_routing`
- `bounded_orchestration`
- `max_switches`

### 22.3 Model-control settings

Current `ModelControlSettings` contains:

- `model_profile`

### 22.4 Settings store

`CliSettingsStore` supports:

- `load()`
- `save()`
- `reset_all()`
- `reset_user()`
- `reset_model_controls()`
- `reset()`

### 22.5 Settings format

The current canonical persisted settings format is nested:

- `user`
- `model_controls`

The settings layer also supports backward-compatible loading from older flat shapes by converting them into the current nested format.

---

## 23. Evolution engine

The runtime instantiates an `EvolutionEngine` from `router/evolution/`.

The current codebase includes the evolution namespace and revision-engine surface, but normal runtime flow remains execution-first and control-first.

At the object-model level, evolution-related structures include:

- `FailureLog`
- `RevisionProposal`

These support structured recording of regime failures and candidate revisions.

---

## 24. Current practical boundaries

The current runtime provides:

- analyzer-led route proposal when enabled, with deterministic feature-led fallback,
- classifier-assisted direct-path gating,
- regime composition from typed primitives,
- stage-specific output contracts,
- external validation,
- bounded single-repair correction,
- deterministic state updates,
- deterministic forward handoffs,
- multi-step orchestration with stop-policy control,
- provider-aware model execution,
- serializable session state.

The runtime does not reduce to a single static prompt. It is a staged execution system with external control boundaries.

Its current behavior depends on:

- analyzer output quality,
- validator strictness,
- regime composition quality,
- handoff extraction quality,
- stop-policy correctness,
- switch orchestration quality.

The system currently treats stage outputs as contractual artifacts and uses them as inputs to later control decisions rather than as freeform prose.

---

## 25. Short operational summary

The current Cognitive Router accepts a task, extracts deterministic signals, obtains analyzer-led routing when enabled (or deterministic feature-led routing when not), chooses either direct fast-path execution or staged regime execution, validates model output against an external contract with bounded repair, computes deterministic state and handoff summaries, and may continue through a bounded multi-stage loop that uses stop policy, misrouting detection, escalation policy, invalid-output recovery rules, and switch orchestration to decide whether another stage should run.
