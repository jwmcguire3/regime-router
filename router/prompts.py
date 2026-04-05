from __future__ import annotations

import textwrap
from typing import Dict, List, Optional, Set

from .models import (
    ARTIFACT_FIELDS,
    ARTIFACT_HINTS,
    COMPLETION_SIGNAL_HINTS,
    FAILURE_SIGNAL_HINTS,
    REGIME_PURPOSE_HINTS,
    Regime,
    Stage,
)
from .state import Handoff
from .routing import extract_structural_signals


class PromptBuilder:
    REPAIR_MODE_SCHEMA = "schema_repair"
    REPAIR_MODE_SEMANTIC = "semantic_repair"
    REPAIR_MODE_REDUCE_GENERICITY = "reduce_genericity_repair"

    @staticmethod
    def build_system_prompt(regime: Regime, task_signals: Optional[List[str]] = None, risk_profile: Optional[Set[str]] = None) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_list = "\n".join(f"- {f}" for f in fields)
        task_signals = task_signals or []

        field_rules = PromptBuilder._field_rules(regime.stage)
        signal_lines = (
            "\n".join(f"- {sig.replace('_', ' ')}" for sig in task_signals)
            if task_signals
            else "- no explicit structural signals detected"
        )
        risk_line = ", ".join(sorted(risk_profile or set())) or "none"
        synthesis_constraints = ""
        if regime.stage == Stage.SYNTHESIS:
            synthesis_constraints = textwrap.dedent(
                """
                Synthesis anchoring constraints:
                - Treat the extracted structural signals above as required anchors.
                - Every synthesis field must reinterpret, connect, or test those anchors.
                - If a sentence still works after removing task-specific signals, it is too generic and must be rewritten.
                - pressure_points must falsify or weaken the interpretation, not describe implementation difficulty.
                """
            ).strip()

        return textwrap.dedent(
            f"""
            You are executing exactly one reasoning regime.

            Active regime: {regime.name}
            Stage: {regime.stage.value}

            Follow these instructions exactly:
            {regime.instruction_block()}

            Output requirements:
            - Stay inside the active regime.
            - Do not switch regimes on your own.
            - Return only valid JSON.
            - Do not wrap the JSON in markdown fences.
            - Keep prose concise, concrete, and specific to the input.
            - Do not restate the task label in multiple fields.
            - Each artifact field must contribute non-overlapping information.
            - If the input is too thin to support a strong artifact, say so concretely inside the relevant fields rather than padding with abstractions.

            Top-level JSON keys must be:
            - regime
            - purpose
            - artifact_type
            - artifact
            - completion_signal
            - failure_signal
            - recommended_next_regime

            artifact_type must be exactly: {artifact_name}
            purpose should align with: {REGIME_PURPOSE_HINTS[regime.stage]}
            completion_signal should use this control language: {COMPLETION_SIGNAL_HINTS[regime.stage]}
            failure_signal should use this control language: {FAILURE_SIGNAL_HINTS[regime.stage]}
            recommended_next_regime must be a valid regime stage value
            (exploration|synthesis|epistemic|adversarial|operator|builder)

            artifact must include these keys:
            {field_list}

            Field rules for this stage:
            {field_rules}
            
            Do not introduce external domains, industries, or generic project types.

            All frames must be constructed directly from the structural signals in the input:
            {signal_lines}

            If a frame could apply to any generic project, it is invalid.

            Frames must reinterpret these signals, not replace them.
            Frames must describe what the project *is structurally*, not what it *does*.
            {synthesis_constraints}
            Active risk profile: {risk_line}

            Avoid phrases like:
            - "technology"
            - "solution"
            - "effort"
            - "project to build"

            Prefer:
            - "the project is a system for..."
            - "the project behaves like..."
            - "the project is an attempt to resolve..."
            """
        ).strip()

    @staticmethod
    def build_user_prompt(
        task: str,
        regime: Regime,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        prior_handoff: Optional[Handoff] = None,
    ) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        signals = ", ".join(task_signals or []) or "none"
        risks = ", ".join(sorted(risk_profile or set())) or "none"
        prior_handoff_section = ""
        if prior_handoff is not None:
            previous_regime = prior_handoff.source_regime_name or "unknown"
            source_stage_label = prior_handoff.source_stage.value if prior_handoff.source_stage else "unknown"

            def _bullet_list(items: List[str]) -> str:
                if not items:
                    return "- none"
                return "\n".join(f"- {item}" for item in items)

            element_instructions = []
            if prior_handoff.stable_elements:
                element_instructions.append(
                    f"Stable (build on these, do not re-derive):\n{_bullet_list(prior_handoff.stable_elements)}"
                )
            if prior_handoff.tentative_elements:
                element_instructions.append(
                    f"Tentative (retest if relevant):\n{_bullet_list(prior_handoff.tentative_elements)}"
                )
            if prior_handoff.broken_elements:
                element_instructions.append(
                    f"Broken (reopen these):\n{_bullet_list(prior_handoff.broken_elements)}"
                )
            if prior_handoff.do_not_relitigate:
                element_instructions.append(
                    f"Do not relitigate unless broken: {', '.join(prior_handoff.do_not_relitigate)}"
                )
            if not element_instructions:
                element_instructions = ["Build on this context. Do not re-derive what is already established."]

            element_section = "\n\n".join(element_instructions)
            downstream_discipline = textwrap.dedent(
                """
                Downstream execution discipline:
                - Preserve the original task and use this context to constrain how you solve it.
                - Treat stable elements as settled unless a concrete break condition appears in this stage.
                - Reopen only the explicitly broken elements and repair them with new task-grounded evidence.
                - Avoid re-deriving do-not-relitigate elements unless you can show they are broken now.
                - Your response must directly advance the minimum useful artifact above.
                """
            ).strip()

            prior_handoff_section = textwrap.dedent(
                f"""

                ## Prior Stage Context
                The previous regime ({previous_regime}, stage: {source_stage_label}) produced the following assessment:

                Dominant frame: {prior_handoff.dominant_frame}
                What is known:
                {_bullet_list(prior_handoff.what_is_known)}
                What remains uncertain:
                {_bullet_list(prior_handoff.what_remains_uncertain)}
                Active contradictions:
                {_bullet_list(prior_handoff.active_contradictions)}
                Assumptions in play:
                {_bullet_list(prior_handoff.assumptions_in_play)}
                Main risk if continuing: {prior_handoff.main_risk_if_continue}
                Minimum useful artifact: {prior_handoff.minimum_useful_artifact}
                {f"Prior artifact summary: {prior_handoff.prior_artifact_summary}" if prior_handoff.prior_artifact_summary else ""}

                {element_section}

                {downstream_discipline}
                """
            ).rstrip()
        return textwrap.dedent(
            f"""
            Task:
            {task}

            Structural signals:
            {signals}

            Risk profile:
            {risks}
            {prior_handoff_section}

          Return one JSON object with exactly these top-level keys:
          regime, purpose, artifact_type, artifact, completion_signal, failure_signal, recommended_next_regime

            Use only the information in the task.
            Do not invent missing specifics.
            If the task is abstract or underspecified, reflect that explicitly instead of using generic filler.
            """
        ).strip()

    @staticmethod
    def _field_rules(stage: Stage) -> str:
        rules = {
            Stage.SYNTHESIS: """
            - central_claim: one-sentence structural interpretation that explicitly uses at least two task signals (directly or close transforms)
            - organizing_idea: explain the mechanism that generates central_claim from the signals; do not restate central_claim
            - key_tensions: 2-4 pressure pairs, each rooted in specific task signals
            - supporting_structure: 2-4 signal-anchored observations tied to exact extracted signals, not broad project language
            - pressure_points: frame-break conditions that would falsify or materially weaken the interpretation; not execution risks
            """,
            Stage.EPISTEMIC: """
            - supported_claims: claims directly supported by the input
            - plausible_but_unproven: ideas that fit the input but are not established
            - contradictions: tensions or mismatches that remain unresolved
            - omitted_due_to_insufficient_support: things intentionally not claimed
            - decision_relevant_conclusions: conclusions that most affect the next decision
            """,
            Stage.ADVERSARIAL: """
            - top_destabilizers: the most decision-changing objections
            - hidden_assumptions: assumptions the frame depends on
            - break_conditions: specific conditions that would cause failure
            - survivable_revisions: best fixes that preserve what still works
            - residual_risks: risks that remain even after revision
            """,
            Stage.OPERATOR: """
            - decision: one clear choice
            - rationale: why this choice wins now under current constraints
            - tradeoff_accepted: what is being knowingly sacrificed
            - next_actions: immediate executable steps
            - fallback_trigger: what condition should cause reconsideration
            - review_point: when or under what condition to review the decision
            """,
            Stage.EXPLORATION: """
            - candidate_frames: 2-5 structurally distinct framings
            - selection_criteria: what should determine which frame wins
            - unresolved_axes: what key unknowns still separate the frames
            """,
            Stage.BUILDER: """
            - reusable_pattern: the core repeatable pattern
            - modules: main components
            - interfaces: how the modules connect
            - required_inputs: what the system needs
            - produced_outputs: what it creates
            - implementation_sequence: build order
            - compounding_path: how this becomes more reusable over time
            """,
            
        }
        return textwrap.dedent(rules.get(stage, "")).strip()
        
    @staticmethod
    def build_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
        repair_mode: str = REPAIR_MODE_SEMANTIC,
    ) -> str:
        if repair_mode == PromptBuilder.REPAIR_MODE_SCHEMA:
            return PromptBuilder._build_schema_repair_prompt(task, regime, invalid_output, validation)
        if repair_mode == PromptBuilder.REPAIR_MODE_REDUCE_GENERICITY:
            return PromptBuilder._build_reduce_genericity_repair_prompt(
                task,
                regime,
                invalid_output,
                validation,
                task_signals=task_signals,
            )
        return PromptBuilder._build_semantic_repair_prompt(
            task,
            regime,
            invalid_output,
            validation,
            task_signals=task_signals,
        )

    @staticmethod
    def _build_schema_repair_prompt(task: str, regime: Regime, invalid_output: str, validation: Dict[str, object]) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        required_fields = ", ".join(ARTIFACT_FIELDS[regime.stage])
        missing_keys = validation.get("missing_keys", [])
        missing_fields = validation.get("missing_artifact_fields", [])
        parse_error = validation.get("error", "n/a")

        return textwrap.dedent(
            f"""
            Your previous output failed structural/schema validation.

            Original task:
            {task}

            Required top-level keys: regime, purpose, artifact_type, artifact, completion_signal, failure_signal, recommended_next_regime
            artifact_type must be exactly: {artifact_name}
            Required artifact fields: {required_fields}

            Schema failures:
            - parse_or_structure_error: {parse_error}
            - missing_keys: {missing_keys}
            - missing_artifact_fields: {missing_fields}

            Previous invalid output:
            {invalid_output}

            Repair rules:
            - Return only valid JSON.
            - Keep existing content where possible; perform minimal structural edits.
            - Do not add commentary or markdown fences.
            - completion_signal should fit this regime language: {COMPLETION_SIGNAL_HINTS[regime.stage]}
            - failure_signal should fit this regime language: {FAILURE_SIGNAL_HINTS[regime.stage]}
            """
        ).strip()

    @staticmethod
    def _build_semantic_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
    ) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_text = ", ".join(fields)
        extracted_signals = task_signals or extract_structural_signals(task)
        signal_text = ", ".join(extracted_signals) if extracted_signals else "none"

        failures = validation.get("semantic_failures", [])
        failure_text = "\n".join(f"- {f}" for f in failures) if failures else "- output failed validation"
        failed_fields = PromptBuilder._failed_fields(failures, fields)
        failed_field_text = ", ".join(failed_fields) if failed_fields else "unknown (repair only where failures apply)"

        return textwrap.dedent(
            f"""
            Your previous output failed semantic validation.

            Original task:
            {task}

            Extracted structural signals:
            {signal_text}

            Required artifact:
            {artifact_name}

            Required fields:
            {field_text}

            Validation failures:
            {failure_text}

            Fields requiring rewrite:
            {failed_field_text}

            Previous invalid output:
            {invalid_output}

            Repair instructions:
            - Return only valid JSON with the same top-level schema.
            - Make minimal edits.
            - Rewrite only fields that failed validation; keep non-failed fields unchanged.
            - Keep central_claim and organizing_idea distinct (no paraphrase duplication).
            - Treat pressure_points as frame-break conditions that can falsify or materially weaken the frame.
            - Do not introduce external domains, industries, teams, stakeholders, or generic project nouns.
            - Do not explain your edits.
            - Do not include markdown fences.
            """
        ).strip()

    @staticmethod
    def _build_reduce_genericity_repair_prompt(
        task: str,
        regime: Regime,
        invalid_output: str,
        validation: Dict[str, object],
        *,
        task_signals: Optional[List[str]] = None,
    ) -> str:
        artifact_name = ARTIFACT_HINTS[regime.stage]
        fields = ARTIFACT_FIELDS[regime.stage]
        field_text = ", ".join(fields)
        extracted_signals = task_signals or extract_structural_signals(task)
        signal_text = ", ".join(extracted_signals) if extracted_signals else "none"
        failures = validation.get("semantic_failures", [])
        failure_text = "\n".join(f"- {f}" for f in failures) if failures else "- output is too generic"
        failed_fields = PromptBuilder._failed_fields(failures, fields)
        failed_field_text = ", ".join(failed_fields) if failed_fields else "unknown (repair only where failures apply)"

        return textwrap.dedent(
            f"""
            Your previous output is structurally valid but too generic.

            Original task:
            {task}

            Extracted structural signals:
            {signal_text}

            Required artifact:
            {artifact_name}

            Required fields:
            {field_text}

            Genericity failures:
            {failure_text}

            Fields requiring rewrite:
            {failed_field_text}

            Previous invalid output:
            {invalid_output}

            Reduce-genericity repair instructions:
            - Return only valid JSON with the same top-level schema.
            - Make minimal edits and rewrite only failed fields.
            - Replace generic filler with signal-anchored statements grounded in the original task.
            - Do not introduce any external domain nouns (technology, industry, solution, team, stakeholders, innovation, etc.).
            - Keep central_claim and organizing_idea distinct.
            - pressure_points must be frame-break tests, not execution concerns.
            - Do not explain your edits or use markdown fences.
            """
        ).strip()

    @staticmethod
    def _failed_fields(failures: List[str], fields: List[str]) -> List[str]:
        detected: List[str] = []
        for failure in failures:
            for field in fields:
                if failure.startswith(f"{field} ") and field not in detected:
                    detected.append(field)
        return detected
