from __future__ import annotations

import json
import re
import textwrap
from typing import Dict, List, Optional, Tuple

from .llm import ModelClient
from .models import (
    ControlAuthority,
    PolicyEvent,
    RegimeConfidenceResult,
    RoutingDecision,
    Severity,
    Stage,
    TaskAnalyzerOutput,
)


class TaskAnalyzer:
    def __init__(self, model_client: ModelClient, model: str) -> None:
        self.model_client = model_client
        self.model = model
        self.last_error_summary: Optional[str] = None

    def analyze(self, task: str) -> Optional[TaskAnalyzerOutput]:
        self.last_error_summary = None
        response = self.model_client.generate(
            model=self.model,
            system=self._build_system_prompt(),
            prompt=self._build_user_prompt(task),
            stream=False,
            temperature=0.0,
            num_predict=500,
        )
        raw_text = str(response.get("response", "")).strip()
        parsed, parse_error = self._parse_response_payload(raw_text)
        initial_parse_error = parse_error
        if parsed is None:
            repaired_text = self._attempt_json_repair(raw_text)
            if repaired_text:
                parsed, parse_error = self._parse_response_payload(repaired_text)
        if parsed is None:
            self.last_error_summary = initial_parse_error or parse_error or "Analyzer returned invalid/non-JSON output."
            return None
        validated = self._validate_output(parsed)
        if validated is not None:
            return validated

        repaired_payload = self._repair_missing_fields(parsed, raw_text)
        if repaired_payload is not None:
            repaired_validated = self._validate_output(repaired_payload)
            if repaired_validated is not None:
                self.last_error_summary = "Analyzer missing-field repair attempted and succeeded."
                return repaired_validated
            self.last_error_summary = "Analyzer missing-field repair attempted but repaired payload remained invalid."
            return None

        self.last_error_summary = "Analyzer output invalid; missing-field repair not applicable or failed."
        return None

    def _build_system_prompt(self) -> str:
        return textwrap.dedent(
            """
            You are a constrained task analyzer for a deterministic router.
            Return ONLY a strict JSON object and no markdown.
            Do not produce narrative outside JSON.
            Your job is to provide typed analysis signals, not final routing.

            Required JSON schema:
            {
              "bottleneck_label": "string",
              "candidate_regimes": ["exploration|synthesis|epistemic|adversarial|operator|builder"],
              "stage_scores": {
                "exploration": number,
                "synthesis": number,
                "epistemic": number,
                "adversarial": number,
                "operator": number,
                "builder": number
              },
              "structural_signals": ["string"],
              "decision_pressure": integer 0-10,
              "fragility_pressure": integer 0-10,
              "possibility_space_need": integer 0-10,
              "synthesis_pressure": integer 0-10,
              "evidence_quality": integer 0-10,
              "recurrence_potential": integer 0-10,
              "confidence": number 0-1,
              "rationale": "short string",
              "risk_tags": ["string"],
              "likely_endpoint_regime": "exploration|synthesis|epistemic|adversarial|operator|builder",
              "endpoint_confidence": number 0-1
            }

            Estimate which regime will produce the minimum useful artifact for this task.
            Most tasks terminate at operator. Builder is only appropriate when the task
            explicitly involves creating reusable, recurring infrastructure — not when
            recurrence is merely hypothetical. If unsure, default to operator.

            Explain in 1-2 sentences why this stage is the current bottleneck, referencing specific
            features of the input task. Do not use generic language like 'best fit' or 'most suitable'.
            """
        ).strip()

    def propose_route(self, task: str) -> RoutingDecision:
        analyzer_result = self.analyze(task)
        return self.decision_from_analysis(
            task=task,
            analyzer_result=analyzer_result,
        )

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
    ) -> RoutingDecision:
        if analyzer_result is None:
            summary = self.last_error_summary or "Analyzer failed without a detailed error summary."
            return RoutingDecision(
                bottleneck=task,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="Analyzer unavailable; exploration is the safest low-confidence fallback.",
                switch_trigger="Switch when one frame becomes clearly more decision-relevant than alternatives.",
                endpoint_confidence=0.3,
                confidence=RegimeConfidenceResult(
                    level=Severity.LOW.value,
                    rationale="Analyzer failed; defaulting to conservative exploration fallback.",
                    top_stage_score=0,
                    runner_up_score=0,
                    score_gap=0,
                    nontrivial_stage_count=0,
                    weak_lexical_dependence=True,
                    structural_feature_state="sparse",
                ),
                analyzer_enabled=True,
                analyzer_used=True,
                analyzer_summary=summary,
                inference_quality="analyzer_led",
            )

        ranked = sorted(
            analyzer_result.stage_scores.items(),
            key=lambda item: (-item[1], list(Stage).index(item[0])),
        )
        primary = ranked[0][0] if ranked else Stage.EXPLORATION
        runner_up = next((stage for stage, _ in ranked if stage != primary), Stage.SYNTHESIS)
        pre_policy_primary_regime = primary
        pre_policy_runner_up_regime = runner_up

        primary, runner_up, policy_warnings, policy_actions, policy_events = self._apply_routing_policy(
            primary=primary,
            runner_up=runner_up,
            analyzer_result=analyzer_result,
        )

        if primary == runner_up:
            runner_up = Stage.SYNTHESIS if primary != Stage.SYNTHESIS else Stage.EXPLORATION

        confidence_score = analyzer_result.confidence
        if confidence_score >= 0.8:
            confidence_level = Severity.HIGH.value
        elif confidence_score >= 0.5:
            confidence_level = Severity.MEDIUM.value
        else:
            confidence_level = Severity.LOW.value

        top_score = int(round(ranked[0][1])) if ranked else 0
        runner_up_score = int(round(next((score for stage, score in ranked if stage == runner_up), 0.0)))
        nontrivial_stage_count = sum(1 for _, score in ranked if score > 0)
        confidence = RegimeConfidenceResult(
            level=confidence_level,
            rationale=f"Analyzer confidence={confidence_score:.2f}.",
            top_stage_score=top_score,
            runner_up_score=runner_up_score,
            score_gap=max(0, top_score - runner_up_score),
            nontrivial_stage_count=nontrivial_stage_count,
            weak_lexical_dependence=False,
            structural_feature_state="rich" if analyzer_result.structural_signals else "sparse",
        )

        stage_reasons = {
            Stage.OPERATOR: (
                "Decision-intent pressure is currently the bottleneck.",
                "Switch when the decision, tradeoffs, and fallback are explicit.",
            ),
            Stage.EPISTEMIC: (
                "Evidence quality and uncertainty calibration are currently the bottleneck.",
                "Switch when supported and unsupported claims are separated clearly enough to decide.",
            ),
            Stage.SYNTHESIS: (
                "The task needs a coherent organizing frame before action.",
                "Switch when one frame becomes dominant and exclusion criteria are explicit.",
            ),
            Stage.EXPLORATION: (
                "The task still needs broader option-space coverage before narrowing.",
                "Switch when distinct frames are surfaced and one starts to dominate.",
            ),
            Stage.BUILDER: (
                "The bottleneck is converting patterns into reusable operating structure.",
                "Switch when modules, repeatability, and implementation order are concretely defined.",
            ),
            Stage.ADVERSARIAL: (
                "The bottleneck is latent fragility that must be stress-tested.",
                "Switch when key failure modes are surfaced and mitigation revisions are clear.",
            ),
        }
        why_primary_wins_now, switch_trigger = stage_reasons[primary]

        stage_progression = list(Stage)
        likely_endpoint = analyzer_result.likely_endpoint_regime
        endpoint_confidence = analyzer_result.endpoint_confidence
        builder_endpoint_softened = False

        if (
            likely_endpoint == Stage.BUILDER
            and analyzer_result.recurrence_potential == 0
            and analyzer_result.confidence < 0.8
        ):
            likely_endpoint = Stage.OPERATOR
            builder_endpoint_softened = True
            policy_actions.append("builder endpoint softened to operator")
            policy_events.append(
                PolicyEvent(
                    rule_name="builder_endpoint_softened_to_operator",
                    authority=ControlAuthority.SOFT_GUARDRAIL.value,
                    consumed_features=["likely_endpoint_regime", "recurrence_potential", "confidence"],
                    action="soften_endpoint_to_operator",
                    detail=(
                        "builder endpoint softened to operator: likely_endpoint=builder, "
                        f"recurrence_potential={analyzer_result.recurrence_potential}, "
                        f"analyzer_confidence={analyzer_result.confidence:.2f}"
                    ),
                )
            )

        if (
            stage_progression.index(likely_endpoint) < stage_progression.index(primary)
            and not builder_endpoint_softened
        ):
            prior_endpoint = likely_endpoint
            likely_endpoint = primary
            policy_actions.append("endpoint proposed before primary regime; clamped to primary")
            policy_events.append(
                PolicyEvent(
                    rule_name="endpoint_clamped_to_primary",
                    authority=ControlAuthority.HARD_VETO.value,
                    consumed_features=["likely_endpoint_regime", "primary_regime"],
                    action="clamp_endpoint_to_primary",
                    detail=(
                        "endpoint proposed before primary regime; clamped to primary: "
                        f"endpoint_before={prior_endpoint.value}, primary={primary.value}"
                    ),
                )
            )

        summary_parts = [
            f"Analyzer confidence={confidence_score:.2f}",
            f"rationale={analyzer_result.rationale}",
            f"candidates={[stage.value for stage in analyzer_result.candidate_regimes]}",
            f"endpoint={likely_endpoint.value}@{endpoint_confidence:.2f}",
        ]
        summary_parts.extend(policy_warnings)
        summary_parts.extend(policy_actions)
        return RoutingDecision(
            bottleneck=analyzer_result.bottleneck_label,
            primary_regime=primary,
            runner_up_regime=runner_up,
            why_primary_wins_now=why_primary_wins_now,
            switch_trigger=switch_trigger,
            pre_policy_primary_regime=pre_policy_primary_regime,
            pre_policy_runner_up_regime=pre_policy_runner_up_regime,
            likely_endpoint_regime=likely_endpoint.value,
            endpoint_confidence=endpoint_confidence,
            confidence=confidence,
            analyzer_enabled=True,
            analyzer_used=True,
            analyzer_summary="; ".join(summary_parts),
            inference_quality="analyzer_led",
            policy_warnings=policy_warnings,
            policy_actions=policy_actions,
            policy_events=policy_events,
        )

    def _apply_routing_policy(
        self,
        *,
        primary: Stage,
        runner_up: Stage,
        analyzer_result: TaskAnalyzerOutput,
    ) -> tuple[Stage, Stage, list[str], list[str], list[PolicyEvent]]:
        policy_warnings: list[str] = []
        policy_actions: list[str] = []
        policy_events: list[PolicyEvent] = []

        if (
            primary == Stage.OPERATOR
            and analyzer_result.decision_pressure == 0
            and analyzer_result.confidence < 0.8
        ):
            policy_warnings.append("operator support weak; soft guardrail only")
            policy_events.append(
                PolicyEvent(
                    rule_name="operator_weak_support",
                    authority=ControlAuthority.ADVISORY_ONLY.value,
                    consumed_features=["decision_pressure"],
                    action="advisory_warning",
                    detail="operator proposed without decision_pressure support",
                )
            )
            if runner_up not in {Stage.EXPLORATION, primary}:
                prior_runner_up = runner_up
                runner_up = Stage.EXPLORATION
                policy_actions.append("runner-up softened toward exploration")
                policy_events.append(
                    PolicyEvent(
                        rule_name="runner_up_softened_toward_exploration",
                        authority=ControlAuthority.SOFT_GUARDRAIL.value,
                        consumed_features=["primary_regime", "runner_up_regime", "decision_pressure"],
                        action="soften_runner_up",
                        detail=f"runner_up changed from {prior_runner_up.value} to exploration due to weak operator support",
                    )
                )

        if (
            primary == Stage.BUILDER
            and analyzer_result.recurrence_potential == 0
            and analyzer_result.confidence < 0.8
        ):
            policy_warnings.append("builder support weak; advisory only")
            policy_events.append(
                PolicyEvent(
                    rule_name="builder_weak_support",
                    authority=ControlAuthority.ADVISORY_ONLY.value,
                    consumed_features=["recurrence_potential"],
                    action="advisory_warning",
                    detail="builder proposed without recurrence_potential support",
                )
            )

        if (
            primary == Stage.ADVERSARIAL
            and analyzer_result.fragility_pressure == 0
            and analyzer_result.confidence < 0.8
        ):
            policy_warnings.append("adversarial support weak; advisory only")
            policy_events.append(
                PolicyEvent(
                    rule_name="adversarial_weak_support",
                    authority=ControlAuthority.ADVISORY_ONLY.value,
                    consumed_features=["fragility_pressure"],
                    action="advisory_warning",
                    detail="adversarial proposed without fragility_pressure support",
                )
            )

        if runner_up == primary:
            runner_up = Stage.SYNTHESIS if primary != Stage.SYNTHESIS else Stage.EXPLORATION

        return primary, runner_up, policy_warnings, policy_actions, policy_events

    def _build_user_prompt(self, task: str) -> str:
        return textwrap.dedent(
            f"""
            Analyze this task bottleneck and return schema-compliant JSON only.

            task:
            {task}
            """
        ).strip()

    @staticmethod
    def _strip_markdown_code_fences(raw_text: str) -> Tuple[str, bool]:
        text = raw_text.strip()
        fenced = bool(re.match(r"^```[\w-]*\s*", text))
        if not fenced:
            return text, False
        text = re.sub(r"^```[\w-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return text.strip(), True

    @staticmethod
    def _extract_first_json_object(raw_text: str) -> Optional[str]:
        start = raw_text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(raw_text)):
            ch = raw_text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw_text[start : idx + 1]
        return None

    def _parse_response_payload(self, raw_text: str) -> Tuple[Optional[object], Optional[str]]:
        text = raw_text.strip()
        if not text:
            return None, "Analyzer non-JSON response: empty output."
        try:
            return json.loads(text), None
        except json.JSONDecodeError:
            pass

        stripped, had_fences = self._strip_markdown_code_fences(text)
        if had_fences:
            try:
                return json.loads(stripped), None
            except json.JSONDecodeError:
                pass

        extracted = self._extract_first_json_object(stripped if had_fences else text)
        if extracted:
            try:
                return json.loads(extracted), None
            except json.JSONDecodeError:
                error_prefix = "Analyzer malformed JSON after extraction"
                if had_fences:
                    error_prefix += " from fenced JSON"
                return None, f"{error_prefix}."

        if had_fences:
            return None, "Analyzer fenced JSON did not contain a valid JSON object."
        if ("{" in text) != ("}" in text):
            return None, "Analyzer malformed JSON response."
        if "{" not in text and "}" not in text:
            return None, "Analyzer non-JSON response: no JSON object found."
        return None, "Analyzer malformed JSON response."

    def _attempt_json_repair(self, raw_text: str) -> str:
        repair_prompt = textwrap.dedent(
            f"""
            Convert the following analyzer output into valid JSON only.
            Return ONLY one JSON object. Do not include markdown or commentary.

            previous_output:
            {raw_text}
            """
        ).strip()
        try:
            repair_response = self.model_client.generate(
                model=self.model,
                system=self._build_system_prompt(),
                prompt=repair_prompt,
                stream=False,
                temperature=0.0,
                num_predict=500,
            )
        except Exception:
            return ""
        return str(repair_response.get("response", "")).strip()

    def _repair_missing_fields(self, raw_payload: object, raw_text: str) -> Optional[dict]:
        if not isinstance(raw_payload, dict):
            return None

        required = {
            "bottleneck_label",
            "candidate_regimes",
            "stage_scores",
            "structural_signals",
            "decision_pressure",
            "fragility_pressure",
            "possibility_space_need",
            "synthesis_pressure",
            "evidence_quality",
            "recurrence_potential",
            "confidence",
            "rationale",
        }
        missing_top_level = sorted(required - set(raw_payload.keys()))
        missing_stage_scores: List[str] = []

        candidate_regimes = raw_payload.get("candidate_regimes")
        if candidate_regimes is not None:
            if not isinstance(candidate_regimes, list):
                return None
            for item in candidate_regimes:
                if not isinstance(item, str):
                    return None
                try:
                    Stage(item)
                except ValueError:
                    return None

        scores_raw = raw_payload.get("stage_scores")
        if scores_raw is not None:
            if not isinstance(scores_raw, dict):
                return None
            for key, value in scores_raw.items():
                if key not in {stage.value for stage in Stage}:
                    return None
                if not isinstance(value, (int, float)):
                    return None
            missing_stage_scores = [stage.value for stage in Stage if stage.value not in scores_raw]

        if not missing_top_level and not missing_stage_scores:
            return None

        repair_prompt = textwrap.dedent(
            f"""
            Repair this analyzer JSON object by adding missing required fields only.
            Preserve every existing field/value exactly when already present and valid.
            Add only the fields listed below that are missing.
            Return exactly one valid JSON object.
            No markdown. No commentary.

            Missing top-level required fields: {json.dumps(missing_top_level)}
            Missing stage_scores entries: {json.dumps(missing_stage_scores)}

            Original analyzer output:
            {raw_text}
            """
        ).strip()
        try:
            repair_response = self.model_client.generate(
                model=self.model,
                system=self._build_system_prompt(),
                prompt=repair_prompt,
                stream=False,
                temperature=0.0,
                num_predict=500,
            )
        except Exception:
            return None

        repaired_text = str(repair_response.get("response", "")).strip()
        parsed_repaired, _ = self._parse_response_payload(repaired_text)
        if not isinstance(parsed_repaired, dict):
            return None
        return parsed_repaired

    @staticmethod
    def _validate_output(payload: object) -> Optional[TaskAnalyzerOutput]:
        if not isinstance(payload, dict):
            return None
        required = {
            "bottleneck_label",
            "candidate_regimes",
            "stage_scores",
            "structural_signals",
            "decision_pressure",
        }
        if not required.issubset(payload.keys()):
            return None
        if not isinstance(payload["bottleneck_label"], str) or not payload["bottleneck_label"].strip():
            return None
        candidates_raw = payload["candidate_regimes"]
        if not isinstance(candidates_raw, list):
            return None
        candidate_regimes: List[Stage] = []
        for item in candidates_raw:
            if not isinstance(item, str):
                return None
            try:
                candidate_regimes.append(Stage(item))
            except ValueError:
                return None
        if not candidate_regimes:
            return None

        scores_raw = payload["stage_scores"]
        if not isinstance(scores_raw, dict):
            return None
        stage_scores: Dict[Stage, float] = {}
        for stage in Stage:
            value = scores_raw.get(stage.value)
            if not isinstance(value, (int, float)):
                return None
            stage_scores[stage] = float(value)

        signals_raw = payload["structural_signals"]
        if not isinstance(signals_raw, list) or any(not isinstance(s, str) for s in signals_raw):
            return None

        int_fields = (
            "decision_pressure",
            "fragility_pressure",
            "possibility_space_need",
            "synthesis_pressure",
            "evidence_quality",
            "recurrence_potential",
        )
        for field_name in int_fields:
            value = payload.get(field_name, 0)
            if not isinstance(value, int) or value < 0 or value > 10:
                return None

        confidence = payload.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            return None

        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            return None

        risk_tags_raw = payload.get("risk_tags", [])
        if not isinstance(risk_tags_raw, list) or any(not isinstance(tag, str) for tag in risk_tags_raw):
            return None

        endpoint_stage_raw = payload.get("likely_endpoint_regime", Stage.OPERATOR.value)
        if not isinstance(endpoint_stage_raw, str):
            return None
        try:
            likely_endpoint_regime = Stage(endpoint_stage_raw)
        except ValueError:
            return None

        endpoint_confidence = payload.get("endpoint_confidence", 0.7)
        if not isinstance(endpoint_confidence, (int, float)) or endpoint_confidence < 0 or endpoint_confidence > 1:
            return None

        return TaskAnalyzerOutput(
            bottleneck_label=payload["bottleneck_label"].strip(),
            candidate_regimes=candidate_regimes,
            stage_scores=stage_scores,
            structural_signals=[s for s in signals_raw if s.strip()],
            decision_pressure=payload["decision_pressure"],
            fragility_pressure=payload.get("fragility_pressure", 0),
            possibility_space_need=payload.get("possibility_space_need", 0),
            synthesis_pressure=payload.get("synthesis_pressure", 0),
            evidence_quality=payload.get("evidence_quality", 0),
            recurrence_potential=payload.get("recurrence_potential", 0),
            confidence=float(confidence),
            rationale=rationale.strip(),
            risk_tags=[tag.strip() for tag in risk_tags_raw if tag.strip()],
            likely_endpoint_regime=likely_endpoint_regime,
            endpoint_confidence=float(endpoint_confidence),
        )


# ============================================================
# Prompt builder + validator
