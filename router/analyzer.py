from __future__ import annotations

import json
import re
import textwrap
from typing import Dict, List, Optional, Set, Tuple

from .llm import ModelClient
from .models import RegimeConfidenceResult, RoutingDecision, RoutingFeatures, Severity, Stage, TaskAnalyzerOutput


class TaskAnalyzer:
    def __init__(self, model_client: ModelClient, model: str) -> None:
        self.model_client = model_client
        self.model = model
        self.last_error_summary: Optional[str] = None

    def analyze(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
        classifier_signal: Optional[Dict[str, object]] = None,
    ) -> Optional[TaskAnalyzerOutput]:
        self.last_error_summary = None
        response = self.model_client.generate(
            model=self.model,
            system=self._build_system_prompt(),
            prompt=self._build_user_prompt(task, routing_features, task_signals, risk_profile, classifier_signal),
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
              "evidence_quality": integer 0-10,
              "recurrence_potential": integer 0-10,
              "confidence": number 0-1,
              "rationale": "short string",
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

    def propose_route(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
        classifier_signal: Optional[Dict[str, object]] = None,
    ) -> RoutingDecision:
        analyzer_result = self.analyze(task, routing_features, task_signals, risk_profile, classifier_signal)
        return self.decision_from_analysis(
            task=task,
            analyzer_result=analyzer_result,
            routing_features=routing_features,
        )

    def decision_from_analysis(
        self,
        *,
        task: str,
        analyzer_result: Optional[TaskAnalyzerOutput],
        routing_features: RoutingFeatures,
    ) -> RoutingDecision:
        if analyzer_result is None:
            summary = self.last_error_summary or "Analyzer failed without a detailed error summary."
            return RoutingDecision(
                bottleneck=task,
                primary_regime=Stage.EXPLORATION,
                runner_up_regime=Stage.SYNTHESIS,
                why_primary_wins_now="Analyzer unavailable; exploration is the safest low-confidence fallback.",
                switch_trigger="Switch when one frame becomes clearly more decision-relevant than alternatives.",
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
            )

        ranked = sorted(
            analyzer_result.stage_scores.items(),
            key=lambda item: (-item[1], list(Stage).index(item[0])),
        )
        primary = ranked[0][0] if ranked else Stage.EXPLORATION
        runner_up = next((stage for stage, _ in ranked if stage != primary), Stage.SYNTHESIS)

        notes: List[str] = []
        has_decision_markers = bool(routing_features.detected_markers.get("decision_tradeoff_commitment"))
        if primary == Stage.OPERATOR and routing_features.decision_pressure == 0 and not has_decision_markers:
            primary = Stage.EXPLORATION
            notes.append("operator proposed without decision evidence; demoted to exploration")
        if primary == Stage.BUILDER and routing_features.recurrence_potential == 0:
            primary = Stage.EXPLORATION
            notes.append("builder proposed without recurrence potential; demoted to exploration")
        if primary == Stage.ADVERSARIAL and routing_features.fragility_pressure == 0:
            primary = Stage.EXPLORATION
            notes.append("adversarial proposed without fragility pressure; demoted to exploration")

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
            structural_feature_state="rich" if routing_features.structural_signals else "sparse",
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

        if likely_endpoint == Stage.BUILDER and analyzer_result.recurrence_potential == 0:
            likely_endpoint = Stage.OPERATOR
            notes.append("builder endpoint proposed without recurrence potential; demoted to operator")

        if stage_progression.index(likely_endpoint) < stage_progression.index(primary):
            likely_endpoint = primary
            notes.append("endpoint proposed before primary regime; clamped to primary")

        summary_parts = [
            f"Analyzer confidence={confidence_score:.2f}",
            f"rationale={analyzer_result.rationale}",
            f"candidates={[stage.value for stage in analyzer_result.candidate_regimes]}",
            f"endpoint={likely_endpoint.value}@{endpoint_confidence:.2f}",
        ]
        summary_parts.extend(notes)
        return RoutingDecision(
            bottleneck=analyzer_result.bottleneck_label,
            primary_regime=primary,
            runner_up_regime=runner_up,
            why_primary_wins_now=why_primary_wins_now,
            switch_trigger=switch_trigger,
            likely_endpoint_regime=likely_endpoint.value,
            endpoint_confidence=endpoint_confidence,
            confidence=confidence,
            analyzer_enabled=True,
            analyzer_used=True,
            analyzer_summary="; ".join(summary_parts),
        )

    def _build_user_prompt(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
        classifier_signal: Optional[Dict[str, object]] = None,
    ) -> str:
        classifier_route_type = "unknown"
        classifier_confidence = "n/a"
        classifier_source = "n/a"
        if classifier_signal is not None:
            classifier_route_type = str(classifier_signal.get("route_type", classifier_route_type))
            confidence = classifier_signal.get("confidence")
            classifier_confidence = f"{confidence}" if confidence is not None else classifier_confidence
            classifier_source = str(classifier_signal.get("classification_source", classifier_source))

        feature_blob = {
            "structural_signals": routing_features.structural_signals,
            "decision_pressure": routing_features.decision_pressure,
            "evidence_demand": routing_features.evidence_demand,
            "fragility_pressure": routing_features.fragility_pressure,
            "recurrence_potential": routing_features.recurrence_potential,
            "possibility_space_need": routing_features.possibility_space_need,
            "detected_markers": routing_features.detected_markers,
            "task_signals": task_signals,
            "risk_profile": sorted(risk_profile),
        }
        return textwrap.dedent(
            f"""
            Analyze this task bottleneck and return schema-compliant JSON only.

            task:
            {task}

            Classifier assessment: {classifier_route_type}, confidence: {classifier_confidence}, source: {classifier_source}

            deterministic_features:
            {json.dumps(feature_blob, ensure_ascii=False)}
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
            "evidence_quality",
            "recurrence_potential",
            "confidence",
            "rationale",
        }
        if not required.issubset(payload.keys()):
            return None
        if not isinstance(payload["bottleneck_label"], str) or not payload["bottleneck_label"].strip():
            return None
        if not isinstance(payload["rationale"], str):
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

        int_fields = ("decision_pressure", "evidence_quality", "recurrence_potential")
        for field_name in int_fields:
            value = payload[field_name]
            if not isinstance(value, int) or value < 0 or value > 10:
                return None

        confidence = payload["confidence"]
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
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
            evidence_quality=payload["evidence_quality"],
            recurrence_potential=payload["recurrence_potential"],
            confidence=float(confidence),
            rationale=payload["rationale"].strip(),
            likely_endpoint_regime=likely_endpoint_regime,
            endpoint_confidence=float(endpoint_confidence),
        )


# ============================================================
# Prompt builder + validator
