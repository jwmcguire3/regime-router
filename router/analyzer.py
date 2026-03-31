from __future__ import annotations

import json
import re
import textwrap
from typing import Dict, List, Optional, Set, Tuple

from .llm import ModelClient
from .models import RoutingFeatures, Stage, TaskAnalyzerOutput


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
    ) -> Optional[TaskAnalyzerOutput]:
        self.last_error_summary = None
        response = self.model_client.generate(
            model=self.model,
            system=self._build_system_prompt(),
            prompt=self._build_user_prompt(task, routing_features, task_signals, risk_profile),
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
              "rationale": "short string"
            }
            """
        ).strip()

    def _build_user_prompt(
        self,
        task: str,
        routing_features: RoutingFeatures,
        task_signals: List[str],
        risk_profile: Set[str],
    ) -> str:
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
        )


# ============================================================
# Prompt builder + validator
