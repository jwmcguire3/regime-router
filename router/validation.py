from __future__ import annotations

import json
from typing import Dict, List, Optional, Set

from .models import (
    ARTIFACT_FIELDS,
    ARTIFACT_HINTS,
    COMPLETION_SIGNAL_HINTS,
    FAILURE_SIGNAL_HINTS,
    Stage,
    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL,
    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED,
    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED,
)
from .routing import extract_structural_signals


class OutputValidator:
    GENERIC_PHRASES = {
        "exploring",
        "assessing",
        "understanding",
        "navigating",
        "complexity",
        "various factors",
        "multiple perspectives",
        "deeper analysis",
        "careful consideration",
        "systemic issues",
        "abstract dynamics",
    }

    FORBIDDEN_GENERIC = {
        "technology",
        "stakeholders",
        "innovation",
        "solution",
        "industry",
        "team",
    }

    GENERIC_PRESSURE_WORDS = {
        "execute",
        "execution",
        "implement",
        "implementation",
        "roadmap",
        "timeline",
        "deliverable",
        "milestone",
        "resourcing",
        "coordination",
        "alignment",
        "rollout",
    }

    STAGE_VALUES = {stage.value for stage in Stage}
    ALLOWED_PROFILES = {"strict", "balanced", "lenient", "off"}

    def validate(
        self,
        stage: Stage,
        raw_response: str,
        task: str = "",
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        model_profile: str = "strict",
    ) -> Dict[str, object]:
        profile = (model_profile or "strict").strip().lower()
        if profile not in self.ALLOWED_PROFILES:
            profile = "strict"

        result: Dict[str, object] = {
            "model_profile": profile,
            "valid_json": False,
            "is_valid": False,
            "required_keys_present": False,
            "artifact_fields_present": False,
            "missing_keys": [],
            "missing_artifact_fields": [],
            "artifact_type_matches": False,
            "contract_controls_valid": False,
            "semantic_valid": False,
            "semantic_failures": [],
            "parsed": None,
        }

        try:
            parsed = json.loads(raw_response)
            result["valid_json"] = True
            result["parsed"] = parsed
        except json.JSONDecodeError as e:
            result["error"] = f"JSON decode error: {e}"
            return result

        required = {
            "regime",
            "purpose",
            "artifact_type",
            "artifact",
            "completion_signal",
            "failure_signal",
            "recommended_next_regime",
        }
        parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
        missing_keys = sorted(required - parsed_keys)
        result["missing_keys"] = missing_keys
        result["required_keys_present"] = len(missing_keys) == 0

        if not result["required_keys_present"]:
            return result

        control_failures: List[str] = []
        regime_raw = str(parsed.get("regime", "")).strip().lower()
        if stage.value not in regime_raw:
            control_failures.append(
                f"regime field mismatch: output claims '{parsed.get('regime')}' but active regime is {stage.value}"
            )

        artifact = parsed.get("artifact", {})
        if not isinstance(artifact, dict):
            result["error"] = "artifact must be a JSON object"
            return result

        required_artifact_fields = set(ARTIFACT_FIELDS[stage])
        artifact_keys = set(artifact.keys())
        missing_artifact_fields = sorted(required_artifact_fields - artifact_keys)
        result["missing_artifact_fields"] = missing_artifact_fields
        result["artifact_fields_present"] = len(missing_artifact_fields) == 0
        result["artifact_type_matches"] = parsed.get("artifact_type") == ARTIFACT_HINTS[stage]
        control_failures.extend(self._validate_control_fields(stage, parsed))
        result["control_failures"] = control_failures
        result["contract_controls_valid"] = len(control_failures) == 0

        structural_valid = bool(
            result["valid_json"]
            and result["required_keys_present"]
            and result["artifact_fields_present"]
            and result["artifact_type_matches"]
            and result["contract_controls_valid"]
        )

        if not structural_valid:
            result["is_valid"] = False
            return result

        semantic_failures = self._semantic_checks(
            stage,
            artifact,
            task,
            task_signals=task_signals,
            risk_profile=risk_profile,
            model_profile=profile,
        )
        result["semantic_failures"] = semantic_failures
        result["semantic_valid"] = len(semantic_failures) == 0
        result["is_valid"] = structural_valid and result["semantic_valid"]
        return result

    def _validate_control_fields(self, stage: Stage, parsed: Dict[str, object]) -> List[str]:
        failures: List[str] = []
        purpose = parsed.get("purpose")
        completion_signal = parsed.get("completion_signal")
        failure_signal = parsed.get("failure_signal")
        recommended_next_regime = parsed.get("recommended_next_regime")

        if not isinstance(purpose, str) or not purpose.strip():
            failures.append("purpose must be a non-empty string")

        if not isinstance(completion_signal, str) or not completion_signal.strip():
            failures.append("completion_signal must be a non-empty string")
        else:
            expected_completion_tokens = set(COMPLETION_SIGNAL_HINTS[stage].split("_"))
            signal_tokens = self._tokenize(completion_signal)
            if not signal_tokens.intersection(expected_completion_tokens):
                failures.append("completion_signal is not stage-appropriate")

        if not isinstance(failure_signal, str) or not failure_signal.strip():
            failures.append("failure_signal must be a non-empty string")
        else:
            expected_failure_tokens = set(FAILURE_SIGNAL_HINTS[stage].split("_"))
            signal_tokens = self._tokenize(failure_signal)
            if not signal_tokens.intersection(expected_failure_tokens):
                failures.append("failure_signal is not stage-appropriate")

        if not isinstance(recommended_next_regime, str) or not recommended_next_regime.strip():
            failures.append("recommended_next_regime must be a non-empty string")
        elif recommended_next_regime not in self.STAGE_VALUES:
            failures.append("recommended_next_regime must be a valid regime stage")

        return failures

    def _profile_config(self, model_profile: str) -> Dict[str, object]:
        profile = (model_profile or "strict").strip().lower()
        if profile == "balanced":
            return {
                "min_words_per_field": 2,
                "jaccard_similarity_limit": 0.85,
                "task_overlap_min": 1,
                "check_generic_filler": True,
                "check_forbidden_generic": True,
                "check_stage_specific": True,
            }
        if profile == "lenient":
            return {
                "min_words_per_field": 1,
                "jaccard_similarity_limit": 0.93,
                "task_overlap_min": 1,
                "check_generic_filler": False,
                "check_forbidden_generic": False,
                "check_stage_specific": False,
            }
        if profile == "off":
            return {
                "min_words_per_field": 0,
                "jaccard_similarity_limit": 1.01,
                "task_overlap_min": 0,
                "check_generic_filler": False,
                "check_forbidden_generic": False,
                "check_stage_specific": False,
            }
        # strict
        return {
            "min_words_per_field": 3,
            "jaccard_similarity_limit": 0.75,
            "task_overlap_min": 2,
            "check_generic_filler": True,
            "check_forbidden_generic": True,
            "check_stage_specific": True,
        }

    def _semantic_checks(
        self,
        stage: Stage,
        artifact: Dict[str, object],
        task: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
        model_profile: str = "strict",
    ) -> List[str]:
        cfg = self._profile_config(model_profile)
        flat_fields: Dict[str, str] = {}
        for k, v in artifact.items():
            if isinstance(v, list):
                flat_fields[k] = " ".join(str(x) for x in v).strip()
            else:
                flat_fields[k] = str(v).strip()

        values = {k: self._normalize(v) for k, v in flat_fields.items() if v}
        all_text = " ".join(flat_fields.values()).lower()
        artifact_tokens = self._tokenize(all_text)
        extracted_signals = set(extract_structural_signals(task))
        active_signals = extracted_signals | set(task_signals or [])

        failures: List[str] = []
        failures.extend(self._check_field_length(flat_fields, cfg))
        failures.extend(self._check_field_repetition(values, cfg))
        failures.extend(self._check_generic_filler(values, cfg))
        failures.extend(self._check_task_grounding(values, task, cfg))
        failures.extend(
            self._check_signal_grounding(
                flat_fields,
                artifact_tokens,
                active_signals,
                task,
                cfg,
                model_profile,
            )
        )
        failures.extend(self._check_stage_specific(stage, flat_fields, values, active_signals, cfg))
        return failures

    def _check_field_length(self, flat_fields: Dict[str, str], cfg: Dict[str, object]) -> List[str]:
        failures: List[str] = []
        min_words_per_field = int(cfg["min_words_per_field"])
        if min_words_per_field > 0:
            for k, v in flat_fields.items():
                if len(v.split()) < min_words_per_field:
                    failures.append(f"{k} is too short to be meaningful")
        return failures

    def _check_field_repetition(self, values: Dict[str, str], cfg: Dict[str, object]) -> List[str]:
        failures: List[str] = []
        jaccard_limit = float(cfg["jaccard_similarity_limit"])
        keys = list(values.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if values[a] and values[a] == values[b]:
                    failures.append(f"{a} duplicates {b}")
                elif self._jaccard(values[a], values[b]) > jaccard_limit:
                    failures.append(f"{a} is too similar to {b}")
        return failures

    def _check_generic_filler(self, values: Dict[str, str], cfg: Dict[str, object]) -> List[str]:
        failures: List[str] = []
        if bool(cfg["check_generic_filler"]):
            for k, v in values.items():
                hits = [p for p in self.GENERIC_PHRASES if p in v]
                if hits:
                    failures.append(f"{k} contains generic filler: {', '.join(hits[:3])}")
        return failures

    def _check_task_grounding(self, values: Dict[str, str], task: str, cfg: Dict[str, object]) -> List[str]:
        failures: List[str] = []
        task_tokens = self._tokenize(task)
        overlap_min = int(cfg["task_overlap_min"])
        if task_tokens and overlap_min > 0:
            grounded = False
            for v in values.values():
                overlap = task_tokens & self._tokenize(v)
                if len(overlap) >= overlap_min:
                    grounded = True
                    break
            if not grounded:
                failures.append("artifact is not grounded in the task specifics")
        return failures

    def _check_signal_grounding(
        self,
        flat_fields: Dict[str, str],
        artifact_tokens: Set[str],
        active_signals: Set[str],
        task: str,
        cfg: Dict[str, object],
        model_profile: str,
    ) -> List[str]:
        failures: List[str] = []
        all_text = " ".join(flat_fields.values()).lower()
        if active_signals:
            if bool(cfg["check_forbidden_generic"]):
                forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
                if forbidden_hits:
                    failures.append(
                        f"artifact introduces forbidden generic domain nouns: {', '.join(forbidden_hits)}"
                    )

            if model_profile in {"strict", "balanced"}:
                token_expectations = {
                    STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED: ["expand", "define"],
                    STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL: ["concrete", "small"],
                    STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED: ["fragment", "spine"],
                }
                # Structural-signal grounding is advisory only: keep the token scan for
                # potential diagnostics, but do not turn missing matches into a hard
                # semantic validation failure.
                for signal in active_signals:
                    tokens = token_expectations.get(signal, [])
                    if tokens and all(tok in all_text for tok in tokens):
                        break
        return failures

    def _check_stage_specific(
        self,
        stage: Stage,
        flat_fields: Dict[str, str],
        values: Dict[str, str],
        active_signals: Set[str],
        cfg: Dict[str, object],
    ) -> List[str]:
        failures: List[str] = []
        all_text = " ".join(flat_fields.values()).lower()
        artifact_tokens = self._tokenize(all_text)
        if bool(cfg["check_stage_specific"]) and stage == Stage.SYNTHESIS:
            if "central_claim" in values and "organizing_idea" in values:
                if self._jaccard(values["central_claim"], values["organizing_idea"]) > 0.65:
                    failures.append("organizing_idea restates central_claim instead of explaining it")

            if "supporting_structure" in values:
                token_count = len(self._tokenize(values["supporting_structure"]))
                if token_count < 6:
                    failures.append("supporting_structure is too thin")

            if active_signals:
                if "central_claim" in flat_fields and self._signal_overlap_count(flat_fields["central_claim"], list(active_signals)) < 1:
                    failures.append("central_claim is not anchored to the task's structural signals")
                if "organizing_idea" in flat_fields and self._signal_overlap_count(flat_fields["organizing_idea"], list(active_signals)) < 1:
                    failures.append("organizing_idea is not anchored to the task's structural signals")
                if "key_tensions" in flat_fields and self._signal_overlap_count(flat_fields["key_tensions"], list(active_signals)) < 1:
                    failures.append("key_tensions are not tied to the task's structural signals")

                if "pressure_points" in flat_fields:
                    pressure_text = flat_fields["pressure_points"].lower()
                    generic_pressure_hits = sorted(word for word in self.GENERIC_PRESSURE_WORDS if word in pressure_text)
                    if generic_pressure_hits:
                        failures.append(
                            "pressure_points use generic execution language instead of frame pressure tests: "
                            + ", ".join(generic_pressure_hits[:4])
                        )
                    if self._signal_overlap_count(flat_fields["pressure_points"], list(active_signals)) < 1:
                        failures.append("pressure_points do not test the frame against the task's original structural signals")

        if bool(cfg["check_stage_specific"]) and stage == Stage.EXPLORATION:
            forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
            if forbidden_hits and bool(cfg["check_forbidden_generic"]):
                failures.append(
                    f"exploration artifact introduces ungrounded generic domain terms: {', '.join(forbidden_hits)}"
                )

            if active_signals and not any(tok in all_text for tok in ("expand", "define", "small", "spine", "fragment", "concrete")):
                failures.append("exploration artifact does not engage the task's structural signals")

        return failures

    def _normalize(self, text: str) -> str:
        return " ".join(self._tokenize(text))

    def _tokenize(self, text: str) -> set[str]:
        return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if len(t) > 2}

    def _jaccard(self, a: str, b: str) -> float:
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    def _signal_overlap_count(self, text: str, signals: List[str]) -> int:
        signal_tokens = {
            STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED: {"expand", "define"},
            STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL: {"concrete", "small"},
            STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED: {"fragment", "spine"},
        }
        text_tokens = self._tokenize(text)
        overlap_count = 0
        for signal in signals:
            expected = signal_tokens.get(signal)
            if expected:
                if text_tokens & expected:
                    overlap_count += 1
                continue
            fallback_tokens = self._tokenize(signal.replace("_", " "))
            if fallback_tokens and text_tokens & fallback_tokens:
                overlap_count += 1
        return overlap_count
