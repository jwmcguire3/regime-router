from __future__ import annotations

import json
from typing import Dict, List, Optional, Set

from .models import (
    ARTIFACT_FIELDS,
    ARTIFACT_HINTS,
    CANONICAL_FAILURE_IF_OVERUSED,
    REGIME_OUTPUT_CONTRACT_KEYS,
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

    REQUIRED_SIGNAL_WORDS = {
        "expand",
        "define",
        "small",
        "spine",
        "fragment",
    }

    def validate(
        self,
        stage: Stage,
        raw_response: str,
        task: str = "",
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {
            "valid_json": False,
            "required_keys_present": False,
            "artifact_fields_present": False,
            "missing_keys": [],
            "missing_artifact_fields": [],
            "artifact_type_matches": False,
            "stage_matches": False,
            "control_fields_valid": False,
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

        required = set(REGIME_OUTPUT_CONTRACT_KEYS)
        parsed_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
        missing_keys = sorted(required - parsed_keys)
        result["missing_keys"] = missing_keys
        result["required_keys_present"] = len(missing_keys) == 0

        if not result["required_keys_present"]:
            return result

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
        result["stage_matches"] = parsed.get("stage") == stage.value
        result["control_fields_valid"] = self._validate_control_fields(stage, parsed, result)

        structural_valid = bool(
            result["valid_json"]
            and result["required_keys_present"]
            and result["artifact_fields_present"]
            and result["artifact_type_matches"]
            and result["stage_matches"]
            and result["control_fields_valid"]
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
        )
        result["semantic_failures"] = semantic_failures
        result["semantic_valid"] = len(semantic_failures) == 0
        result["is_valid"] = structural_valid and result["semantic_valid"]
        return result

    def _validate_control_fields(self, stage: Stage, parsed: Dict[str, object], result: Dict[str, object]) -> bool:
        errors: List[str] = []
        for key in ("purpose", "completion_signal", "failure_signal"):
            value = parsed.get(key)
            if not isinstance(value, str) or len(value.strip()) < 3:
                errors.append(f"{key} must be a non-empty string")

        next_regime = parsed.get("recommended_next_regime")
        allowed_next_regimes = {s.value for s in Stage} | {"none"}
        if not isinstance(next_regime, str) or next_regime not in allowed_next_regimes:
            errors.append(
                "recommended_next_regime must be one of: "
                + ", ".join(sorted(allowed_next_regimes))
            )

        failure_signal = str(parsed.get("failure_signal", "")).lower()
        expected_failure = CANONICAL_FAILURE_IF_OVERUSED[stage].lower()
        expected_tokens = {tok for tok in self._tokenize(expected_failure) if len(tok) > 3}
        if expected_tokens and not (self._tokenize(failure_signal) & expected_tokens):
            errors.append("failure_signal does not align with the regime's dominant failure mode")

        if errors:
            result["error"] = "; ".join(errors)
            return False
        return True

    def _semantic_checks(
        self,
        stage: Stage,
        artifact: Dict[str, object],
        task: str,
        task_signals: Optional[List[str]] = None,
        risk_profile: Optional[Set[str]] = None,
    ) -> List[str]:
        failures: List[str] = []

        flat_fields = {}
        for k, v in artifact.items():
            if isinstance(v, list):
                flat_fields[k] = " ".join(str(x) for x in v).strip()
            else:
                flat_fields[k] = str(v).strip()

        values = {k: self._normalize(v) for k, v in flat_fields.items() if v}

        # 1. Empty or ultra-short content
        for k, v in flat_fields.items():
            if len(v.split()) < 3:
                failures.append(f"{k} is too short to be meaningful")

        # 2. Repetition across fields
        keys = list(values.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                if values[a] and values[a] == values[b]:
                    failures.append(f"{a} duplicates {b}")
                elif self._jaccard(values[a], values[b]) > 0.75:
                    failures.append(f"{a} is too similar to {b}")

        # 3. Generic filler detection
        for k, v in values.items():
            hits = [p for p in self.GENERIC_PHRASES if p in v]
            if hits:
                failures.append(f"{k} contains generic filler: {', '.join(hits[:3])}")

        # 4. Task grounding check
        task_tokens = self._tokenize(task)
        if task_tokens:
            grounded = False
            for v in values.values():
                overlap = task_tokens & self._tokenize(v)
                if len(overlap) >= 2:
                    grounded = True
                    break
            if not grounded:
                failures.append("artifact is not grounded in the task specifics")

        # 5. Task-specific grounding checks for abstract framing tasks
        all_text = " ".join(flat_fields.values()).lower()
        artifact_tokens = self._tokenize(all_text)
        task_text = task.lower()

        extracted_signals = set(extract_structural_signals(task))
        active_signals = extracted_signals | set(task_signals or [])
        if active_signals:
            forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
            if forbidden_hits:
                failures.append(
                    f"artifact introduces forbidden generic domain nouns: {', '.join(forbidden_hits)}"
                )

            token_expectations = {
                STRUCTURAL_SIGNAL_EXPANSION_WHEN_DEFINED: ["expand", "define"],
                STRUCTURAL_SIGNAL_CONCRETE_TOO_SMALL: ["concrete", "small"],
                STRUCTURAL_SIGNAL_FRAGMENTS_SPINE_MISSED: ["fragment", "spine"],
            }
            matched = 0
            for signal in active_signals:
                tokens = token_expectations.get(signal, [])
                if all(tok in all_text for tok in tokens):
                    matched += 1
            if matched < 1:
                failures.append(
                    "artifact is not grounded in the task's core structural signals"
                )

        # 6. Stage-specific checks
        if stage == Stage.SYNTHESIS:
            if "central_claim" in values and "organizing_idea" in values:
                if self._jaccard(values["central_claim"], values["organizing_idea"]) > 0.65:
                    failures.append("organizing_idea restates central_claim instead of explaining it")

            if "supporting_structure" in values:
                token_count = len(self._tokenize(values["supporting_structure"]))
                if token_count < 6:
                    failures.append("supporting_structure is too thin")

            if active_signals:
                if "central_claim" in flat_fields and self._signal_overlap_count(flat_fields["central_claim"], list(active_signals)) < 1:
                    failures.append(
                        "central_claim is not anchored to the task's structural signals"
                    )
                if "organizing_idea" in flat_fields and self._signal_overlap_count(flat_fields["organizing_idea"], list(active_signals)) < 1:
                    failures.append(
                        "organizing_idea is not anchored to the task's structural signals"
                    )
                if "key_tensions" in flat_fields and self._signal_overlap_count(flat_fields["key_tensions"], list(active_signals)) < 1:
                    failures.append(
                        "key_tensions are not tied to the task's structural signals"
                    )

                if "pressure_points" in flat_fields:
                    pressure_text = flat_fields["pressure_points"].lower()
                    generic_pressure_hits = sorted(
                        word for word in self.GENERIC_PRESSURE_WORDS if word in pressure_text
                    )
                    if generic_pressure_hits:
                        failures.append(
                            "pressure_points use generic execution language instead of frame pressure tests: "
                            + ", ".join(generic_pressure_hits[:4])
                        )
                    if self._signal_overlap_count(flat_fields["pressure_points"], list(active_signals)) < 1:
                        failures.append(
                            "pressure_points do not test the frame against the task's original structural signals"
                        )

        if stage == Stage.EXPLORATION:
            forbidden_hits = sorted(t for t in self.FORBIDDEN_GENERIC if t in artifact_tokens)
            if forbidden_hits:
                failures.append(
                    f"exploration artifact introduces ungrounded generic domain terms: {', '.join(forbidden_hits)}"
                )

            if active_signals and not any(tok in all_text for tok in ("expand", "define", "small", "spine", "fragment", "concrete")):
                failures.append(
                    "exploration artifact does not engage the task's structural signals"
                )

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
# ============================================================
# Evolution engine
# ============================================================
