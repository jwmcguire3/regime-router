from __future__ import annotations

from typing import Dict

from ..prompts import PromptBuilder


def select_repair_mode(validation: Dict[str, object]) -> str:
    if not validation.get("valid_json", False):
        return PromptBuilder.REPAIR_MODE_SCHEMA
    if (
        not validation.get("required_keys_present", False)
        or not validation.get("artifact_fields_present", False)
        or not validation.get("artifact_type_matches", False)
        or not validation.get("contract_controls_valid", False)
        or bool(validation.get("missing_keys", []))
        or bool(validation.get("missing_artifact_fields", []))
        or bool(validation.get("control_failures", []))
    ):
        return PromptBuilder.REPAIR_MODE_SCHEMA

    semantic_failures = [str(f).lower() for f in validation.get("semantic_failures", [])]
    genericity_markers = (
        "generic filler",
        "forbidden generic domain nouns",
        "ungrounded generic domain terms",
    )
    if any(marker in failure for failure in semantic_failures for marker in genericity_markers):
        return PromptBuilder.REPAIR_MODE_REDUCE_GENERICITY
    return PromptBuilder.REPAIR_MODE_SEMANTIC
