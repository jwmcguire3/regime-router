from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class UserSettings:
    model: str = "dolphin29:latest"
    use_task_analyzer: bool = True
    task_analyzer_model: str = "dolphin29:latest"
    debug_routing: bool = False
    bounded_orchestration: bool = True
    max_switches: int = 2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UserSettings":
        defaults = cls()
        data = {
            "model": str(raw.get("model", defaults.model)),
            "use_task_analyzer": bool(raw.get("use_task_analyzer", defaults.use_task_analyzer)),
            "task_analyzer_model": str(raw.get("task_analyzer_model", defaults.task_analyzer_model)),
            "debug_routing": bool(raw.get("debug_routing", defaults.debug_routing)),
            "bounded_orchestration": bool(raw.get("bounded_orchestration", defaults.bounded_orchestration)),
            "max_switches": int(raw.get("max_switches", defaults.max_switches)),
        }
        if data["max_switches"] < 0:
            data["max_switches"] = defaults.max_switches
        return cls(**data)


@dataclass
class ModelControlSettings:
    # Profile-only control plane (simple now; granular overrides can be added later)
    model_profile: str = "strict"  # strict | balanced | lenient | off

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ModelControlSettings":
        defaults = cls()
        profile = str(raw.get("model_profile", defaults.model_profile)).strip().lower()
        allowed = {"strict", "balanced", "lenient", "off"}
        if profile not in allowed:
            profile = defaults.model_profile
        return cls(model_profile=profile)


@dataclass
class CliSettings:
    user: UserSettings = field(default_factory=UserSettings)
    model_controls: ModelControlSettings = field(default_factory=ModelControlSettings)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CliSettings":
        if not isinstance(raw, dict):
            return cls()

        # New nested format
        if "user" in raw or "model_controls" in raw:
            user_raw = raw.get("user", {})
            model_controls_raw = raw.get("model_controls", {})
            if not isinstance(user_raw, dict):
                user_raw = {}
            if not isinstance(model_controls_raw, dict):
                model_controls_raw = {}
            return cls(
                user=UserSettings.from_dict(user_raw),
                model_controls=ModelControlSettings.from_dict(model_controls_raw),
            )

        # Backward-compatible migration from old flat format
        return cls(
            user=UserSettings.from_dict(raw),
            model_controls=ModelControlSettings.from_dict(raw),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user": asdict(self.user),
            "model_controls": asdict(self.model_controls),
        }


class CliSettingsStore:
    def __init__(self, path: str = ".router_settings.json") -> None:
        self.path = Path(path)

    def load(self) -> CliSettings:
        if not self.path.exists():
            return CliSettings()
        with self.path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return CliSettings()
        return CliSettings.from_dict(raw)

    def save(self, settings: CliSettings) -> Path:
        if self.path.parent != Path("."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        return self.path

    def reset_all(self) -> CliSettings:
        settings = CliSettings()
        self.save(settings)
        return settings

    def reset_user(self) -> CliSettings:
        current = self.load()
        current.user = UserSettings()
        self.save(current)
        return current

    def reset_model_controls(self) -> CliSettings:
        current = self.load()
        current.model_controls = ModelControlSettings()
        self.save(current)
        return current

    # Backward-compatible alias for existing callers
    def reset(self) -> CliSettings:
        return self.reset_all()
