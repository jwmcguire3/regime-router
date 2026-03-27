from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class CliSettings:
    model: str = "llama3"
    use_task_analyzer: bool = True
    task_analyzer_model: str = "llama3"
    debug_routing: bool = False
    bounded_orchestration: bool = True
    max_switches: int = 2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "CliSettings":
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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

    def reset(self) -> CliSettings:
        settings = CliSettings()
        self.save(settings)
        return settings
