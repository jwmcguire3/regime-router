from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .models import Regime, Stage
from .state import RouterState, SessionRecord, router_state_from_jsonable, to_jsonable


class SessionStore:
    def __init__(self, root: str = "runs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: SessionRecord, filename: Optional[str] = None) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_name = filename or f"run_{timestamp}.json"
        if not safe_name.endswith(".json"):
            safe_name += ".json"
        path = self.root / safe_name
        with path.open("w", encoding="utf-8") as f:
            json.dump(to_jsonable(record), f, indent=2, ensure_ascii=False)
        return path

    def load(self, filename: str) -> Dict[str, object]:
        path = self.root / filename
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "router_state" not in data:
            data["router_state"] = None
        return data

    def load_router_state(self, filename: str, resolve_stage: Callable[[Stage], Regime]) -> Optional[RouterState]:
        data = self.load(filename)
        if not isinstance(data, dict):
            return None
        return router_state_from_jsonable(data.get("router_state"), resolve_stage)

    def list_runs(self) -> List[str]:
        return sorted(p.name for p in self.root.glob("*.json"))


# ============================================================
# CLI helpers
