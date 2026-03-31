from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..models import Stage


@dataclass(frozen=True)
class RegimeOutputContract:
    stage: Stage
    raw_response: str
    validation: Dict[str, object]
