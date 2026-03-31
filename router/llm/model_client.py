from __future__ import annotations

from typing import Dict, Protocol


class ModelClient(Protocol):
    def generate(
        self,
        *,
        model: str,
        system: str,
        prompt: str,
        stream: bool = False,
        temperature: float = 0.2,
        num_predict: int = 1200,
    ) -> Dict[str, object]:
        ...

    def list_models(self) -> Dict[str, object]:
        ...
