from __future__ import annotations

import json
from typing import Dict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaModelClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
        payload = {
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        url = f"{self.base_url}/api/generate"
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {e.code}: {detail}") from e
        except URLError as e:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}. Is it running?") from e

    def list_models(self) -> Dict[str, object]:
        url = f"{self.base_url}/api/tags"
        req = Request(url, method="GET")
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except Exception as e:
            raise RuntimeError(f"Could not list Ollama models from {self.base_url}: {e}") from e
