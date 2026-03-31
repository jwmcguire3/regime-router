from __future__ import annotations

import json
from typing import Dict
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class OpenAIModelClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key
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
            "temperature": temperature,
            "max_tokens": num_predict,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")
                content = json.loads(body)["choices"][0]["message"]["content"]
                return {"response": content}
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible HTTP error {e.code}: {detail}") from e

    def list_models(self) -> Dict[str, object]:
        url = f"{self.base_url}/models"
        req = Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with urlopen(req, timeout=self.timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
