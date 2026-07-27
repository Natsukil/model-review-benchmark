from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import ModelProfile


class ModelClient:
    def __init__(self, profile: ModelProfile, timeout: int = 300, max_retries: int | None = None):
        self.profile = profile
        self.timeout = timeout
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("CBM_MODEL_MAX_RETRIES", "3"))

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0,
        max_tokens: int | None = None,
    ) -> tuple[dict[str, Any], float]:
        body: dict[str, Any] = {
            "model": self.profile.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.profile.max_output_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self.profile.send_auth and self.profile.api_key:
            headers["Authorization"] = f"Bearer {self.profile.api_key}"
        request = urllib.request.Request(
            self.profile.base_url + "/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        started = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                retryable = exc.code in {408, 429} or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")[:2000]
                    except Exception:
                        detail = ""
                    suffix = f": {detail}" if detail else ""
                    raise RuntimeError(f"model request failed after {attempt + 1} attempt(s): HTTP {exc.code}{suffix}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise RuntimeError(f"model request failed after {attempt + 1} attempt(s): {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
        else:  # pragma: no cover - defensive; the loop always returns or raises
            raise RuntimeError(f"model request failed: {last_error}")
        return payload, time.perf_counter() - started
