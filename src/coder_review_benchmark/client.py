from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .config import ModelProfile


class ModelRequestError(RuntimeError):
    def __init__(self, message: str, *, raw_response: str | None, request_attempts: int):
        super().__init__(message)
        self.raw_response = raw_response
        self.request_attempts = request_attempts


class ModelClient:
    def __init__(self, profile: ModelProfile, timeout: int = 300, max_retries: int | None = None):
        self.profile = profile
        self.timeout = timeout
        configured_retries = max_retries if max_retries is not None else int(os.getenv("CBM_MODEL_MAX_RETRIES", "1"))
        self.max_retries = max(0, min(1, configured_retries))
        self.last_request_attempts = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stream: bool | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        body: dict[str, Any] = {
            "model": self.profile.model_name,
            "messages": messages,
            "temperature": self.profile.temperature if temperature is None else temperature,
            "top_p": self.profile.top_p if top_p is None else top_p,
            "seed": self.profile.seed if seed is None else seed,
            "stream": self.profile.stream if stream is None else stream,
            "repeat_penalty": self.profile.repeat_penalty if repeat_penalty is None else repeat_penalty,
            "presence_penalty": self.profile.presence_penalty if presence_penalty is None else presence_penalty,
            "frequency_penalty": self.profile.frequency_penalty if frequency_penalty is None else frequency_penalty,
            "max_tokens": max_tokens if max_tokens is not None else self.profile.max_output_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if response_format is not None:
            body["response_format"] = response_format
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
            self.last_request_attempts = attempt + 1
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
                    raise ModelRequestError(f"model request failed after {attempt + 1} attempt(s): HTTP {exc.code}{suffix}", raw_response=detail or None, request_attempts=attempt + 1) from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise ModelRequestError(f"model request failed after {attempt + 1} attempt(s): {exc}", raw_response=None, request_attempts=attempt + 1) from exc
            time.sleep(min(2 ** attempt, 8))
        else:  # pragma: no cover - defensive; the loop always returns or raises
            raise RuntimeError(f"model request failed: {last_error}")
        payload["_request_attempts"] = self.last_request_attempts
        return payload, time.perf_counter() - started
