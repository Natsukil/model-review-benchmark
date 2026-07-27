from __future__ import annotations

import json
import urllib.request
from typing import Any


class LMStudioLifecycle:
    """Small client for LM Studio's native /api/v1 model lifecycle API."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 600):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
            method=method,
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise RuntimeError(f"LM Studio {path} returned a non-object response")
        return payload

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/models")

    def load_model(self, model: str, **settings: Any) -> dict[str, Any]:
        return self._request("POST", "/api/v1/models/load", {"model": model, **settings})

    def verify_loaded(self, model: str, instance_id: str | None = None) -> dict[str, Any]:
        payload = self.list_models()
        serialized = json.dumps(payload, ensure_ascii=False)
        if model not in serialized and (not instance_id or instance_id not in serialized):
            raise RuntimeError(f"LM Studio did not report loaded model {model!r}")
        return payload

    def unload_model(self, instance_id: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/models/unload", {"instance_id": instance_id})
