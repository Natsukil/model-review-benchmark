import io
import urllib.error

import pytest

from coder_review_benchmark.client import ModelClient, ModelRequestError
from coder_review_benchmark.config import ModelProfile


def test_no_auth_profile_omits_authorization(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices": [{"message": {"content": "OK"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(dict(request.header_items()))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    profile = ModelProfile(
        id="lmstudio",
        model_name="qwen3-coder-30b",
        base_url="http://localhost:1234/v1",
        api_key="stale-token",
        parser="native_tool_calls",
        max_output_tokens=64,
        max_context_tokens=32768,
        max_concurrency=1,
        send_auth=False,
    )
    ModelClient(profile).chat([{"role": "user", "content": "hello"}])
    assert "Authorization" not in captured


def test_http_error_preserves_raw_response_and_attempt_count(monkeypatch):
    profile = ModelProfile("m", "m", "http://invalid/v1", "", "native_tool_calls", 64, 32768, 1, send_auth=False)
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b'{"error":"unsupported schema"}'))
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(ModelRequestError) as caught:
        ModelClient(profile).chat([{"role": "user", "content": "x"}])
    assert caught.value.status_code == 400
    assert caught.value.response_body == '{"error":"unsupported schema"}'
    assert caught.value.attempts == 1
    assert caught.value.elapsed >= 0
