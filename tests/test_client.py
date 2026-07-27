from coder_review_benchmark.client import ModelClient
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
