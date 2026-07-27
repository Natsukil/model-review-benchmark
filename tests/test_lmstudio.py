import json

from coder_review_benchmark.lmstudio import LMStudioLifecycle


def test_lmstudio_native_lifecycle_endpoints(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload): self.payload = payload
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        body = json.loads(request.data) if request.data else None
        calls.append((request.method, request.full_url, body))
        if request.full_url.endswith("/load"):
            return Response({"status": "loaded", "instance_id": "instance-1"})
        if request.full_url.endswith("/unload"):
            return Response({"instance_id": "instance-1"})
        return Response({"models": [{"key": "model-a", "loaded_instances": ["instance-1"]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lifecycle = LMStudioLifecycle("http://localhost:1234", "token")
    lifecycle.list_models()
    loaded = lifecycle.load_model("model-a", context_length=32768)
    lifecycle.verify_loaded("model-a", loaded["instance_id"])
    lifecycle.unload_model(loaded["instance_id"])
    assert [call[0] for call in calls] == ["GET", "POST", "GET", "POST"]
    assert calls[1][1].endswith("/api/v1/models/load")
    assert calls[1][2] == {"model": "model-a", "context_length": 32768}
    assert calls[-1][1].endswith("/api/v1/models/unload")
