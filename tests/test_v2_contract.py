import json
from pathlib import Path

from coder_review_benchmark.adapters import MARTIAN_RESPONSE_SCHEMA, SWE_RESPONSE_SCHEMA, MartianReviewAdapter, SWEReviewAdapter
from coder_review_benchmark.client import ModelClient
from coder_review_benchmark.config import ModelProfile
from coder_review_benchmark.context_policy import MAX_INPUT_CHARS, apply_context
from coder_review_benchmark.runner import run_review_task
from coder_review_benchmark import cli


def _profile() -> ModelProfile:
    return ModelProfile("test", "test-model", "http://example.invalid/v1", "dummy", "native_tool_calls", 4096, 32768, 1)


def test_client_payload_has_fair_lane_parameters_and_schema(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ModelClient(_profile())
    client.chat([{"role": "user", "content": "x"}], response_format=SWE_RESPONSE_SCHEMA)
    assert captured["max_tokens"] == 4096
    assert captured["temperature"] == 0.0
    assert captured["top_p"] == 1.0
    assert captured["seed"] == 42
    assert captured["stream"] is False
    assert captured["repeat_penalty"] == 1.0
    assert captured["presence_penalty"] == 0.0
    assert captured["frequency_penalty"] == 0.0
    assert captured["response_format"] == SWE_RESPONSE_SCHEMA


def test_schemas_and_review_budget_are_fixed():
    assert SWE_RESPONSE_SCHEMA["json_schema"]["strict"] is True
    assert MARTIAN_RESPONSE_SCHEMA["json_schema"]["strict"] is True
    assert SWE_RESPONSE_SCHEMA["json_schema"]["schema"]["properties"]["decision"]["enum"] == ["approve", "request_changes"]
    finding = SWE_RESPONSE_SCHEMA["json_schema"]["schema"]["properties"]["findings"]["items"]
    assert finding["properties"]["severity"]["enum"] == ["low", "medium", "high", "critical"]
    assert "compatibility" in finding["properties"]["category"]["enum"]
    result = apply_context("ISSUE\n\nDIFF:\n" + "x" * 500_000, diff="x" * 500_000)
    assert result.final_chars <= MAX_INPUT_CHARS == 100000
    assert result.final_tokens is None


def test_review_task_uses_4096_and_records_all_hashes():
    class Dummy:
        profile = _profile()
        def chat(self, messages, **kwargs):
            assert kwargs["max_tokens"] == 4096
            assert kwargs["response_format"] == SWE_RESPONSE_SCHEMA
            return {"choices": [{"message": {"content": '{"decision":"approve","summary":"ok","findings":[]}'}, "finish_reason": "stop"}]}, 0.01

    result = run_review_task(Dummy(), {"problem_statement": "issue", "model_patch": "diff --git a/a b/b\n"})
    for key in ("benchmark_serialization_sha256", "user_content_sha256", "messages_sha256"):
        assert len(result[key]) == 64
    assert result["final_input_tokens"] is None


def test_three_models_get_identical_frozen_messages():
    task = {"problem_statement": "issue", "model_patch": "diff --git a/a b/b\n@@ -1 +1 @@\n-a\n+b"}
    prepared = [SWEReviewAdapter().prepare(task) for _ in range(3)]
    assert len({p.messages_sha256 for p in prepared}) == 1
    assert len({json.dumps(p.messages, ensure_ascii=False, sort_keys=True) for p in prepared}) == 1
    martian_task = {"pr_title": "title", "pr_body": "body", "patch": "diff --git a/a b/b\n"}
    martian = [MartianReviewAdapter().prepare(martian_task) for _ in range(3)]
    assert len({p.messages_sha256 for p in martian}) == 1
    assert len({json.dumps(p.messages, ensure_ascii=False, sort_keys=True) for p in martian}) == 1


def test_env_example_has_no_real_hf_credential():
    content = Path(".env.example").read_text(encoding="utf-8")
    assert "HF_TOKEN=" in content
    assert "HF_TOKEN=HFA" not in content


def test_doctor_checks_all_three_fair_lane_models(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "get_model_profile", lambda model_id: _profile())
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    cli._doctor("missing-profile")
    report = json.loads(capsys.readouterr().out)
    assert "qwen2.5-coder-7b" in report
    assert "qwen3-coder-30b" in report
    assert "qwen3-coder-next-80b" in report
