from pathlib import Path

from coder_review_benchmark.config import ModelProfile
from coder_review_benchmark.runner import _agent_prompt, run_agent_task


class DummyQwen25Client:
    def __init__(self):
        self.messages = []
        self.responses = [
            {"choices": [{"message": {"content": "read_file(path='example.py')"}}]},
            {"choices": [{"message": {"content": "The division can raise ZeroDivisionError."}}]},
        ]

    def chat(self, messages, tools):
        self.messages.append(list(messages))
        return self.responses.pop(0), 0.01


def test_qwen25_text_tool_loop(tmp_path: Path):
    (tmp_path / "example.py").write_text("def divide(a, b):\n    return a / b\n", encoding="utf-8")
    profile = ModelProfile(
        id="qwen2.5-coder-7b",
        model_name="qwen2.5-coder:latest",
        base_url="http://example.invalid/v1",
        api_key="test",
        parser="qwen25_text_fallback",
        max_output_tokens=4096,
        max_context_tokens=32768,
        max_concurrency=1,
    )
    client = DummyQwen25Client()

    result = run_agent_task(
        client,
        profile,
        {"prompt": "Inspect example.py"},
        tmp_path,
        max_turns=3,
    )

    assert result["status"] == "completed"
    assert result["events"][0]["tool_calls"][0]["name"] == "read_file"
    assert result["events"][0]["tool_calls"][0]["arguments"] == {"path": "example.py"}
    followup = client.messages[1]
    assert followup[-2] == {"role": "assistant", "content": "read_file(path='example.py')"}
    assert followup[-1]["role"] == "user"
    assert "<tool_result name='read_file'>" in followup[-1]["content"]
    assert "return a / b" in followup[-1]["content"]


def test_agent_prompt_does_not_leak_gold_patches():
    prompt = _agent_prompt(
        {
            "title": "Fix the parser",
            "body": "Parsing fails on empty input.",
            "fix_patch": "SECRET_GOLD_FIX",
            "test_patch": "SECRET_HIDDEN_TEST",
            "resolved_issues": [{"number": 7, "title": "Empty input", "body": "It crashes."}],
        }
    )
    assert "Fix the parser" in prompt
    assert "Empty input" in prompt
    assert "SECRET_GOLD_FIX" not in prompt
    assert "SECRET_HIDDEN_TEST" not in prompt
