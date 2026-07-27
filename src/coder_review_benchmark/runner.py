from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from .client import ModelClient
from .config import ModelProfile
from .tool_parser import parse_tool_calls
from .tools import SafeWorkspace, TOOL_SCHEMAS
from .scoring import parse_martian_review, parse_review
from .adapters import MartianReviewAdapter, ReviewInput, SWEReviewAdapter


SYSTEM = "You are a careful software engineer. Explore the repository with tools, make the smallest correct change, run tests, and report what you did."
QWEN25_TOOL_PROTOCOL = """

You are operating in a tool-use loop. Before claiming to have inspected or
changed a file, you must call the corresponding tool and wait for its result.
When calling a tool, output exactly one Python-style call and no other text,
for example:

list_files(path='.')

Use keyword arguments containing only JSON-like literal values. Never invent a
tool result. After the tool result is returned, either call another tool in the
same format or provide the final answer without a tool call.

For repository issue-resolution tasks, a directory listing alone is never
enough. Do not finish until you have inspected relevant source code, attempted
a concrete fix, and run at least one relevant check. Do not repeat the task or
tool output in your response.
"""


def _agent_prompt(task: dict[str, Any]) -> str:
    title = str(task.get("title") or "Resolve the reported issue")
    body = str(task.get("body") or "")
    issues = task.get("resolved_issues") or []
    issue_text = "\n\n".join(
        f"Related issue #{issue.get('number')}: {issue.get('title', '')}\n{issue.get('body') or ''}"
        for issue in issues
        if isinstance(issue, dict)
    )
    sections = [f"Implement a correct fix for this repository issue.\n\nTitle: {title}"]
    if body:
        sections.append(f"Description:\n{body}")
    if issue_text:
        sections.append(issue_text)
    sections.append("Inspect the repository, make the smallest correct code change, and run relevant tests.")
    return "\n\n".join(sections)


def run_agent_task(
    client: ModelClient,
    profile: ModelProfile,
    task: dict[str, Any],
    workspace: Path,
    max_turns: int = 20,
    command_timeout: int = 120,
    tool_executor: Any | None = None,
) -> dict[str, Any]:
    system = SYSTEM + (QWEN25_TOOL_PROTOCOL if profile.parser == "qwen25_text_fallback" else "")
    prompt = task.get("prompt") or task.get("problem_statement") or _agent_prompt(task)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    tools = tool_executor or SafeWorkspace(workspace, command_timeout)
    events: list[dict[str, Any]] = []
    started = time.perf_counter()
    malformed = 0
    for turn in range(max_turns):
        response, elapsed = client.chat(messages, TOOL_SCHEMAS)
        parsed = parse_tool_calls(response, profile.parser)
        events.append({"turn": turn, "elapsed": elapsed, "response": response, "parse_error": parsed.error})
        if parsed.error:
            malformed += 1
            messages.extend([{ "role": "assistant", "content": parsed.text }, {"role": "user", "content": "Your tool call format was invalid. Emit exactly one registered tool call with JSON arguments, or provide the final answer."}])
            continue
        if not parsed.tool_calls:
            return {"status": "completed", "answer": parsed.text, "events": events, "malformed_calls": malformed, "wall_seconds": time.perf_counter() - started}
        textual_fallback = all(call.source == "qwen25_text_fallback" for call in parsed.tool_calls)
        if textual_fallback:
            messages.append({"role": "assistant", "content": parsed.text})
        for call in parsed.tool_calls:
            try:
                result = tools.execute(call.name, call.arguments)
                events[-1].setdefault("tool_calls", []).append({"name": call.name, "arguments": call.arguments, "source": call.source, "result": result})
                if textual_fallback:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"<tool_result name={call.name!r}>\n{result}\n</tool_result>\n"
                            "Do not repeat this output. Continue with exactly one tool call, "
                            "or give a concise final answer only after making and testing the fix."
                        ),
                    })
                else:
                    call_id = call.call_id or str(uuid.uuid4())
                    messages.append({"role": "assistant", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": call.name, "arguments": json.dumps(call.arguments)}}], "content": None})
                    messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
            except Exception as exc:
                events[-1].setdefault("tool_calls", []).append({"name": call.name, "arguments": call.arguments, "source": call.source, "error": str(exc)})
                messages.append({"role": "user", "content": f"Tool {call.name} failed: {exc}"})
    return {"status": "max_turns", "answer": "", "events": events, "malformed_calls": malformed, "wall_seconds": time.perf_counter() - started}


def run_review_task(client: ModelClient, task: dict[str, Any], *, protocol: str = "swe", context_policy: str = "common-100k-char-v1", adapter: Any | None = None) -> dict[str, Any]:
    adapter = adapter or (MartianReviewAdapter() if protocol == "martian" else SWEReviewAdapter())
    prepared: ReviewInput = adapter.prepare(task, context_policy)
    max_output = min(getattr(client.profile, "max_output_tokens", 4096), 4096)
    response_kwargs = {"max_tokens": max_output}
    if getattr(client.profile, "structured_output", True):
        response_kwargs["response_format"] = adapter.response_format
    response, elapsed = client.chat(prepared.messages, **response_kwargs)
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    parsed = parse_martian_review(content) if protocol == "martian" else parse_review(content, protocol="swe")
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    choice = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    status = "completed" if parsed["schema_valid"] else ("format_failure" if not parsed["format_valid"] else "schema_failure")
    result = {"status": status, "answer": content, "review": parsed, "elapsed": elapsed, "malformed_calls": 0,
              "prompt_version": prepared.prompt_version, "prompt_sha256": prepared.prompt_sha256,
              "original_input_chars": prepared.original_input_chars, "final_input_chars": prepared.final_input_chars,
              "truncated": prepared.truncated, "truncation_reason": prepared.truncation_reason,
              "original_input_tokens": prepared.original_input_tokens, "final_input_tokens": prepared.final_input_tokens,
              "benchmark_serialization_sha256": prepared.benchmark_serialization_sha256, "user_content_sha256": prepared.user_content_sha256,
              "messages_sha256": prepared.messages_sha256,
              "finish_reason": finish_reason,
              "request_attempts": int(response.get("_request_attempts") or getattr(client, "last_request_attempts", 1) or 1)}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key in usage:
            result[key] = usage[key]
    return result
