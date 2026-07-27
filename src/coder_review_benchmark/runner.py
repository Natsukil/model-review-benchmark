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
from .scoring import parse_review


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


def run_review_task(client: ModelClient, task: dict[str, Any]) -> dict[str, Any]:
    code = task.get("old") or task.get("code") or task.get("patch") or task.get("model_patch") or ""
    review = task.get("review") or task.get("problem_statement") or task.get("pr_body") or ""
    prompt = ("Review the following code change. Return only one JSON object with keys decision, summary, and findings. "
              "decision must be exactly approve or reject. Use approve only when there are no actionable correctness issues. "
              "findings must be an array; each finding must contain path, line, severity, category, and description. "
              "If you approve, return an empty findings array. Report correctness issues, not style preferences.\n\n"
              f"CODE/DIFF:\n{code}\n\nREQUEST/REVIEW:\n{review}")
    response, elapsed = client.chat(
        [{"role": "system", "content": "You are a precise code reviewer. Report only actionable, evidence-based issues."}, {"role": "user", "content": prompt}],
        max_tokens=min(client.profile.max_output_tokens, 2048),
    )
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    parsed = parse_review(content)
    return {"status": "completed" if parsed["parseable"] else "format_failure", "answer": content, "review": parsed, "elapsed": elapsed, "malformed_calls": 0}
