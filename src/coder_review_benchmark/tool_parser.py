from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]
    source: str
    call_id: str | None = None


@dataclass
class ParseResult:
    tool_calls: list[ParsedToolCall]
    text: str
    error: str | None = None


def _one(obj: Any, source: str) -> ParsedToolCall:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        raise ValueError("tool call must contain a string name")
    args = obj.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return ParsedToolCall(obj["name"], args, source)


def _qwen25_one(obj: Any) -> ParsedToolCall:
    """Normalize textual tool-call shapes emitted by Qwen2.5 templates."""
    if not isinstance(obj, dict):
        raise ValueError("tool call must be a JSON object")

    if isinstance(obj.get("name"), str):
        name = obj["name"]
    elif isinstance(obj.get("tool"), str):
        name = obj["tool"]
    else:
        raise ValueError("tool call must contain a string name or tool")

    if "arguments" in obj:
        args = obj["arguments"]
    elif "parameters" in obj:
        args = obj["parameters"]
    else:
        # Some Qwen2.5/Ollama templates flatten arguments next to "tool".
        args = {key: value for key, value in obj.items() if key not in {"name", "tool"}}

    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError("tool arguments must be a JSON object")
    return ParsedToolCall(name, args, "qwen25_text_fallback")


def _qwen25_python_call(text: str) -> ParsedToolCall | None:
    """Parse ``tool_name(key='literal')`` without evaluating model code."""
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
    candidates = [candidate]
    # Small tool-tuned models sometimes prefix a short explanation before an
    # otherwise valid standalone call. Parse only a complete call-shaped line.
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*\([^\n]*\))\s*$",
            candidate,
            flags=re.MULTILINE,
        )
    )
    expression = None
    for item in candidates:
        try:
            parsed = ast.parse(item, mode="eval").body
        except SyntaxError:
            continue
        if isinstance(parsed, ast.Call) and isinstance(parsed.func, ast.Name):
            expression = parsed
            break
    if expression is None:
        return None
    if expression.args:
        raise ValueError("textual tool calls must use keyword arguments")
    args: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            raise ValueError("textual tool calls cannot use **arguments")
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError("tool arguments must be literals") from exc
    return ParsedToolCall(expression.func.id, args, "qwen25_text_fallback")


def parse_tool_calls(response: dict[str, Any], mode: str = "native_tool_calls") -> ParseResult:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    native = message.get("tool_calls") or []
    if native:
        try:
            calls = []
            for c in native:
                parsed = _one({"name": c["function"]["name"], "arguments": c["function"].get("arguments", "{}")}, "native")
                parsed.call_id = c.get("id")
                calls.append(parsed)
            return ParseResult(calls, message.get("content") or "")
        except (KeyError, TypeError, ValueError) as exc:
            if mode == "native_tool_calls":
                return ParseResult([], message.get("content") or "", f"invalid native tool call: {exc}")
    text = message.get("content") or ""
    if mode != "qwen25_text_fallback":
        return ParseResult([], text)

    json_patterns = [
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
    ]
    for pattern in json_patterns:
        matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            continue
        calls: list[ParsedToolCall] = []
        for match in matches:
            try:
                calls.append(_qwen25_one(json.loads(match.group(1))))
            except (json.JSONDecodeError, ValueError) as exc:
                return ParseResult([], text, f"invalid textual tool call: {exc}")
        return ParseResult(calls, text)

    # Some Ollama templates emit a command-like line and then hallucinate the
    # tool result inside the same code fence. Only the command is authoritative;
    # the benchmark executes the registered tool itself.
    command = re.search(
        r"^\s*>\s*([A-Za-z_][A-Za-z0-9_.-]*)(?:\s+(\{[^\n]*\}))?\s*$",
        text,
        flags=re.MULTILINE,
    )
    if command:
        try:
            args = json.loads(command.group(2)) if command.group(2) else {}
            return ParseResult(
                [
                    _qwen25_one(
                        {"tool": command.group(1), "arguments": args}
                    )
                ],
                text,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            return ParseResult([], text, f"invalid textual tool call: {exc}")

    try:
        python_call = _qwen25_python_call(text)
        if python_call:
            return ParseResult([python_call], text)
    except ValueError as exc:
        return ParseResult([], text, f"invalid textual tool call: {exc}")
    return ParseResult([], text)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a strict JSON object from a structured review response."""
    text = text.strip()
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
