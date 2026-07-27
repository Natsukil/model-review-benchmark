from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

MAX_INPUT_CHARS = 100_000
CHAT_TEMPLATE_V1 = "<|im_start|>{role}\n{content}<|im_end|>\n"
CHAT_TEMPLATE_SHA256 = hashlib.sha256(CHAT_TEMPLATE_V1.encode()).hexdigest()


@dataclass(frozen=True)
class ContextResult:
    text: str
    original_chars: int
    final_chars: int
    truncated: bool
    reason: str | None
    sha256: str
    original_tokens: int | None
    final_tokens: int | None
    template_sha256: str = CHAT_TEMPLATE_SHA256


def apply_chat_template(messages: list[dict[str, Any]]) -> str:
    return "".join(CHAT_TEMPLATE_V1.format(role=str(m.get("role", "")), content=str(m.get("content", ""))) for m in messages)


def messages_sha256(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _preserve_diff_header(diff: str, budget: int) -> str:
    if len(diff) <= budget:
        return diff
    if budget <= 0:
        return ""
    lines = diff.splitlines(keepends=True)
    header = "".join(line for line in lines if line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")))
    if len(header) >= budget:
        return header[:budget]
    marker = "\n[DIFF TRUNCATED deterministically]\n"
    body_budget = budget - len(header)
    if body_budget <= len(marker):
        return (header + diff[:body_budget])[:budget]
    body_budget -= len(marker)
    head = body_budget * 3 // 4
    tail = body_budget - head
    omitted = max(0, len(diff) - head - tail)
    return header + diff[:head] + f"\n[DIFF TRUNCATED: omitted {omitted} chars]\n" + diff[-tail:]


def _candidate(text: str, diff: str | None, char_budget: int) -> str:
    if not diff:
        return text[:char_budget]
    marker = "\n\nDIFF:\n"
    split = text.find(marker)
    if split < 0:
        return text[:char_budget]
    prefix = text[:split + len(marker)]
    return prefix + _preserve_diff_header(text[split + len(marker):], max(0, char_budget - len(prefix)))


def apply_context(
    text: str,
    policy: str = "common-100k-char-v1",
    *,
    diff: str | None = None,
    system: str = "",
) -> ContextResult:
    if policy not in {"common-100k-char-v1", "native-context"}:
        raise ValueError("context policy must be common-100k-char-v1 or native-context")
    original_chars = len(text)
    # No model tokenizer is bundled. Token fields therefore remain null and
    # the fallback protocol is explicitly named and measured in characters.
    original_tokens = None
    if policy == "native-context" or original_chars <= MAX_INPUT_CHARS:
        final, reason = text, None
    else:
        final = _candidate(text, diff, MAX_INPUT_CHARS)[:MAX_INPUT_CHARS]
        reason = "common-100k-char-budget"
    final_tokens = None
    return ContextResult(final, original_chars, len(final), len(final) < original_chars, reason, hashlib.sha256(final.encode()).hexdigest(), original_tokens, final_tokens)
