from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

MAX_INPUT_CHARS = 100_000


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


def messages_sha256(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _head_tail(text: str, budget: int, label: str) -> str:
    if len(text) <= budget:
        return text
    marker = f"\n[{label} TRUNCATED]\n"
    if budget <= len(marker) + 2:
        return text[:budget]
    remaining = budget - len(marker)
    head = remaining // 2
    return text[:head] + marker + text[-(remaining - head):]


def _preserve_diff_header(diff: str, budget: int) -> str:
    if len(diff) <= budget:
        return diff
    if budget <= 0:
        return ""
    lines = diff.splitlines(keepends=True)
    header = "".join(line for line in lines if line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")))
    # Reserve bounded space for file/hunk headers, then deterministically keep
    # both the beginning and end of the complete diff.
    header_budget = min(len(header), budget // 3)
    kept_headers = _head_tail(header, header_budget, "DIFF HEADERS") if header_budget else ""
    kept_body = _head_tail(diff, budget - len(kept_headers), "DIFF")
    return kept_headers + kept_body


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
