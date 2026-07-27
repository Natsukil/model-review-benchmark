from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any

MAX_CONTEXT_TOKENS = 32_768
OUTPUT_RESERVED_TOKENS = 4_096
MAX_INPUT_TOKENS = MAX_CONTEXT_TOKENS - OUTPUT_RESERVED_TOKENS
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
    original_tokens: int
    final_tokens: int
    template_sha256: str = CHAT_TEMPLATE_SHA256


def apply_chat_template(messages: list[dict[str, Any]]) -> str:
    return "".join(CHAT_TEMPLATE_V1.format(role=str(m.get("role", "")), content=str(m.get("content", ""))) for m in messages)


def count_tokens(text: str) -> int:
    """Deterministic tokenizer fallback shared by every model in the fair lane.

    It is deliberately conservative for UTF-8 text and is applied after the
    benchmark chat template. If a deployment supplies a tokenizer, it may be
    used for diagnostics, but it must not alter the frozen fair-lane messages.
    """
    encoded = text.encode("utf-8")
    if not encoded:
        return 0
    # A conservative byte-BPE approximation: no fewer than one token per four
    # bytes, plus standalone whitespace/control boundaries.
    boundaries = len(re.findall(r"\s+", text))
    return max(1, math.ceil(len(encoded) / 4) + boundaries // 8)


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
    policy: str = "common-32k",
    *,
    diff: str | None = None,
    system: str = "",
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> ContextResult:
    if policy not in {"common-32k", "native-context"}:
        raise ValueError("context policy must be common-32k or native-context")
    original_chars = len(text)
    original_tokens = count_tokens(apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": text}]))
    if policy == "native-context" or original_tokens <= max_input_tokens:
        final, reason = text, None
    else:
        lo, hi = 0, len(text)
        final = text[:1]
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = _candidate(text, diff, mid)
            tokens = count_tokens(apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": candidate}]))
            if tokens <= max_input_tokens:
                final = candidate
                lo = mid + 1
            else:
                hi = mid - 1
        reason = "common-32k-token-budget"
    final_tokens = count_tokens(apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": final}]))
    # Defensive correction for marker/header edge cases.
    while policy == "common-32k" and final_tokens > max_input_tokens and final:
        final = final[:-max(1, len(final) // 100)]
        final_tokens = count_tokens(apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": final}]))
    return ContextResult(final, original_chars, len(final), len(final) < original_chars, reason, hashlib.sha256(final.encode()).hexdigest(), original_tokens, final_tokens)
