from __future__ import annotations

from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class ContextResult:
    text: str
    original_chars: int
    final_chars: int
    truncated: bool
    reason: str | None
    sha256: str


def _preserve_diff_header(diff: str, budget: int) -> str:
    if len(diff) <= budget:
        return diff
    lines = diff.splitlines(keepends=True)
    header: list[str] = []
    for line in lines:
        if line.startswith(("diff --git ", "--- ", "+++ ", "@@ ")):
            header.append(line)
    prefix = "".join(header)
    if len(prefix) >= budget:
        return prefix[:budget]
    remaining = budget - len(prefix)
    # Keep the beginning and end of the diff deterministically; headers remain visible.
    if remaining < 80:
        return prefix + diff[:remaining]
    head = remaining * 3 // 4
    tail = remaining - head
    return prefix + diff[:head] + f"\n[DIFF TRUNCATED: omitted {len(diff) - head - tail} chars]\n" + diff[-tail:]


def apply_context(text: str, policy: str = "common-32k", *, diff: str | None = None) -> ContextResult:
    if policy not in {"common-32k", "native-context"}:
        raise ValueError("context policy must be common-32k or native-context")
    original = len(text)
    if policy == "native-context":
        final = text
        reason = None
    else:
        # Character budget is deliberately explicit; it is not a token estimate.
        budget = 100_000
        if original <= budget:
            final, reason = text, None
        else:
            marker = "\n\n[CONTEXT TRUNCATED deterministically]\n\n"
            # Protect the task/description prefix and truncate the diff section.
            split = text.find("\n\nDIFF:\n")
            if split < 0:
                final = text[:budget]
            else:
                prefix = text[: split + len("\n\nDIFF:\n")]
                final = prefix + _preserve_diff_header(text[split + len("\n\nDIFF:\n"):], max(1, budget - len(prefix) - len(marker)))
            final = final[:budget]
            reason = "common-32k-character-budget"
    return ContextResult(final, original, len(final), len(final) < original, reason, hashlib.sha256(final.encode()).hexdigest())
