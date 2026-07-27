from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .config import ROOT
from .context_policy import ContextResult, apply_context

PROMPT_VERSION = "model-only-v2"


@dataclass(frozen=True)
class ReviewInput:
    messages: list[dict[str, str]]
    prompt_version: str
    prompt_sha256: str
    original_input_chars: int
    final_input_chars: int
    truncated: bool
    truncation_reason: str | None


def _template(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def _render(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


class SWEReviewAdapter:
    protocol = "swe-review"
    prompt_file = "swe_review_model_only_v2.txt"

    def prepare(self, task: dict, context_policy: str = "common-32k") -> ReviewInput:
        issue = str(task.get("problem_statement") or "")
        patch = str(task.get("model_patch") or "")
        if not issue or not patch:
            raise ValueError("SWE-Review requires non-empty problem_statement and model_patch")
        rendered = _render(_template(self.prompt_file), {"problem_statement": issue, "model_patch": patch})
        ctx = apply_context(rendered, context_policy, diff=patch)
        system = "You are a precise software reviewer. Follow the requested JSON schema exactly."
        return ReviewInput([{"role": "system", "content": system}, {"role": "user", "content": ctx.text}], PROMPT_VERSION, ctx.sha256, ctx.original_chars, ctx.final_chars, ctx.truncated, ctx.reason)


class MartianReviewAdapter:
    protocol = "martian"
    prompt_file = "martian_model_only_v2.txt"

    def prepare(self, task: dict, context_policy: str = "common-32k") -> ReviewInput:
        patch = str(task.get("patch") or "")
        if not patch:
            raise ValueError("Martian requires a non-empty PR diff")
        rendered = _render(_template(self.prompt_file), {"pr_title": str(task.get("pr_title") or ""), "pr_body": str(task.get("pr_body") or ""), "patch": patch})
        ctx = apply_context(rendered, context_policy, diff=patch)
        system = "You are a precise software reviewer. Follow the requested JSON schema exactly."
        return ReviewInput([{"role": "system", "content": system}, {"role": "user", "content": ctx.text}], PROMPT_VERSION, ctx.sha256, ctx.original_chars, ctx.final_chars, ctx.truncated, ctx.reason)


def prompt_sha256(adapter: object) -> str:
    path = ROOT / "prompts" / getattr(adapter, "prompt_file")
    return hashlib.sha256(path.read_bytes()).hexdigest()
