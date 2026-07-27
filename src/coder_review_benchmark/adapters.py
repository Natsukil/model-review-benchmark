from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

from .config import ROOT
from .context_policy import CHAT_TEMPLATE_SHA256, ContextResult, apply_context, messages_sha256

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
    original_input_tokens: int | None
    final_input_tokens: int | None
    template_sha256: str
    user_content_sha256: str
    messages_sha256: str


FINDING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "path": {"type": "string"},
        "line": {"type": "integer", "minimum": 1},
        "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        "category": {"type": "string", "enum": ["correctness", "security", "reliability", "performance", "compatibility", "testing", "maintainability", "other"]},
        "description": {"type": "string"},
    },
    "required": ["path", "line", "severity", "category", "description"],
}

SWE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "swe_review_v2", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"decision": {"type": "string", "enum": ["approve", "request_changes"]}, "summary": {"type": "string"}, "findings": {"type": "array", "items": FINDING_SCHEMA}}, "required": ["decision", "summary", "findings"]}},
}

MARTIAN_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {"name": "martian_review_v2", "strict": True, "schema": {"type": "object", "additionalProperties": False, "properties": {"summary": {"type": "string"}, "findings": {"type": "array", "items": FINDING_SCHEMA}}, "required": ["summary", "findings"]}},
}


def _template(name: str) -> str:
    return (ROOT / "prompts" / name).read_text(encoding="utf-8")


def _render(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


class SWEReviewAdapter:
    protocol = "swe-review"
    prompt_file = "swe_review_model_only_v2.txt"
    response_format = SWE_RESPONSE_SCHEMA

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256((ROOT / "prompts" / self.prompt_file).read_bytes()).hexdigest()

    def prepare(self, task: dict, context_policy: str = "common-100k-char-v1") -> ReviewInput:
        issue = str(task.get("problem_statement") or "")
        patch = str(task.get("model_patch") or "")
        if not issue or not patch:
            raise ValueError("SWE-Review requires non-empty problem_statement and model_patch")
        rendered = _render(_template(self.prompt_file), {"problem_statement": issue, "model_patch": patch})
        system = "You are a precise software reviewer. Follow the requested JSON schema exactly."
        ctx = apply_context(rendered, context_policy, diff=patch, system=system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": ctx.text}]
        return ReviewInput(messages, PROMPT_VERSION, self.prompt_sha256, ctx.original_chars, ctx.final_chars, ctx.truncated, ctx.reason, ctx.original_tokens, ctx.final_tokens, CHAT_TEMPLATE_SHA256, ctx.sha256, messages_sha256(messages))


class MartianReviewAdapter:
    protocol = "martian"
    prompt_file = "martian_model_only_v2.txt"
    response_format = MARTIAN_RESPONSE_SCHEMA

    def prepare(self, task: dict, context_policy: str = "common-100k-char-v1") -> ReviewInput:
        patch = str(task.get("patch") or "")
        if not patch:
            raise ValueError("Martian requires a non-empty PR diff")
        rendered = _render(_template(self.prompt_file), {"pr_title": str(task.get("pr_title") or ""), "pr_body": str(task.get("pr_body") or ""), "patch": patch})
        system = "You are a precise software reviewer. Follow the requested JSON schema exactly."
        ctx = apply_context(rendered, context_policy, diff=patch, system=system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": ctx.text}]
        return ReviewInput(messages, PROMPT_VERSION, self.prompt_sha256, ctx.original_chars, ctx.final_chars, ctx.truncated, ctx.reason, ctx.original_tokens, ctx.final_tokens, CHAT_TEMPLATE_SHA256, ctx.sha256, messages_sha256(messages))

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256((ROOT / "prompts" / self.prompt_file).read_bytes()).hexdigest()

def prompt_sha256(adapter: object) -> str:
    path = ROOT / "prompts" / getattr(adapter, "prompt_file")
    return hashlib.sha256(path.read_bytes()).hexdigest()
