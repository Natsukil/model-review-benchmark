from __future__ import annotations

import json
from typing import Any

from .client import ModelClient
from .tool_parser import parse_json_object


MATCH_PROMPT = """You are evaluating an AI code review finding.

Golden issue:
{golden}

Candidate finding:
{candidate}

Decide whether both describe the same underlying bug or actionable issue.
Accept different wording when the substance is the same. Do not match merely
because they mention the same file or broad topic.

Return only JSON:
{{"match": true, "confidence": 0.0, "reasoning": "brief explanation"}}
"""

BATCH_MATCH_PROMPT = """You are evaluating an AI code review against a golden review.

Golden issues (each may be matched at most once):
{goldens}

Candidate findings (each may be matched at most once):
{candidates}

Match findings only when they identify the same underlying actionable bug. Do
not match merely because they mention the same file or broad topic. Return a
maximum-cardinality one-to-one matching. Omit uncertain pairs.

Return only JSON in this exact shape:
{{"matches":[{{"golden_index":0,"candidate_index":0,"confidence":0.0,"reasoning":"brief explanation"}}]}}
"""

MATCH_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "martian_judge_match_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "match": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
            "required": ["match", "confidence", "reasoning"],
        },
    },
}

BATCH_MATCH_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "martian_judge_batch_v1",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "golden_index": {"type": "integer", "minimum": 0},
                            "candidate_index": {"type": "integer", "minimum": 0},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["golden_index", "candidate_index", "confidence", "reasoning"],
                    },
                }
            },
            "required": ["matches"],
        },
    },
}


def _content(response: dict[str, Any]) -> str:
    return str(((response.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def match_finding(judge: ModelClient, golden: str, candidate: str) -> dict[str, Any]:
    response, elapsed = judge.chat(
        [
            {"role": "system", "content": "You are a precise code-review judge. Always return valid JSON."},
            {"role": "user", "content": MATCH_PROMPT.format(golden=golden, candidate=candidate)},
        ],
        temperature=0,
        response_format=MATCH_RESPONSE_SCHEMA,
    )
    raw = _content(response)
    parsed = parse_json_object(raw)
    if not parsed or not isinstance(parsed.get("match"), bool):
        return {"error": "judge returned invalid JSON", "raw": raw, "elapsed": elapsed}
    return {
        "match": parsed["match"],
        "confidence": float(parsed.get("confidence", 0.0)),
        "reasoning": str(parsed.get("reasoning", "")),
        "elapsed": elapsed,
    }


def finding_text(finding: dict[str, Any]) -> str:
    return json.dumps(
        {
            "path": finding.get("path"),
            "line": finding.get("line"),
            "severity": finding.get("severity"),
            "category": finding.get("category"),
            "description": finding.get("description", ""),
        },
        ensure_ascii=False,
    )


def score_review(review: dict[str, Any], golden_comments: list[dict[str, Any]], judge: ModelClient) -> dict[str, Any]:
    findings = review.get("findings", []) if review.get("parseable") else []
    candidates = [finding_text(item) for item in findings]
    empty = {
        "tp": 0,
        "fp": len(candidates),
        "fn": len(golden_comments),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "matches": [],
        "errors": [],
        "judge_calls": 0,
        "judge_elapsed": 0.0,
    }
    if not golden_comments or not candidates:
        return empty

    goldens = [str(item.get("comment", "")) for item in golden_comments]
    response, elapsed = judge.chat(
        [
            {"role": "system", "content": "You are a precise code-review judge. Always return valid JSON."},
            {
                "role": "user",
                "content": BATCH_MATCH_PROMPT.format(
                    goldens=json.dumps(goldens, ensure_ascii=False, indent=2),
                    candidates=json.dumps(candidates, ensure_ascii=False, indent=2),
                ),
            },
        ],
        temperature=0,
        response_format=BATCH_MATCH_RESPONSE_SCHEMA,
    )
    raw = _content(response)
    parsed = parse_json_object(raw)
    if not parsed or not isinstance(parsed.get("matches"), list):
        empty["errors"] = [{"error": "judge returned invalid batch JSON", "raw": raw}]
        empty["judge_calls"] = 1
        empty["judge_elapsed"] = elapsed
        return empty

    matches: list[dict[str, Any]] = []
    matched_candidates: set[int] = set()
    matched_goldens: set[int] = set()
    errors: list[dict[str, Any]] = []
    for item in parsed["matches"]:
        if not isinstance(item, dict):
            errors.append({"error": "judge match is not an object", "match": item})
            continue
        try:
            gi = int(item["golden_index"])
            ci = int(item["candidate_index"])
        except (KeyError, TypeError, ValueError):
            errors.append({"error": "judge match has invalid indices", "match": item})
            continue
        if not 0 <= gi < len(goldens) or not 0 <= ci < len(candidates):
            errors.append({"error": "judge match index is out of range", "match": item})
            continue
        if gi in matched_goldens or ci in matched_candidates:
            errors.append({"error": "judge returned a non-unique match", "match": item})
            continue
        matched_goldens.add(gi)
        matched_candidates.add(ci)
        matches.append(
            {
                "golden_index": gi,
                "candidate_index": ci,
                "confidence": float(item.get("confidence", 0.0)),
                "reasoning": str(item.get("reasoning", "")),
            }
        )

    if errors:
        for error in errors:
            error.setdefault("raw", raw)

    tp = len(matches)
    fp = len(candidates) - tp
    fn = len(golden_comments) - len(matched_goldens)
    precision = tp / len(candidates) if candidates else 0.0
    recall = tp / len(golden_comments) if golden_comments else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "errors": errors,
        "judge_calls": 1,
        "judge_elapsed": elapsed,
    }
