from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .tool_parser import parse_json_object


def score_mcqa(answer: str, gold: str) -> float:
    return float(answer.strip().upper()[:1] == gold.strip().upper()[:1])


def score_acr(gold: str, prediction: str) -> dict[str, float]:
    strip = lambda x: re.sub(r"\s+", "", x)
    no_comments = lambda x: re.sub(r"/\*.*?\*/|//.*?$|^\s*#.*?$", "", x, flags=re.M | re.S)
    return {"em": float(prediction.strip() == gold.strip()), "em_no_space": float(strip(prediction) == strip(gold)), "em_no_comment": float(strip(no_comments(prediction)) == strip(no_comments(gold)))}


def normalize_finding(item: dict[str, Any]) -> dict[str, Any]:
    return {"path": item.get("path", ""), "line": item.get("line", item.get("start_line")), "severity": str(item.get("severity", "unknown")).lower(), "category": str(item.get("category", "unknown")).lower(), "description": item.get("description", item.get("comment", ""))}


def parse_review(text: str) -> dict[str, Any]:
    obj = parse_json_object(text)
    if not obj:
        return {"parseable": False, "raw": text, "findings": []}
    findings = obj.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    return {"parseable": True, "decision": obj.get("decision"), "findings": [normalize_finding(x) for x in findings if isinstance(x, dict)]}


def aggregate_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))

