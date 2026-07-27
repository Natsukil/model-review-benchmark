from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

from .tool_parser import parse_json_object

SWE_DECISIONS = {"approve", "request_changes"}
DECISION_ALIASES = {"reject": "request_changes"}
SEVERITIES = {"low", "medium", "high", "critical"}
CATEGORIES = {"correctness", "security", "reliability", "performance", "compatibility", "testing", "maintainability", "other"}


def score_mcqa(answer: str, gold: str) -> float:
    return float(answer.strip().upper()[:1] == gold.strip().upper()[:1])


def score_acr(gold: str, prediction: str) -> dict[str, float]:
    strip = lambda x: re.sub(r"\s+", "", x)
    no_comments = lambda x: re.sub(r"/\*.*?\*/|//.*?$|^\s*#.*?$", "", x, flags=re.M | re.S)
    return {"em": float(prediction.strip() == gold.strip()), "em_no_space": float(strip(prediction) == strip(gold)), "em_no_comment": float(strip(no_comments(prediction)) == strip(no_comments(gold)))}


def normalize_decision(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return DECISION_ALIASES.get(normalized, normalized) if normalized in SWE_DECISIONS | set(DECISION_ALIASES) else None


def normalize_finding(item: dict[str, Any]) -> dict[str, Any]:
    return {"path": item.get("path", ""), "line": item.get("line", item.get("start_line")), "severity": str(item.get("severity", "unknown")).lower(), "category": str(item.get("category", "unknown")).lower(), "description": item.get("description", item.get("comment", ""))}


def _validate_findings(value: Any) -> tuple[bool, list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return False, [], ["findings must be an array"]
    findings: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"findings[{index}] must be an object")
            continue
        finding = normalize_finding(item)
        if not isinstance(finding["path"], str) or not finding["path"]:
            errors.append(f"findings[{index}].path must be a non-empty string")
        if not isinstance(finding["line"], int) or isinstance(finding["line"], bool) or finding["line"] < 1:
            errors.append(f"findings[{index}].line must be a positive integer")
        if not isinstance(finding["description"], str) or not finding["description"].strip():
            errors.append(f"findings[{index}].description must be a non-empty string")
        if finding["severity"] not in SEVERITIES:
            errors.append(f"findings[{index}].severity is invalid")
        if finding["category"] not in CATEGORIES:
            errors.append(f"findings[{index}].category is invalid")
        findings.append(finding)
    return not errors, findings, errors


def parse_review(text: str, protocol: str = "swe") -> dict[str, Any]:
    """Strict model-only review parser. Invalid decisions never become reject."""
    obj = parse_json_object(text)
    base = {"format_valid": False, "schema_valid": False, "parseable": False, "decision": None, "findings": [], "raw": text, "schema_errors": []}
    if not isinstance(obj, dict):
        base["schema_errors"] = ["response is not a JSON object"]
        return base
    base["format_valid"] = True
    findings_ok, findings, errors = _validate_findings(obj.get("findings"))
    base["findings"] = findings
    summary = obj.get("summary")
    if not isinstance(summary, str):
        errors.append("summary must be a string")
    if protocol in {"swe", "swe-review"}:
        decision = normalize_decision(obj.get("decision"))
        base["decision"] = decision
        if decision is None:
            errors.append("decision must be approve or request_changes")
    base["schema_errors"] = errors
    base["schema_valid"] = findings_ok and not errors
    base["parseable"] = base["schema_valid"]
    return base


def parse_martian_review(text: str) -> dict[str, Any]:
    return parse_review(text, protocol="martian")


def _rates(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"precision": precision, "recall": recall, "f1": f1, "specificity": specificity, "mcc": (tp * tn - fp * fn) / denom if denom else 0.0, "total": total}


def calculate_swe_metrics(rows: list[dict[str, Any]], *, include_breakdowns: bool = True) -> dict[str, Any]:
    n = len(rows)
    format_valid = sum(bool(r.get("review", {}).get("format_valid")) for r in rows)
    schema_valid = sum(bool(r.get("review", {}).get("schema_valid")) for r in rows)
    valid = [r for r in rows if bool(r.get("review", {}).get("schema_valid")) and r.get("review", {}).get("decision") in SWE_DECISIONS]
    tp = sum(r.get("review", {}).get("decision") == "request_changes" and not bool(r.get("expected_resolved")) for r in valid)
    fn = sum(r.get("review", {}).get("decision") == "approve" and not bool(r.get("expected_resolved")) for r in valid)
    tn = sum(r.get("review", {}).get("decision") == "approve" and bool(r.get("expected_resolved")) for r in valid)
    fp = sum(r.get("review", {}).get("decision") == "request_changes" and bool(r.get("expected_resolved")) for r in valid)
    valid_correct = sum((r.get("review", {}).get("decision") == ("approve" if bool(r.get("expected_resolved")) else "request_changes")) for r in valid)
    all_correct = sum((r.get("review", {}).get("decision") == ("approve" if bool(r.get("expected_resolved")) else "request_changes")) for r in rows)
    rates = _rates(tp, fp, fn, tn)
    approve = sum(r.get("review", {}).get("decision") == "approve" for r in rows)
    changes = sum(r.get("review", {}).get("decision") == "request_changes" for r in rows)
    unresolved = sum(not bool(r.get("expected_resolved")) for r in rows)
    result = {"sample_count": n, "format_completion_rate": format_valid / n if n else 0.0, "schema_completion_rate": schema_valid / n if n else 0.0, "decision_accuracy_all": all_correct / n if n else 0.0, "decision_accuracy_valid": valid_correct / len(valid) if valid else 0.0, "balanced_accuracy": (rates["recall"] + rates["specificity"]) / 2, "defect_recall": tp / unresolved if unresolved else 0.0, "false_acceptance_rate": fn / unresolved if unresolved else 0.0, "false_rejection_rate": fp / (n - unresolved) if n - unresolved else 0.0, "MCC": rates["mcc"], "approve_rate": approve / n if n else 0.0, "request_changes_rate": changes / n if n else 0.0, "invalid_decision_rate": 1 - len(valid) / n if n else 0.0, "confusion_matrix": {"approve_resolved": tn, "request_changes_resolved": fp, "approve_unresolved": fn, "request_changes_unresolved": tp}}
    if include_breakdowns:
        result["by_generator_model"] = _group_swe(rows, "generator_model")
        result["by_difficulty"] = _group_swe(rows, "difficulty")
    return result


def _group_swe(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "unknown")), []).append(row)
    return {name: calculate_swe_metrics(group, include_breakdowns=False) for name, group in sorted(groups.items())}


def calculate_martian_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [r for r in rows if isinstance(r.get("judge"), dict)]
    tp = sum(int(r.get("judge", {}).get("tp", 0)) for r in judged)
    fp = sum(int(r.get("judge", {}).get("fp", 0)) for r in judged)
    fn = sum(int(r.get("judge", {}).get("fn", 0)) for r in judged)
    fn += sum(int(r.get("golden_finding_count", 0)) for r in rows if not isinstance(r.get("judge"), dict))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    per_pr = [(float(r.get("judge", {}).get("precision", 0.0)), float(r.get("judge", {}).get("recall", 0.0)), float(r.get("judge", {}).get("f1", 0.0))) for r in judged]
    repos: dict[str, list[dict[str, Any]]] = {}
    languages: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        url = str(row.get("pr_url", ""))
        match = re.search(r"github\.com/([^/]+/[^/]+)/pull/", url)
        repos.setdefault(match.group(1).lower() if match else "unknown", []).append(row)
        languages.setdefault(str(row.get("language", "unknown")), []).append(row)
    def group_metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        judged_group = [r for r in group if isinstance(r.get("judge"), dict)]
        gtp = sum(int(r.get("judge", {}).get("tp", 0)) for r in judged_group)
        gfp = sum(int(r.get("judge", {}).get("fp", 0)) for r in judged_group)
        gfn = sum(int(r.get("judge", {}).get("fn", 0)) for r in judged_group)
        gfn += sum(int(r.get("golden_finding_count", 0)) for r in group if not isinstance(r.get("judge"), dict))
        gp = gtp / (gtp + gfp) if gtp + gfp else 0.0
        gr = gtp / (gtp + gfn) if gtp + gfn else 0.0
        return {"sample_count": len(group), "micro_precision": gp, "micro_recall": gr, "micro_f1": 2 * gp * gr / (gp + gr) if gp + gr else 0.0}
    macro_precision = sum(v[0] for v in per_pr) / len(per_pr) if per_pr else 0.0
    macro_recall = sum(v[1] for v in per_pr) / len(per_pr) if per_pr else 0.0
    macro_f1 = sum(v[2] for v in per_pr) / len(per_pr) if per_pr else 0.0
    return {"sample_count": len(rows), "tp": tp, "fp": fp, "fn": fn, "raw_tp": tp, "raw_fp": fp, "raw_fn": fn, "micro_precision": precision, "micro_recall": recall, "micro_f1": f1, "precision": precision, "recall": recall, "f1": f1, "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1, "per_pr_macro_precision": macro_precision, "per_pr_macro_recall": macro_recall, "per_pr_macro_f1": macro_f1, "average_findings": sum(len(r.get("review", {}).get("findings", [])) for r in rows) / len(rows) if rows else 0.0, "zero_finding_prs": sum(not r.get("review", {}).get("findings") for r in rows), "judge_calls": sum(int(r.get("judge", {}).get("judge_calls", 0)) for r in rows if isinstance(r.get("judge"), dict)), "judge_errors": sum(len(r.get("judge", {}).get("errors", [])) for r in rows if isinstance(r.get("judge"), dict)), "judge_error_rate": sum(bool(r.get("judge", {}).get("errors")) for r in rows if isinstance(r.get("judge"), dict)) / len(judged) if judged else 0.0, "unscored_samples": len(rows) - len(judged), "per_repository_metrics": {name: group_metrics(group) for name, group in sorted(repos.items())}, "by_language": {name: group_metrics(group) for name, group in sorted(languages.items())}}


def aggregate_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(row.get(key, "unknown")) for row in rows))
