from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ROOT, get_model_profile, suite_config
from .data import multi_swe_image_name, prepare_agentic_local_profile, prepare_profile, prepare_review_profile
from .client import ModelClient
from .aggregate import summarize_runs
from .report import write_report
from .runner import run_agent_task, run_review_task
from .judge import score_review
from .tools import DockerWorkspace, evaluate_patch_in_image


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="coder-review-benchmark")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--profile", default="balanced")
    local_prep = sub.add_parser("prepare-agentic-local")
    local_prep.add_argument("--profile", default="docker-local")
    local_prep.add_argument("--count", type=int, default=80)
    local_prep.add_argument("--instance-id", action="append", dest="instance_ids")
    review_prep = sub.add_parser("prepare-review")
    review_prep.add_argument("--profile", default="review-hour")
    review_prep.add_argument("--swe-review-count", type=int, default=500)
    review_prep.add_argument("--martian-count", type=int, default=50)
    review_prep.add_argument("--max-patch-chars", type=int, default=40000)
    doctor = sub.add_parser("doctor"); doctor.add_argument("--profile", default="balanced")
    probe = sub.add_parser("probe"); probe.add_argument("--model", required=True)
    run = sub.add_parser("run"); run.add_argument("--suite", required=True); run.add_argument("--model", required=True); run.add_argument("--profile", default="balanced"); run.add_argument("--concurrency", type=int, default=1); run.add_argument("--limit", type=int); run.add_argument("--max-turns", type=int, default=20)
    rep = sub.add_parser("report"); rep.add_argument("--run-dir", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    summary.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    summary.add_argument("--profile", default="review-hour")
    return p


def _doctor(profile_name: str = "balanced") -> None:
    checks: dict[str, object] = {}
    for model_id in ("qwen2.5-coder-7b", "qwen3-coder-30b", "judge"):
        try:
            profile = get_model_profile(model_id)
            checks[model_id] = {
                "ok": True,
                "model_name": profile.model_name,
                "base_url": profile.base_url,
                "parser": profile.parser,
                "authentication": "bearer" if profile.send_auth else "none",
                "api_key_configured": None if not profile.send_auth else bool(profile.api_key and profile.api_key != "dummy-key"),
                "configured_key_ignored": bool(profile.api_key) if not profile.send_auth else False,
            }
        except Exception as exc:
            checks[model_id] = {"ok": False, "error": str(exc)}
    docker = shutil.which("docker")
    docker_ok = False
    docker_error = "docker executable not found"
    if docker:
        probe = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=10)
        docker_ok = probe.returncode == 0
        docker_error = (probe.stderr or probe.stdout).strip()[-500:]
    checks["docker"] = {"ok": docker_ok, "error": None if docker_ok else docker_error}
    harness_probe = subprocess.run(
        [sys.executable, "-c", "import multi_swe_bench"], capture_output=True, text=True
    )
    checks["multi_swe_bench_python"] = {"ok": harness_probe.returncode == 0, "error": harness_probe.stderr.strip()[-500:] if harness_probe.returncode else None}
    selections = ROOT / "data" / "selections" / profile_name
    for name in ("agentic", "martian", "swe_review", "codereviewqa"):
        path = selections / f"{name}.jsonl"
        count = sum(1 for line in path.open(encoding="utf-8") if line.strip()) if path.exists() else 0
        checks[f"selection_{name}"] = {"ok": path.exists(), "count": count, "profile": profile_name}
    print(json.dumps(checks, ensure_ascii=False, indent=2))


def _probe(model_id: str) -> None:
    profile = get_model_profile(model_id)
    response, elapsed = ModelClient(profile).chat([{"role": "user", "content": "Reply with the single word READY."}])
    print(json.dumps({"model": model_id, "elapsed_seconds": elapsed, "response": response}, ensure_ascii=False, indent=2))


def _fetch_pr_diff(url: str, max_chars: int = 120_000) -> str:
    cache_dir = ROOT / "data" / "cache" / "pr_diffs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()}.diff"
    if cache_path.exists():
        diff = cache_path.read_text(encoding="utf-8", errors="replace")
        return diff if len(diff) <= max_chars else diff[:max_chars] + "\n[DIFF TRUNCATED]"
    diff_url = url.rstrip("/") + ".diff"
    request = urllib.request.Request(diff_url, headers={"Accept": "text/plain", "User-Agent": "coder-review-benchmark"})
    retries = max(0, int(os.getenv("CBM_DIFF_MAX_RETRIES", "3")))
    retry_delay = max(0.0, float(os.getenv("CBM_DIFF_RETRY_DELAY", "1")))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                diff = response.read().decode("utf-8", errors="replace")
            break
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay * (2 ** attempt))
    else:
        raise RuntimeError(
            f"failed to fetch PR diff {url} after {retries + 1} attempts: {last_error}"
        ) from last_error
    cache_path.write_text(diff, encoding="utf-8")
    return diff if len(diff) <= max_chars else diff[:max_chars] + "\n[DIFF TRUNCATED]"


def _decision_correct(decision: object, resolved: object) -> bool:
    value = str(decision or "").lower().replace("-", "_").replace(" ", "_")
    approve = value in {"approve", "approved", "accept", "accepted"}
    expected = bool(resolved)
    return approve == expected


def _approve_decision(decision: object) -> bool:
    value = str(decision or "").lower().replace("-", "_").replace(" ", "_")
    return value in {"approve", "approved", "accept", "accepted"}


def _swe_review_breakdown(rows: list[dict[str, object]], key: str) -> dict[str, dict[str, object]]:
    values: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        values.setdefault(str(row.get(key, "unknown")), []).append(row)
    result: dict[str, dict[str, object]] = {}
    for value, group in sorted(values.items()):
        result[value] = {
            "count": len(group),
            "completion_rate": sum(row.get("status") == "completed" for row in group) / len(group),
            "decision_accuracy": sum(bool(row.get("decision_correct")) for row in group) / len(group),
        }
    return result


def _run(suite: str, model_id: str, profile_name: str, limit: int | None, max_turns: int = 20) -> None:
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if suite == "agentic":
        docker = shutil.which("docker")
        if not docker:
            raise RuntimeError("agentic evaluation requires Docker with WSL integration; run `doctor` first")
        docker_probe = subprocess.run([docker, "info"], capture_output=True, text=True, timeout=10)
        if docker_probe.returncode != 0:
            raise RuntimeError("agentic evaluation requires a running Docker daemon; run `doctor` first")
    profile = get_model_profile(model_id)
    judge_profile = get_model_profile("judge") if suite == "martian" else None
    judge_client = ModelClient(judge_profile, timeout=60) if judge_profile else None
    selection_dir = ROOT / "data" / "selections" / profile_name
    source = selection_dir / ("codereviewqa.jsonl" if suite == "codereviewqa" else f"{suite}.jsonl")
    if not source.exists():
        raise RuntimeError(f"Selection missing: {source}; run prepare first")
    run_dir = ROOT / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{model_id}_{suite}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "suite": suite,
        "profile": profile_name,
        "model_profile": model_id,
        "model_name": profile.model_name,
        "parser": profile.parser,
        "max_output_tokens": profile.max_output_tokens,
        "max_context_tokens": profile.max_context_tokens,
        "judge_model_name": judge_profile.model_name if judge_profile else None,
        "limit": limit,
        "max_turns": max_turns,
        "evaluation_method": "official_image_tests_no_uploads" if suite == "agentic" else None,
        "evaluation_version": "batch-one-to-one-v1" if suite == "martian" else "review-v1",
        "environment": {"qwen_base_url": profile.base_url, "judge_base_url": judge_profile.base_url if judge_profile else None},
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    results_path = run_dir / "results.jsonl"
    run_started = time.monotonic()
    with source.open(encoding="utf-8") as f, results_path.open("w", encoding="utf-8") as output:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break
            item = json.loads(line)
            task = item.get("record", item)
            try:
                if suite == "martian":
                    if not isinstance(task.get("url"), str) or not isinstance(task.get("comments"), list):
                        raise RuntimeError("Martian selection contains a non-golden record; run prepare --profile balanced again")
                    task = dict(task)
                    task["patch"] = _fetch_pr_diff(task["url"])
                    task["review"] = task.get("pr_title", "")
                workspace = run_dir / "workspaces" / str(index)
                workspace.mkdir(parents=True, exist_ok=True)
                if suite in {"codereviewqa", "martian", "swe_review"}:
                    result = run_review_task(ModelClient(profile), task)
                else:
                    image = item.get("image") or multi_swe_image_name(task)
                    command_timeout = int(os.getenv("CBM_AGENTIC_COMMAND_TIMEOUT", "120"))
                    evaluation_timeout = int(os.getenv("CBM_AGENTIC_EVAL_TIMEOUT", "1800"))
                    with DockerWorkspace(image, str(task["repo"]), command_timeout=command_timeout) as tools:
                        result = run_agent_task(
                            ModelClient(profile), profile, task, workspace,
                            max_turns=max_turns, command_timeout=command_timeout, tool_executor=tools,
                        )
                        model_patch = tools.diff()
                    result["image"] = image
                    result["instance_id"] = task.get("instance_id")
                    result["model_patch"] = model_patch
                    result["evaluation"] = evaluate_patch_in_image(
                        image, str(task["repo"]), model_patch, timeout=evaluation_timeout
                    )
                if suite == "martian" and judge_client:
                    result["judge"] = score_review(result["review"], task["comments"], judge_client)
                if suite == "swe_review":
                    result["decision_correct"] = _decision_correct(result.get("review", {}).get("decision"), task.get("resolved"))
                    result.update(
                        {
                            "instance_id": task.get("instance_id"),
                            "generator_model": task.get("generator_model", "unknown"),
                            "difficulty": task.get("difficulty", "unknown"),
                            "expected_resolved": bool(task.get("resolved")),
                        }
                    )
                elif suite == "martian":
                    result.update(
                        {
                            "pr_url": task.get("url"),
                            "golden_finding_count": len(task.get("comments") or []),
                        }
                    )
            except Exception as exc:
                result = {"status": "error", "error": str(exc), "answer": "", "malformed_calls": 0}
            if suite == "swe_review":
                result.setdefault("instance_id", task.get("instance_id"))
                result.setdefault("generator_model", task.get("generator_model", "unknown"))
                result.setdefault("difficulty", task.get("difficulty", "unknown"))
                result.setdefault("expected_resolved", bool(task.get("resolved")))
            elif suite == "martian":
                result.setdefault("pr_url", task.get("url"))
                result.setdefault("golden_finding_count", len(task.get("comments") or []))
            result.update({"index": index, "suite": suite, "model": model_id, "language": item.get("language", "unknown")})
            rows.append(result)
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
    wall_elapsed = time.monotonic() - run_started
    model_elapsed = sum(float(row.get("elapsed") or 0.0) for row in rows)
    if suite == "martian":
        totals = {key: sum(int(row.get("judge", {}).get(key, 0)) for row in rows) for key in ("tp", "fp", "fn")}
        totals["fn"] += sum(int(row.get("golden_finding_count", 0)) for row in rows if not isinstance(row.get("judge"), dict))
        precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
        recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
        totals.update({
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "completion_rate": sum(row.get("status") == "completed" for row in rows) / len(rows) if rows else 0.0,
            "average_findings": sum(len(row.get("review", {}).get("findings", [])) for row in rows) / len(rows) if rows else 0.0,
            "judge_calls": sum(int(row.get("judge", {}).get("judge_calls", 0)) for row in rows),
            "judge_errors": sum(len(row.get("judge", {}).get("errors", [])) for row in rows),
            "unscored_samples": sum(not isinstance(row.get("judge"), dict) for row in rows),
            "model_elapsed_seconds": model_elapsed,
            "judge_elapsed_seconds": sum(float(row.get("judge", {}).get("judge_elapsed", 0.0)) for row in rows),
            "wall_elapsed_seconds": wall_elapsed,
        })
        (run_dir / "metrics.json").write_text(json.dumps({"suite": suite, "judge_model": judge_profile.model_name if judge_profile else None, **totals}, indent=2), encoding="utf-8")
    elif suite == "swe_review":
        completed = sum(row.get("status") == "completed" for row in rows)
        accuracy = sum(bool(row.get("decision_correct")) for row in rows) / len(rows) if rows else 0.0
        valid_rows = [row for row in rows if row.get("status") == "completed" and row.get("review", {}).get("parseable")]
        approved_resolved = sum(_approve_decision(row.get("review", {}).get("decision")) and bool(row.get("expected_resolved")) for row in valid_rows)
        approved_unresolved = sum(_approve_decision(row.get("review", {}).get("decision")) and not bool(row.get("expected_resolved")) for row in valid_rows)
        rejected_resolved = sum(not _approve_decision(row.get("review", {}).get("decision")) and bool(row.get("expected_resolved")) for row in valid_rows)
        rejected_unresolved = sum(not _approve_decision(row.get("review", {}).get("decision")) and not bool(row.get("expected_resolved")) for row in valid_rows)
        (run_dir / "metrics.json").write_text(json.dumps({
            "suite": suite,
            "completion_rate": completed / len(rows) if rows else 0.0,
            "decision_accuracy": accuracy,
            "decision_accuracy_completed": sum(bool(row.get("decision_correct")) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0,
            "confusion_matrix": {
                "approved_resolved": approved_resolved,
                "approved_unresolved": approved_unresolved,
                "rejected_resolved": rejected_resolved,
                "rejected_unresolved": rejected_unresolved,
            },
            "average_findings": sum(len(row.get("review", {}).get("findings", [])) for row in valid_rows) / len(valid_rows) if valid_rows else 0.0,
            "model_elapsed_seconds": model_elapsed,
            "wall_elapsed_seconds": wall_elapsed,
            "by_generator_model": _swe_review_breakdown(rows, "generator_model"),
            "by_difficulty": _swe_review_breakdown(rows, "difficulty"),
        }, indent=2), encoding="utf-8")
    elif suite == "agentic":
        evaluated = [row for row in rows if isinstance(row.get("evaluation"), dict)]
        resolved = sum(bool(row["evaluation"].get("resolved")) for row in evaluated)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "suite": suite,
                    "evaluation_method": "official_image_tests_no_uploads",
                    "evaluated": len(evaluated),
                    "resolved": resolved,
                    "resolve_rate": resolved / len(evaluated) if evaluated else 0.0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    write_report(run_dir)
    print(run_dir)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        _doctor(args.profile)
    elif args.command == "prepare":
        print(prepare_profile(args.profile))
    elif args.command == "prepare-agentic-local":
        print(prepare_agentic_local_profile(
            args.profile,
            args.count,
            instance_ids=set(args.instance_ids) if args.instance_ids else None,
        ))
    elif args.command == "prepare-review":
        print(prepare_review_profile(
            args.profile,
            args.swe_review_count,
            args.martian_count,
            args.max_patch_chars,
        ))
    elif args.command == "probe":
        _probe(args.model)
    elif args.command == "run":
        if args.concurrency != 1:
            raise ValueError("only --concurrency 1 is currently supported; this protects single-model deployments")
        _run(args.suite, args.model, args.profile, args.limit, args.max_turns)
    elif args.command == "report":
        print(write_report(args.run_dir))
    elif args.command == "summarize":
        paths = summarize_runs(args.outputs_dir, args.output_dir, args.profile)
        print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
