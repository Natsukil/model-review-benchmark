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
from .data import multi_swe_image_name, prepare_agentic_local_profile, prepare_profile, prepare_review_profile, prepare_swe_v2_profiles, validate_selection
from .client import ModelClient
from .aggregate import summarize_runs
from .report import write_report
from .runner import run_agent_task, run_review_task
from .scoring import calculate_swe_metrics, normalize_decision
from .adapters import MartianReviewAdapter, SWEReviewAdapter, prompt_sha256
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
    sub.add_parser("prepare-review-v2")
    validate = sub.add_parser("validate-selection")
    validate.add_argument("--profile", required=True)
    validate.add_argument("--suite", default="swe_review", choices=["swe_review", "martian"])
    doctor = sub.add_parser("doctor"); doctor.add_argument("--profile", default="balanced")
    probe = sub.add_parser("probe"); probe.add_argument("--model", required=True)
    run = sub.add_parser("run"); run.add_argument("--suite", required=True); run.add_argument("--model", required=True); run.add_argument("--profile", default="balanced"); run.add_argument("--concurrency", type=int, default=1); run.add_argument("--limit", type=int); run.add_argument("--max-turns", type=int, default=20); run.add_argument("--context-policy", choices=["common-100k-char-v1", "native-context"], default="common-100k-char-v1"); run.add_argument("--experiment-id")
    rep = sub.add_parser("report"); rep.add_argument("--run-dir", type=Path, required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    summary.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    summary.add_argument("--profile", default="review-hour")
    matrix = sub.add_parser("run-matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--report", action="store_true")
    matrix.add_argument("--dry-run", action="store_true")
    return p


def _doctor(profile_name: str = "balanced") -> None:
    checks: dict[str, object] = {}
    for model_id in ("qwen2.5-coder-7b", "qwen3-coder-30b", "qwen3-coder-next-80b", "judge"):
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


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_correct(decision: object, resolved: object) -> bool:
    value = normalize_decision(decision)
    if value is None:
        return False
    return value == ("approve" if bool(resolved) else "request_changes")


def _approve_decision(decision: object) -> bool:
    return normalize_decision(decision) == "approve"


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


def _run(suite: str, model_id: str, profile_name: str, limit: int | None, max_turns: int = 20, context_policy: str = "common-100k-char-v1", experiment_id: str | None = None) -> None:
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
    judge_profile = None  # Martian judging is intentionally deferred to run-matrix.
    selection_dir = ROOT / "data" / "selections" / profile_name
    source = selection_dir / ("codereviewqa.jsonl" if suite == "codereviewqa" else f"{suite}.jsonl")
    if not source.exists():
        raise RuntimeError(f"Selection missing: {source}; run prepare first")
    run_dir = ROOT / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{model_id}_{suite}"
    run_dir.mkdir(parents=True, exist_ok=True)
    review_adapter = MartianReviewAdapter() if suite == "martian" else SWEReviewAdapter()
    dataset_manifest = selection_dir / "manifest.json"
    if not dataset_manifest.exists():
        dataset_manifest = selection_dir / "review_manifest.json"
    manifest = {
        "experiment_id": experiment_id or f"standalone-{run_dir.name}",
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
        "evaluation_version": "model-review-v2" if suite in {"martian", "swe_review"} else ("official_image_tests_v1" if suite == "agentic" else "legacy"),
        "context_policy": context_policy if suite in {"martian", "swe_review"} else None,
        "max_input_chars": 100000 if suite in {"martian", "swe_review"} and context_policy == "common-100k-char-v1" else None,
        "output_reserved_tokens": 4096 if suite in {"martian", "swe_review"} else None,
        "structured_output": profile.structured_output if suite in {"martian", "swe_review"} else None,
        "prompt_version": "model-only-v2" if suite in {"martian", "swe_review"} else None,
        "prompt_sha256": prompt_sha256(review_adapter) if suite in {"martian", "swe_review"} else None,
        "dataset_manifest_sha256": _file_sha256(dataset_manifest),
        "generation_settings": {"temperature": profile.temperature, "top_p": profile.top_p, "seed": profile.seed, "stream": profile.stream, "repeat_penalty": profile.repeat_penalty, "presence_penalty": profile.presence_penalty, "frequency_penalty": profile.frequency_penalty, "structured_output": profile.structured_output},
        "model_artifact": {"filename": os.getenv("CBM_MODEL_ARTIFACT_FILENAME"), "sha256": os.getenv("CBM_MODEL_ARTIFACT_SHA256"), "quantization": os.getenv("CBM_MODEL_QUANTIZATION"), "serving_engine": os.getenv("CBM_SERVING_ENGINE"), "serving_engine_version": os.getenv("CBM_SERVING_ENGINE_VERSION"), "chat_template": os.getenv("CBM_CHAT_TEMPLATE")},
        "dataset_profile": profile_name,
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
                    task["pr_body"] = task.get("pr_body", task.get("body", ""))
                workspace = run_dir / "workspaces" / str(index)
                workspace.mkdir(parents=True, exist_ok=True)
                if suite in {"codereviewqa", "martian", "swe_review"}:
                    result = run_review_task(ModelClient(profile), task, protocol="martian" if suite == "martian" else "swe", context_policy=context_policy)
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
                if suite == "swe_review":
                    result["decision_correct"] = result.get("review", {}).get("decision") == ("approve" if bool(task.get("resolved")) else "request_changes")
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
        metrics = {"suite": suite, "phase": "generate", "sample_count": len(rows), "format_completion_rate": sum(bool(row.get("review", {}).get("format_valid")) for row in rows) / len(rows) if rows else 0.0, "schema_completion_rate": sum(bool(row.get("review", {}).get("schema_valid")) for row in rows) / len(rows) if rows else 0.0, "completion_rate": sum(row.get("status") == "completed" for row in rows) / len(rows) if rows else 0.0, "average_findings": sum(len(row.get("review", {}).get("findings", [])) for row in rows) / len(rows) if rows else 0.0, "zero_finding_prs": sum(not row.get("review", {}).get("findings") for row in rows), "model_elapsed_seconds": model_elapsed, "wall_elapsed_seconds": wall_elapsed}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    elif suite == "swe_review":
        metrics = calculate_swe_metrics(rows)
        (run_dir / "metrics.json").write_text(json.dumps({
            "suite": suite,
            **metrics,
            "completion_rate": metrics["schema_completion_rate"],
            "decision_accuracy": metrics["decision_accuracy_all"],
            "model_elapsed_seconds": model_elapsed,
            "wall_elapsed_seconds": wall_elapsed,
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
        _run(args.suite, args.model, args.profile, args.limit, args.max_turns, args.context_policy, args.experiment_id)
    elif args.command == "prepare-review-v2":
        print(json.dumps({key: str(value) for key, value in prepare_swe_v2_profiles().items()}, ensure_ascii=False, indent=2))
    elif args.command == "validate-selection":
        print(json.dumps(validate_selection(args.profile, args.suite), ensure_ascii=False, indent=2))
    elif args.command == "report":
        print(write_report(args.run_dir))
    elif args.command == "summarize":
        paths = summarize_runs(args.outputs_dir, args.output_dir, args.profile)
        print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2))
    elif args.command == "run-matrix":
        from .experiment import run_matrix
        result = run_matrix(args.config, resume=args.resume, report=args.report, dry_run=args.dry_run)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result.get("failures") else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
