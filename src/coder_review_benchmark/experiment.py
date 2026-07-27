from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from .adapters import MartianReviewAdapter, SWEReviewAdapter, prompt_sha256
from .client import ModelClient
from .config import ROOT, ModelProfile, get_model_profile
from .data import validate_selection
from .experiment_report import generate_experiment_report
from .judge import BATCH_MATCH_PROMPT, score_review
from .lmstudio import LMStudioLifecycle
from .report import write_report
from .runner import run_review_task
from .scoring import calculate_martian_metrics, calculate_swe_metrics


@dataclass(frozen=True)
class ExperimentTask:
    model_profile: str
    suite: str
    dataset_profile: str
    limit: int | None = None

    @property
    def id(self) -> str:
        return "__".join((self.model_profile, self.suite, self.dataset_profile))


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if not isinstance(value, str):
        return value
    pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
    return pattern.sub(lambda match: os.getenv(match.group(1), match.group(2) or ""), value)


def load_experiment_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = _expand(raw)
    if not isinstance(config, dict) or not config.get("experiment_id"):
        raise ValueError("experiment config requires experiment_id")
    if not isinstance(config.get("models"), list) or not config["models"]:
        raise ValueError("experiment config requires a non-empty models array")
    if not isinstance(config.get("datasets"), list) or not config["datasets"]:
        raise ValueError("experiment config requires a non-empty datasets array")
    return config


def build_task_matrix(config: dict[str, Any]) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    for model in config["models"]:
        model_id = str(model.get("profile") if isinstance(model, dict) else model)
        for dataset in config["datasets"]:
            if not isinstance(dataset, dict) or not dataset.get("suite") or not dataset.get("profile"):
                raise ValueError("each dataset requires suite and profile")
            tasks.append(ExperimentTask(model_id, str(dataset["suite"]), str(dataset["profile"]), int(dataset["limit"]) if dataset.get("limit") is not None else None))
    return tasks


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _golden_identity(record: dict[str, Any]) -> tuple[str, str]:
    pr_url = str(record.get("url") or record.get("pr_url") or "")
    sample_id = str(record.get("sample_id") or hashlib.sha256(pr_url.encode("utf-8")).hexdigest())
    return sample_id, _canonical_sha256(record)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _selection_path(task: ExperimentTask, root: Path) -> Path:
    return root / "data" / "selections" / task.dataset_profile / f"{task.suite}.jsonl"


def _manifest_path(task: ExperimentTask, root: Path) -> Path:
    directory = root / "data" / "selections" / task.dataset_profile
    return directory / "manifest.json"


def _common_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "truncated_samples": sum(bool(row.get("truncated")) for row in rows),
        "run_errors": sum(row.get("status") == "error" for row in rows),
        "model_elapsed_seconds": sum(float(row.get("elapsed") or 0.0) for row in rows),
    }


def generate_review_run(
    experiment_dir: Path,
    experiment_id: str,
    task: ExperimentTask,
    profile: ModelProfile,
    *,
    context_policy: str,
    root: Path = ROOT,
    resume: bool = False,
    client_factory: Callable[[ModelProfile], Any] = ModelClient,
    model_metadata: dict[str, Any] | None = None,
) -> Path:
    source = _selection_path(task, root)
    if not source.exists():
        raise RuntimeError(f"selection missing: {source}")
    run_dir = experiment_dir / "runs" / task.id
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter = MartianReviewAdapter() if task.suite == "martian" else SWEReviewAdapter()
    manifest_path = _manifest_path(task, root)
    generation_settings = {key: getattr(profile, key) for key in ("temperature", "top_p", "seed", "stream", "repeat_penalty", "presence_penalty", "frequency_penalty", "structured_output")}
    resume_fields = {
        "evaluation_version": "model-review-v2",
        "model_profile": task.model_profile,
        "model_name": profile.model_name,
        "prompt_sha256": prompt_sha256(adapter),
        "selection_jsonl_sha256": _sha256(source),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "context_policy": context_policy,
        "generation_settings": generation_settings,
        "model_artifact_sha256": (model_metadata or {}).get("sha256") or None,
        "limit": task.limit,
    }
    manifest = {
        "experiment_id": experiment_id,
        "model_profile": task.model_profile,
        "model_name": profile.model_name,
        "suite": task.suite,
        "dataset_profile": task.dataset_profile,
        "evaluation_version": "model-review-v2",
        "phase": "generate",
        "context_policy": context_policy,
        "max_context_tokens": profile.max_context_tokens,
        "max_input_chars": 100000 if context_policy == "common-100k-char-v1" else None,
        "max_output_tokens": profile.max_output_tokens,
        "generation_settings": generation_settings,
        "prompt_version": "model-only-v2",
        "prompt_sha256": prompt_sha256(adapter),
        "dataset_manifest_sha256": _sha256(manifest_path),
        "model_artifact": model_metadata or {},
        "limit": task.limit,
        "resume_fingerprint": {**resume_fields, "sha256": _canonical_sha256(resume_fields)},
    }
    existing_manifest = run_dir / "run_manifest.json"
    if existing_manifest.exists():
        if not resume:
            raise RuntimeError("run directory already contains a manifest; use --resume")
        old = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if old.get("experiment_id") != experiment_id:
            raise RuntimeError("refusing to resume a run from another experiment_id")
        if old.get("resume_fingerprint") != manifest["resume_fingerprint"]:
            raise RuntimeError("resume fingerprint mismatch; prompt, data, policy, settings, model artifact, or evaluation version changed")
    else:
        _write_json(existing_manifest, manifest)
    results_path = run_dir / "results.jsonl"
    completed: dict[int, dict[str, Any]] = {}
    if resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[int(row["index"])] = row
    mode = "a" if completed else "w"
    client = client_factory(profile)
    with source.open(encoding="utf-8") as input_file, results_path.open(mode, encoding="utf-8") as output:
        for index, line in enumerate(input_file):
            if task.limit is not None and index >= task.limit:
                break
            if index in completed:
                continue
            item = json.loads(line)
            record = item.get("record", item)
            try:
                prepared_task = record
                if task.suite == "martian":
                    from .cli import _fetch_pr_diff
                    prepared_task = dict(record)
                    prepared_task["patch"] = _fetch_pr_diff(str(record["url"]))
                    prepared_task["pr_body"] = record.get("pr_body", record.get("body", ""))
                result = run_review_task(client, prepared_task, protocol="martian" if task.suite == "martian" else "swe", context_policy=context_policy)
            except Exception as exc:
                result = {"status": "error", "error": str(exc), "answer": "", "review": {"format_valid": False, "schema_valid": False, "findings": []}, "request_attempts": int(getattr(client, "last_request_attempts", 0) or 0)}
            result.update({"index": index, "suite": task.suite, "model": task.model_profile, "language": item.get("language", "unknown")})
            if task.suite == "swe_review":
                result.update({"instance_id": record.get("instance_id"), "generator_model": record.get("generator_model", "unknown"), "difficulty": record.get("difficulty", "unknown"), "expected_resolved": bool(record.get("resolved"))})
                result["decision_correct"] = result.get("review", {}).get("decision") == ("approve" if bool(record.get("resolved")) else "request_changes")
            else:
                sample_id, golden_record_sha256 = _golden_identity(record)
                result.update({"sample_id": sample_id, "pr_url": record.get("url"), "golden_record_sha256": golden_record_sha256, "golden_finding_count": len(record.get("comments") or [])})
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            completed[index] = result
    rows = [completed[index] for index in sorted(completed)]
    if task.suite == "swe_review":
        metrics = {"suite": task.suite, "phase": "generate", **calculate_swe_metrics(rows)}
    else:
        metrics = {"suite": task.suite, "phase": "generate", "sample_count": len(rows), "format_completion_rate": sum(bool(row.get("review", {}).get("format_valid")) for row in rows) / len(rows) if rows else 0.0, "schema_completion_rate": sum(bool(row.get("review", {}).get("schema_valid")) for row in rows) / len(rows) if rows else 0.0, "average_findings": sum(len(row.get("review", {}).get("findings", [])) for row in rows) / len(rows) if rows else 0.0, "zero_finding_prs": sum(not row.get("review", {}).get("findings") for row in rows)}
    metrics.update(_common_metrics(rows))
    _write_json(run_dir / "metrics.json", metrics)
    write_report(run_dir)
    return run_dir


def judge_martian_run(run_dir: Path, task: ExperimentTask, judge_profile: ModelProfile, *, root: Path = ROOT, resume: bool = False, client_factory: Callable[[ModelProfile], Any] = ModelClient) -> None:
    results_path = run_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_rows = [json.loads(line) for line in _selection_path(task, root).read_text(encoding="utf-8").splitlines() if line.strip()]
    golden_by_id: dict[str, tuple[dict[str, Any], str]] = {}
    for item in source_rows:
        record = item.get("record", item)
        sample_id, record_sha = _golden_identity(record)
        if sample_id in golden_by_id:
            raise RuntimeError(f"duplicate Martian sample_id: {sample_id}")
        golden_by_id[sample_id] = (record, record_sha)
    judge = client_factory(judge_profile)
    for row in rows:
        if resume and isinstance(row.get("judge"), dict) and not row["judge"].get("errors"):
            continue
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id not in golden_by_id:
            raise RuntimeError(f"Martian result has no matching stable sample_id: {sample_id or '<missing>'}")
        record, current_sha = golden_by_id[sample_id]
        if str(row.get("pr_url") or "") != str(record.get("url") or ""):
            raise RuntimeError(f"Martian pr_url mismatch for sample_id {sample_id}")
        if row.get("golden_record_sha256") != current_sha:
            raise RuntimeError(f"Martian golden_record_sha256 mismatch for sample_id {sample_id}")
        try:
            row["judge"] = score_review(row.get("review", {}), record.get("comments") or [], judge)
        except Exception as exc:
            row["judge"] = {"tp": 0, "fp": len(row.get("review", {}).get("findings", [])), "fn": len(record.get("comments") or []), "precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": [], "errors": [{"error": str(exc), "schema_error": "judge execution failed"}], "judge_calls": 1, "judge_elapsed": 0.0, "elapsed": 0.0, "raw_response": None, "request_attempts": int(getattr(judge, "last_request_attempts", 0) or 0), "finish_reason": None, "schema_error": "judge execution failed"}
    temporary = results_path.with_suffix(".jsonl.tmp")
    temporary.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    temporary.replace(results_path)
    metrics = {"suite": "martian", "phase": "judge", **calculate_martian_metrics(rows), **_common_metrics(rows)}
    metrics["judge_elapsed_seconds"] = sum(float(row.get("judge", {}).get("judge_elapsed") or 0.0) for row in rows)
    _write_json(run_dir / "metrics.json", metrics)
    write_report(run_dir)
    if int(metrics.get("judge_errors") or 0):
        raise RuntimeError(f"Martian judge recorded {metrics['judge_errors']} error(s); scoring is incomplete")


def _model_entry(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for entry in config["models"]:
        if isinstance(entry, dict) and entry.get("profile") == model_id:
            return entry
        if entry == model_id:
            return {"profile": model_id}
    return {"profile": model_id}


def _lifecycle(config: dict[str, Any]) -> LMStudioLifecycle | None:
    settings = config.get("lifecycle") or {}
    if not settings.get("enabled"):
        return None
    return LMStudioLifecycle(str(settings["base_url"]), str(settings.get("api_key") or ""), int(settings.get("timeout", 600)))


def run_matrix(
    config_path: Path,
    *,
    resume: bool = False,
    report: bool = False,
    dry_run: bool = False,
    root: Path = ROOT,
    profile_loader: Callable[[str], ModelProfile] = get_model_profile,
    client_factory: Callable[[ModelProfile], Any] = ModelClient,
    task_runner: Callable[..., Path] = generate_review_run,
    judge_runner: Callable[..., None] = judge_martian_run,
    validation_fn: Callable[[str, str], dict[str, Any]] = validate_selection,
    report_runner: Callable[..., dict[str, Path]] = generate_experiment_report,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    report = report or bool(config.get("report", False))
    experiment_id = str(config["experiment_id"])
    tasks = build_task_matrix(config)
    validations = []
    for dataset in config["datasets"]:
        outcome = validation_fn(str(dataset["profile"]), str(dataset["suite"]))
        validations.append(outcome)
        if not outcome.get("valid"):
            raise RuntimeError(f"selection validation failed: {outcome}")
    plan = {"experiment_id": experiment_id, "context_policy": str(config.get("context_policy", "common-100k-char-v1")), "judge_profile": str(config.get("judge", {}).get("profile", "judge")), "lifecycle_enabled": bool((config.get("lifecycle") or {}).get("enabled")), "task_count": len(tasks), "tasks": [asdict(task) | {"task_id": task.id} for task in tasks], "validations": validations, "phases": ["preflight", "generate", "judge", "report"]}
    if dry_run:
        return {"dry_run": True, **plan, "failures": []}

    experiment_dir = root / "outputs" / "experiments" / experiment_id
    state_path = experiment_dir / "state.json"
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if state_path.exists():
        if not resume:
            raise RuntimeError(f"experiment {experiment_id} already exists; use --resume")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_sha:
            raise RuntimeError("experiment config changed; refusing unsafe resume")
    else:
        state = {"experiment_id": experiment_id, "config_sha256": config_sha, "status": "running", "created_at": time.time(), "tasks": {task.id: {**asdict(task), "status": "pending", "run_dir": None, "error": None, "judge_status": "pending" if task.suite == "martian" else "not_applicable"} for task in tasks}, "failures": []}
        _write_json(state_path, state)

    lifecycle = _lifecycle(config)
    failures: list[str] = []
    context_policy = str(config.get("context_policy", "common-100k-char-v1"))
    for model_id in [str(entry.get("profile") if isinstance(entry, dict) else entry) for entry in config["models"]]:
        model_tasks = [task for task in tasks if task.model_profile == model_id]
        entry = _model_entry(config, model_id)
        loaded_instance: str | None = None
        try:
            profile = profile_loader(model_id)
            if lifecycle:
                lifecycle_config = entry.get("lifecycle") or {}
                load_settings = {"context_length": profile.max_context_tokens, **(lifecycle_config.get("load") or {})}
                response = lifecycle.load_model(str(lifecycle_config.get("model") or profile.model_name), **load_settings)
                loaded_instance = str(response.get("instance_id") or lifecycle_config.get("model") or profile.model_name)
                lifecycle.verify_loaded(str(lifecycle_config.get("model") or profile.model_name), loaded_instance)
                profile = replace(profile, model_name=str(lifecycle_config.get("inference_model_name") or loaded_instance))
            probe_response, _ = client_factory(profile).chat([{"role": "user", "content": "Reply READY."}], max_tokens=8)
            if not isinstance(probe_response, dict):
                raise RuntimeError("model probe returned an invalid response")
        except Exception as exc:
            for task in model_tasks:
                state["tasks"][task.id].update({"status": "failed", "error": f"preflight: {exc}"})
                failures.append(task.id)
            _write_json(state_path, state)
            if lifecycle and loaded_instance:
                try: lifecycle.unload_model(loaded_instance)
                except Exception: pass
            continue
        for task in model_tasks:
            task_state = state["tasks"][task.id]
            task_state.update({"status": "running", "error": None})
            _write_json(state_path, state)
            try:
                run_dir = task_runner(experiment_dir, experiment_id, task, profile, context_policy=context_policy, root=root, resume=resume, client_factory=client_factory, model_metadata=entry.get("artifact") or {})
                task_state["run_dir"] = str(run_dir)
                metrics_path = run_dir / "metrics.json"
                if metrics_path.exists():
                    task_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    if int(task_metrics.get("sample_count") or 0) > 0 and int(task_metrics.get("run_errors") or 0) >= int(task_metrics.get("sample_count") or 0):
                        raise RuntimeError("all samples in the task failed")
                task_state.update({"status": "completed", "run_dir": str(run_dir), "error": None})
            except KeyboardInterrupt:
                _write_json(state_path, state)
                if lifecycle and loaded_instance:
                    try:
                        lifecycle.unload_model(loaded_instance)
                    except Exception:
                        pass
                raise
            except Exception as exc:
                task_state.update({"status": "failed", "error": str(exc)})
                failures.append(task.id)
            _write_json(state_path, state)
        if lifecycle and loaded_instance:
            try:
                lifecycle.unload_model(loaded_instance)
            except Exception as exc:
                failures.append(f"unload:{model_id}")
                state.setdefault("lifecycle_errors", []).append({"model": model_id, "error": str(exc)})
                _write_json(state_path, state)

    martian_tasks = [task for task in tasks if task.suite == "martian" and state["tasks"][task.id]["status"] == "completed"]
    judge_instance: str | None = None
    if martian_tasks:
        try:
            judge_profile = profile_loader(str(config.get("judge", {}).get("profile", "judge")))
            if lifecycle:
                judge_lifecycle = config.get("judge", {}).get("lifecycle") or {}
                load_settings = {"context_length": judge_profile.max_context_tokens, **(judge_lifecycle.get("load") or {})}
                response = lifecycle.load_model(str(judge_lifecycle.get("model") or judge_profile.model_name), **load_settings)
                judge_instance = str(response.get("instance_id") or judge_lifecycle.get("model") or judge_profile.model_name)
                lifecycle.verify_loaded(str(judge_lifecycle.get("model") or judge_profile.model_name), judge_instance)
                judge_profile = replace(judge_profile, model_name=str(judge_lifecycle.get("inference_model_name") or judge_instance))
            judge_probe, _ = client_factory(judge_profile).chat([{"role": "user", "content": "Reply READY."}], max_tokens=8)
            if not isinstance(judge_probe, dict):
                raise RuntimeError("judge probe returned an invalid response")
            _write_json(experiment_dir / "judge_manifest.json", {"experiment_id": experiment_id, "evaluation_version": "model-review-v2", "phase": "judge", "model_profile": str(config.get("judge", {}).get("profile", "judge")), "model_name": judge_profile.model_name, "max_context_tokens": judge_profile.max_context_tokens, "max_output_tokens": judge_profile.max_output_tokens, "prompt_version": "batch-one-to-one-v1", "prompt_sha256": hashlib.sha256(BATCH_MATCH_PROMPT.encode()).hexdigest(), "generation_settings": {key: getattr(judge_profile, key) for key in ("temperature", "top_p", "seed", "stream", "repeat_penalty", "presence_penalty", "frequency_penalty", "structured_output")}, "model_artifact": config.get("judge", {}).get("artifact") or {}})
            for task in martian_tasks:
                task_state = state["tasks"][task.id]
                try:
                    judge_runner(Path(task_state["run_dir"]), task, judge_profile, root=root, resume=resume, client_factory=client_factory)
                    task_state["judge_status"] = "completed"
                except Exception as exc:
                    task_state["judge_status"] = "failed"
                    task_state["judge_error"] = str(exc)
                    failures.append(f"judge:{task.id}")
                _write_json(state_path, state)
        except Exception as exc:
            failures.append("judge:preflight")
            state["judge_error"] = str(exc)
            for task in martian_tasks:
                state["tasks"][task.id]["judge_status"] = "failed"
                state["tasks"][task.id]["judge_error"] = f"preflight: {exc}"
            _write_json(state_path, state)
        finally:
            if lifecycle and judge_instance:
                try: lifecycle.unload_model(judge_instance)
                except Exception as exc: failures.append("unload:judge")

    state["failures"] = sorted(set(failures))
    state["status"] = "failed" if state["failures"] else "completed"
    state["finished_at"] = time.time()
    _write_json(state_path, state)
    report_paths = report_runner(experiment_dir) if report else {}
    return {"dry_run": False, **plan, "experiment_dir": str(experiment_dir), "status": state["status"], "failures": state["failures"], "reports": {key: str(value) for key, value in report_paths.items()}}
