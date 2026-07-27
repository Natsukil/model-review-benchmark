import json
from pathlib import Path

import pytest

from coder_review_benchmark.config import ModelProfile
from coder_review_benchmark.experiment import ExperimentTask, build_task_matrix, load_experiment_config, run_matrix


MODELS = ["qwen2.5-coder-7b", "qwen3-coder-30b", "qwen3-coder-next-80b"]


def _config(path: Path, experiment_id: str = "mock-exp") -> Path:
    path.write_text(
        "experiment_id: " + experiment_id + "\n"
        "context_policy: common-100k-char-v1\n"
        "models:\n" + "".join(f"  - profile: {model}\n" for model in MODELS) +
        "datasets:\n"
        "  - suite: swe_review\n    profile: swe-review-balanced-500-v1\n"
        "  - suite: martian\n    profile: martian-offline-50-v1\n"
        "judge:\n  profile: judge\n"
        "lifecycle:\n  enabled: false\n",
        encoding="utf-8",
    )
    return path


def _profile(model_id: str) -> ModelProfile:
    return ModelProfile(model_id, model_id, "http://invalid/v1", "dummy", "native_tool_calls", 4096, 32768, 1)


class FakeClient:
    def __init__(self, profile): self.profile = profile
    def chat(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "READY"}}]}, 0.01


def _fake_task_runner(events, fail_task=None, interrupt_task=None):
    def run(experiment_dir, experiment_id, task, profile, **kwargs):
        events.append(("generate", task.id))
        if task.id == interrupt_task:
            raise KeyboardInterrupt()
        if task.id == fail_task:
            raise RuntimeError("synthetic failure")
        run_dir = experiment_dir / "runs" / task.id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_manifest.json").write_text(json.dumps({"experiment_id": experiment_id, "model_profile": task.model_profile, "suite": task.suite, "dataset_profile": task.dataset_profile, "evaluation_version": "model-review-v2"}), encoding="utf-8")
        (run_dir / "results.jsonl").write_text(json.dumps({"index": 0, "status": "completed", "suite": task.suite, "model": task.model_profile, "messages_sha256": "same", "review": {"findings": []}}) + "\n", encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps({"suite": task.suite, "sample_count": 1}), encoding="utf-8")
        return run_dir
    return run


def _fake_judge(events):
    def judge(run_dir, task, profile, **kwargs):
        events.append(("judge", task.id))
        metrics = {"suite": "martian", "sample_count": 1, "tp": 0, "fp": 0, "fn": 0, "micro_precision": 0.0, "micro_recall": 0.0, "micro_f1": 0.0}
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return judge


def _valid(profile, suite):
    return {"valid": True, "profile": profile, "suite": suite, "count": 1, "errors": []}


def test_matrix_is_model_major_three_by_two(tmp_path):
    config = load_experiment_config(_config(tmp_path / "matrix.yaml"))
    tasks = build_task_matrix(config)
    assert len(tasks) == 6
    assert [(task.model_profile, task.suite) for task in tasks] == [
        (MODELS[0], "swe_review"), (MODELS[0], "martian"),
        (MODELS[1], "swe_review"), (MODELS[1], "martian"),
        (MODELS[2], "swe_review"), (MODELS[2], "martian"),
    ]


def test_generate_all_models_before_delayed_judge_and_report_by_suite(tmp_path):
    events = []
    result = run_matrix(_config(tmp_path / "matrix.yaml"), root=tmp_path, profile_loader=_profile, client_factory=FakeClient, task_runner=_fake_task_runner(events), judge_runner=_fake_judge(events), validation_fn=_valid, report=True)
    assert result["status"] == "completed"
    assert [kind for kind, _ in events] == ["generate"] * 6 + ["judge"] * 3
    report_dir = Path(result["reports"]["directory"])
    for suite in ("swe_review", "martian"):
        for filename in ("report.md", "report.json", "metrics.csv", "per_sample.csv", "report.xlsx"):
            assert (report_dir / suite / filename).exists()
    payload = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "mock-exp"
    assert set(payload["suites"]) == {"swe_review", "martian"}


def test_resume_restarts_interrupted_task_without_replacing_from_history(tmp_path):
    config_path = _config(tmp_path / "matrix.yaml", "resume-exp")
    first_events = []
    interrupted = f"{MODELS[0]}__swe_review__swe-review-balanced-500-v1"
    with pytest.raises(KeyboardInterrupt):
        run_matrix(config_path, root=tmp_path, profile_loader=_profile, client_factory=FakeClient, task_runner=_fake_task_runner(first_events, interrupt_task=interrupted), judge_runner=_fake_judge(first_events), validation_fn=_valid)
    state_path = tmp_path / "outputs" / "experiments" / "resume-exp" / "state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["tasks"][interrupted]["status"] == "running"
    historical = tmp_path / "outputs" / "historical-perfect-run"
    historical.mkdir(parents=True)
    (historical / "metrics.json").write_text('{"sample_count":999}', encoding="utf-8")
    second_events = []
    result = run_matrix(config_path, root=tmp_path, resume=True, report=True, profile_loader=_profile, client_factory=FakeClient, task_runner=_fake_task_runner(second_events), judge_runner=_fake_judge(second_events), validation_fn=_valid)
    assert result["status"] == "completed"
    report_text = Path(result["reports"]["json"]).read_text(encoding="utf-8")
    assert "historical-perfect-run" not in report_text
    assert interrupted in [task_id for kind, task_id in second_events if kind == "generate"]


def test_failed_task_is_reported_and_does_not_stop_remaining_matrix(tmp_path):
    events = []
    failed = f"{MODELS[1]}__martian__martian-offline-50-v1"
    result = run_matrix(_config(tmp_path / "matrix.yaml", "failed-exp"), root=tmp_path, report=True, profile_loader=_profile, client_factory=FakeClient, task_runner=_fake_task_runner(events, fail_task=failed), judge_runner=_fake_judge(events), validation_fn=_valid)
    assert result["status"] == "failed"
    assert failed in result["failures"]
    assert len([event for event in events if event[0] == "generate"]) == 6
    report = json.loads(Path(result["reports"]["json"]).read_text(encoding="utf-8"))
    failed_rows = [row for row in report["runs"] if row["task_id"] == failed]
    assert failed_rows[0]["status"] == "failed"
