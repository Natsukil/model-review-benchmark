import json
from pathlib import Path

import pytest

import coder_review_benchmark.experiment as experiment_module
from coder_review_benchmark.config import ModelProfile
from coder_review_benchmark.experiment import ExperimentTask, _golden_identity, generate_review_run, judge_martian_run, run_matrix


MODELS = ["qwen2.5-coder-7b", "qwen3-coder-30b", "qwen3-coder-next-80b"]


def _profile(model_id: str) -> ModelProfile:
    return ModelProfile(model_id, model_id, "http://invalid/v1", "", "native_tool_calls", 4096, 32768, 1, send_auth=False)


class FakeModelClient:
    def __init__(self, profile):
        self.profile = profile
        self.last_request_attempts = 1

    def chat(self, messages, **kwargs):
        if kwargs.get("max_tokens") == 8:
            content = "READY"
        else:
            assert kwargs["max_tokens"] == 4096
            assert kwargs["response_format"]["json_schema"]["strict"] is True
            content = '{"decision":"approve","summary":"looks correct","findings":[]}'
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}, "_request_attempts": 1}, 0.01


def _write_swe_selection(root: Path, profile: str = "tiny-swe") -> Path:
    directory = root / "data" / "selections" / profile
    directory.mkdir(parents=True, exist_ok=True)
    selection = directory / "swe_review.jsonl"
    selection.write_text(json.dumps({"record": {"instance_id": "one", "problem_statement": "Fix bug", "model_patch": "diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b", "resolved": True, "generator_model": "g", "difficulty": "easy"}}) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"profile": profile, "sha256": "frozen"}), encoding="utf-8")
    return selection


def test_full_mock_pipeline_and_three_model_message_identity(tmp_path):
    _write_swe_selection(tmp_path)
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "experiment_id: integration-v2\ncontext_policy: common-100k-char-v1\nmodels:\n"
        + "".join(f"  - profile: {model}\n" for model in MODELS)
        + "datasets:\n  - suite: swe_review\n    profile: tiny-swe\njudge:\n  profile: judge\nlifecycle:\n  enabled: false\n",
        encoding="utf-8",
    )
    result = run_matrix(config, root=tmp_path, report=True, profile_loader=_profile, client_factory=FakeModelClient, validation_fn=lambda profile, suite: {"valid": True, "count": 1})
    assert result["status"] == "completed"
    hashes = []
    for model in MODELS:
        run = tmp_path / "outputs" / "experiments" / "integration-v2" / "runs" / f"{model}__swe_review__tiny-swe" / "results.jsonl"
        row = json.loads(run.read_text(encoding="utf-8"))
        hashes.append(row["messages_sha256"])
        assert row["decision_correct"] is True
        assert row["request_attempts"] == 1
    assert len(set(hashes)) == 1
    report = json.loads(Path(result["reports"]["json"]).read_text(encoding="utf-8"))
    assert report["status"] == "completed"


def test_resume_rejects_prompt_or_selection_change_before_manifest_overwrite(tmp_path, monkeypatch):
    selection = _write_swe_selection(tmp_path)
    task = ExperimentTask("model", "swe_review", "tiny-swe", 1)
    profile = _profile("model")
    experiment_dir = tmp_path / "outputs" / "experiments" / "resume"
    run_dir = generate_review_run(experiment_dir, "resume", task, profile, context_policy="common-100k-char-v1", root=tmp_path, client_factory=FakeModelClient)
    manifest_before = (run_dir / "run_manifest.json").read_bytes()
    fingerprint = json.loads(manifest_before)["resume_fingerprint"]
    assert {"evaluation_version", "model_profile", "model_name", "prompt_sha256", "selection_jsonl_sha256", "dataset_manifest_sha256", "context_policy", "generation_settings", "model_artifact_sha256", "limit", "sha256"} <= set(fingerprint)
    monkeypatch.setattr(experiment_module, "prompt_sha256", lambda adapter: "0" * 64)
    with pytest.raises(RuntimeError, match="resume fingerprint mismatch"):
        generate_review_run(experiment_dir, "resume", task, profile, context_policy="common-100k-char-v1", root=tmp_path, resume=True, client_factory=FakeModelClient)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before
    monkeypatch.undo()
    selection.write_text(selection.read_text(encoding="utf-8") + json.dumps({"record": {"instance_id": "two"}}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="resume fingerprint mismatch"):
        generate_review_run(experiment_dir, "resume", task, profile, context_policy="common-100k-char-v1", root=tmp_path, resume=True, client_factory=FakeModelClient)
    assert (run_dir / "run_manifest.json").read_bytes() == manifest_before


def test_martian_judge_matches_reordered_gold_by_stable_id(tmp_path):
    directory = tmp_path / "data" / "selections" / "martian-tiny"
    directory.mkdir(parents=True)
    records = [
        {"url": "https://github.com/o/r/pull/1", "comments": [{"comment": "one"}]},
        {"url": "https://github.com/o/r/pull/2", "comments": [{"comment": "two"}]},
    ]
    (directory / "martian.jsonl").write_text("\n".join(json.dumps({"record": record}) for record in reversed(records)) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = []
    for index, record in enumerate(records):
        sample_id, golden_sha = _golden_identity(record)
        rows.append({"index": index, "sample_id": sample_id, "pr_url": record["url"], "golden_record_sha256": golden_sha, "golden_finding_count": 1, "language": "Python", "status": "completed", "review": {"parseable": False, "findings": []}})
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    judge_martian_run(run_dir, ExperimentTask("m", "martian", "martian-tiny"), _profile("judge"), root=tmp_path, client_factory=FakeModelClient)
    judged = [json.loads(line) for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["judge"]["fn"] for row in judged] == [1, 1]


def test_martian_judge_errors_fail_the_judge_phase(tmp_path):
    directory = tmp_path / "data" / "selections" / "martian-error"
    directory.mkdir(parents=True)
    record = {"url": "https://github.com/o/r/pull/3", "comments": [{"comment": "gold"}]}
    (directory / "martian.jsonl").write_text(json.dumps({"record": record}) + "\n", encoding="utf-8")
    sample_id, golden_sha = _golden_identity(record)
    row = {"index": 0, "sample_id": sample_id, "pr_url": record["url"], "golden_record_sha256": golden_sha, "golden_finding_count": 1, "status": "completed", "review": {"parseable": True, "findings": [{"path": "a", "line": 1, "severity": "high", "category": "correctness", "description": "candidate"}]}}
    run_dir = tmp_path / "error-run"
    run_dir.mkdir()
    (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    class InvalidJudge(FakeModelClient):
        def chat(self, messages, **kwargs):
            return {"choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}], "_request_attempts": 1}, 0.01

    with pytest.raises(RuntimeError, match="scoring is incomplete"):
        judge_martian_run(run_dir, ExperimentTask("m", "martian", "martian-error"), _profile("judge"), root=tmp_path, client_factory=InvalidJudge)
