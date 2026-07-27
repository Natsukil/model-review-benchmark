import json

from coder_review_benchmark.aggregate import collect_runs, select_comparison_runs, summarize_runs


def _run(root, name, sample_count, completion_rate, accuracy):
    run = root / name
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps({
        "suite": "swe_review",
        "completion_rate": completion_rate,
        "decision_accuracy": accuracy,
    }))
    (run / "run_manifest.json").write_text(json.dumps({
        "suite": "swe_review",
        "model_profile": "model-a",
        "model_name": "actual-model-a",
        "profile": "review-hour",
    }))
    rows = [
        {"status": "completed" if index < round(sample_count * completion_rate) else "error"}
        for index in range(sample_count)
    ]
    (run / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_summary_prefers_larger_run_over_newer_smoke(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _run(outputs, "20260726-100000_model-a_swe_review", 500, 1.0, 0.5)
    _run(outputs, "20260726-110000_model-a_swe_review", 3, 1.0, 1.0)

    selected = select_comparison_runs(collect_runs(outputs))
    assert len(selected) == 1
    assert selected[0]["sample_count"] == 500
    assert selected[0]["decision_accuracy"] == 0.5


def test_summarize_writes_excel_markdown_and_csv(tmp_path):
    outputs = tmp_path / "outputs"
    reports = tmp_path / "reports"
    outputs.mkdir()
    _run(outputs, "20260726-100000_model-a_swe_review", 2, 1.0, 0.5)

    paths = summarize_runs(outputs, reports)
    assert all(path.exists() for path in paths.values())
    assert "model-a" in paths["markdown"].read_text(encoding="utf-8")
