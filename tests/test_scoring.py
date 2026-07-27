from coder_review_benchmark.scoring import calculate_martian_metrics, calculate_swe_metrics, parse_review


def test_invalid_decision_is_never_counted_as_correct_reject():
    invalid = parse_review('{"decision":"invalid","summary":"bad","findings":[]}')
    rows = [{"expected_resolved": False, "review": invalid}]
    metrics = calculate_swe_metrics(rows)
    assert invalid["decision"] is None
    assert metrics["decision_accuracy_all"] == 0.0
    assert metrics["confusion_matrix"]["request_changes_unresolved"] == 0
    assert metrics["invalid_decision_rate"] == 1.0


def test_martian_metrics_ignore_failed_judges_and_mark_partial():
    rows = [
        {"review": {"findings": []}, "judge": {"status": "completed", "tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 2 / 3, "errors": [], "judge_calls": 1}},
        {"review": {"findings": []}, "judge": {"status": "failed", "errors": [{"error": "HTTP 400"}], "judge_calls": 1}},
    ]
    metrics = calculate_martian_metrics(rows)
    assert metrics["judge_status"] == "partial"
    assert metrics["judge_successful_samples"] == 1
    assert metrics["judge_failed_samples"] == 1
    assert metrics["tp"] == 1 and metrics["fn"] == 1
    assert metrics["micro_f1"] == 2 / 3
