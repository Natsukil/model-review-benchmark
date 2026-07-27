from coder_review_benchmark.scoring import calculate_swe_metrics, parse_review


def test_invalid_decision_is_never_counted_as_correct_reject():
    invalid = parse_review('{"decision":"invalid","summary":"bad","findings":[]}')
    rows = [{"expected_resolved": False, "review": invalid}]
    metrics = calculate_swe_metrics(rows)
    assert invalid["decision"] is None
    assert metrics["decision_accuracy_all"] == 0.0
    assert metrics["confusion_matrix"]["request_changes_unresolved"] == 0
    assert metrics["invalid_decision_rate"] == 1.0
