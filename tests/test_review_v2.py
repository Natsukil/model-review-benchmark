from coder_review_benchmark.adapters import MartianReviewAdapter, SWEReviewAdapter
from coder_review_benchmark.cli import _decision_correct
from coder_review_benchmark.scoring import calculate_swe_metrics, parse_review


def _review(decision):
    return '{"decision":%s,"summary":"ok","findings":[]}' % ("null" if decision is None else '"' + decision + '"')


def test_swe_decision_parser_is_strict_and_normalizes_only_reject_alias():
    assert parse_review(_review("approve"))["decision"] == "approve"
    assert parse_review(_review("request_changes"))["decision"] == "request_changes"
    assert parse_review(_review("reject"))["decision"] == "request_changes"
    for value in (None, "unknown", "approved", 3):
        parsed = parse_review(_review(value) if not isinstance(value, int) else '{"decision":3,"summary":"ok","findings":[]}')
        assert parsed["decision"] is None
        assert not parsed["schema_valid"]


def test_invalid_decision_is_not_confusion_or_reject():
    rows = [
        {"expected_resolved": False, "review": parse_review(_review("approve"))},
        {"expected_resolved": True, "review": parse_review(_review("request_changes"))},
        {"expected_resolved": False, "review": parse_review("not json")},
    ]
    metrics = calculate_swe_metrics(rows)
    assert metrics["sample_count"] == 3
    assert metrics["decision_accuracy_all"] == 0
    assert metrics["confusion_matrix"] == {
        "approve_resolved": 0, "request_changes_resolved": 1,
        "approve_unresolved": 1, "request_changes_unresolved": 0,
    }
    assert metrics["invalid_decision_rate"] > 0
    assert _decision_correct(None, False) is False
    assert _decision_correct("unknown", False) is False
    assert _decision_correct("reject", False) is True


def test_review_adapters_do_not_leak_gold_fields():
    task = {
        "problem_statement": "Issue text",
        "model_patch": "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-old\n+new",
        "resolved": True,
        "gold_patch": "SECRET_GOLD",
        "test_results": "SECRET_TEST",
        "golden_comments": "SECRET_COMMENT",
    }
    prepared = SWEReviewAdapter().prepare(task)
    prompt = prepared.messages[-1]["content"]
    assert "Issue text" in prompt and "SECRET_GOLD" not in prompt and "SECRET_TEST" not in prompt

    martian = MartianReviewAdapter().prepare({"pr_title": "Title", "pr_body": "Body", "patch": "diff --git a/a b/b\n", "comments": [{"comment": "SECRET"}]})
    assert "Title" in martian.messages[-1]["content"] and "SECRET" not in martian.messages[-1]["content"]
