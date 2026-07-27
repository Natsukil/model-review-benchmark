import json

from coder_review_benchmark.adapters import MartianReviewAdapter, SWEReviewAdapter


def test_three_models_receive_byte_identical_messages_and_hashes():
    task = {"problem_statement": "Fix it", "model_patch": "diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b"}
    prepared = [SWEReviewAdapter().prepare(task, "common-100k-char-v1") for _ in range(3)]
    serialized = [json.dumps(item.messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() for item in prepared]
    assert serialized[0] == serialized[1] == serialized[2]
    assert len({item.messages_sha256 for item in prepared}) == 1
    assert all(len(item.benchmark_serialization_sha256) == 64 for item in prepared)


def test_adapters_do_not_expose_gold_or_test_results():
    swe = SWEReviewAdapter().prepare({"problem_statement": "Issue", "model_patch": "diff --git a/a b/a\n", "resolved": True, "gold_comment": "SECRET", "test_results": "SECRET_TEST"})
    martian = MartianReviewAdapter().prepare({"pr_title": "PR", "patch": "diff --git a/a b/a\n", "comments": [{"comment": "SECRET"}]})
    assert "SECRET" not in swe.messages[-1]["content"]
    assert "SECRET_TEST" not in swe.messages[-1]["content"]
    assert "SECRET" not in martian.messages[-1]["content"]
