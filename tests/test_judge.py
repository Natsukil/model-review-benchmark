from coder_review_benchmark.judge import BATCH_MATCH_RESPONSE_SCHEMA, score_review


class DummyJudge:
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0, response_format=None):
        assert response_format == BATCH_MATCH_RESPONSE_SCHEMA
        self.calls += 1
        return {
            "choices": [{"message": {"content": '{"matches":[{"golden_index":0,"candidate_index":0,"confidence":0.95,"reasoning":"same issue"}]}'}}]
        }, 0.01


def test_score_review_matches_findings():
    review = {
        "parseable": True,
        "findings": [
            {
                "path": "src/a.py",
                "line": 10,
                "severity": "high",
                "category": "bug",
                "description": "The value can be None before it is dereferenced.",
            }
        ],
    }
    judge = DummyJudge()
    result = score_review(
        review,
        [{"comment": "Dereferencing a nullable value can crash here.", "severity": "high"}],
        judge,
    )
    assert result["tp"] == 1
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["judge_calls"] == 1
    assert judge.calls == 1


def test_score_review_rejects_duplicate_batch_matches():
    class DuplicateJudge:
        def chat(self, messages, tools=None, temperature=0, response_format=None):
            return {
                "choices": [{"message": {"content": '{"matches":[{"golden_index":0,"candidate_index":0},{"golden_index":1,"candidate_index":0}]}'}}]
            }, 0.01

    review = {
        "parseable": True,
        "findings": [{"path": "a.py", "line": 1, "description": "one issue"}],
    }
    result = score_review(
        review,
        [{"comment": "first issue"}, {"comment": "second issue"}],
        DuplicateJudge(),
    )
    assert result["tp"] == 1
    assert result["fn"] == 1
    assert result["errors"][0]["error"] == "judge returned a non-unique match"
