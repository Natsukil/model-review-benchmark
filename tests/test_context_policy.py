from coder_review_benchmark.context_policy import MAX_INPUT_CHARS, apply_context


def test_common_100k_char_is_deterministic_and_token_neutral():
    diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n" + "x" * 150_000
    text = "Issue\n\nDIFF:\n" + diff
    first = apply_context(text, "common-100k-char-v1", diff=diff)
    second = apply_context(text, "common-100k-char-v1", diff=diff)
    assert first.text == second.text
    assert first.final_chars <= MAX_INPUT_CHARS == 100_000
    assert first.original_tokens is None and first.final_tokens is None
    assert first.reason == "common-100k-char-budget"
    assert "diff --git a/a.py b/a.py" in first.text


def test_common_100k_char_no_longer_raises_value_error():
    result = apply_context("short", "common-100k-char-v1")
    assert result.text == "short"
    assert result.reason is None
