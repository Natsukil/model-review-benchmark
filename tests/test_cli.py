import io
import urllib.error

from coder_review_benchmark import cli


def test_fetch_pr_diff_retries_and_caches(monkeypatch, tmp_path):
    attempts = 0

    def fake_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary TLS failure")
        return io.BytesIO(b"diff --git a/a.py b/a.py\n")

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(cli.time, "sleep", lambda _: None)
    monkeypatch.setenv("CBM_DIFF_MAX_RETRIES", "3")

    url = "https://github.com/example/project/pull/1"
    assert cli._fetch_pr_diff(url).startswith("diff --git")
    assert attempts == 3

    # A cached diff must not issue another network request.
    assert cli._fetch_pr_diff(url).startswith("diff --git")
    assert attempts == 3
