from pathlib import Path

import pytest

from coder_review_benchmark.tools import DockerWorkspace, SafeWorkspace, evaluate_patch_in_image


def test_workspace_blocks_escape(tmp_path: Path):
    ws = SafeWorkspace(tmp_path)
    with pytest.raises(ValueError):
        ws.execute("read_file", {"path": "../secret"})


def test_docker_workspace_blocks_escape():
    with pytest.raises(ValueError):
        DockerWorkspace._relative_path("../../home/fix.patch")


def test_empty_patch_is_unresolved_without_starting_docker():
    result = evaluate_patch_in_image("missing/image:test", "repo", "")
    assert result["resolved"] is False
    assert result["error"] == "model produced an empty patch"
