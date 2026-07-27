from __future__ import annotations

import fnmatch
import os
import shlex
import subprocess
import uuid
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "list_files", "description": "List files below a relative directory", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "search_code", "description": "Search a text pattern in repository files", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Replace a repository text file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "run_command", "description": "Run a short test or inspection command in the repository", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]


class SafeWorkspace:
    def __init__(self, root: Path, command_timeout: int = 120):
        self.root = root.resolve()
        self.command_timeout = command_timeout

    def _path(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path escapes task workspace")
        return candidate

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_files":
            base = self._path(args.get("path", "."))
            return "\n".join(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.parts))[:12000]
        if name == "search_code":
            pattern, base = args["pattern"], self._path(args.get("path", "."))
            hits: list[str] = []
            for path in base.rglob("*"):
                if path.is_file() and path.stat().st_size < 1_000_000:
                    try:
                        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                            if pattern.lower() in line.lower():
                                hits.append(f"{path.relative_to(self.root)}:{number}:{line[:300]}")
                    except OSError:
                        pass
            return "\n".join(hits[:200])
        if name == "read_file":
            path = self._path(args["path"])
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            start, end = int(args.get("start_line", 1)), int(args.get("end_line", min(len(lines), 400)))
            return "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start - 1:end], start))[:30000]
        if name == "write_file":
            path = self._path(args["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["content"], encoding="utf-8")
            return f"wrote {path.relative_to(self.root)}"
        if name == "run_command":
            proc = subprocess.run(args["command"], cwd=self.root, shell=True, capture_output=True, text=True, timeout=self.command_timeout)
            output = (proc.stdout + "\n" + proc.stderr)[-12000:]
            return f"exit_code={proc.returncode}\n{output}"
        raise ValueError(f"unknown tool: {name}")


class DockerWorkspace:
    """A disposable tool workspace backed by an official Multi-SWE-bench image."""

    def __init__(self, image: str, repo: str, command_timeout: int = 120, hide_harness_files: bool = True):
        self.image = image
        self.repo = repo
        self.root = f"/home/{repo}"
        self.command_timeout = command_timeout
        self.hide_harness_files = hide_harness_files
        self.container_name = f"cbm-agent-{uuid.uuid4().hex[:12]}"
        self._started = False

    def __enter__(self) -> "DockerWorkspace":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _docker(self, args: list[str], *, input_text: str | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        command = ["docker", *args]
        try:
            return subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout or self.command_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"container command timed out after {timeout or self.command_timeout}s") from exc

    def _exec(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        input_text: str | None = None,
        timeout: int | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        command = ["exec"]
        if input_text is not None:
            command.append("-i")
        if cwd:
            command.extend(["-w", cwd])
        command.extend([self.container_name, *args])
        proc = self._docker(command, input_text=input_text, timeout=timeout)
        if check and proc.returncode != 0:
            message = (proc.stdout + "\n" + proc.stderr).strip()[-12000:]
            raise RuntimeError(f"container command failed with exit code {proc.returncode}: {message}")
        return proc

    def start(self) -> None:
        inspect = self._docker(["image", "inspect", self.image], timeout=30)
        if inspect.returncode != 0:
            raise RuntimeError(f"required image is not available locally: {self.image}")
        proc = self._docker(
            [
                "run", "--detach", "--rm", "--pull=never",
                "--name", self.container_name,
                "--entrypoint", "/bin/sh", self.image,
                "-c", "while :; do sleep 3600; done",
            ],
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"failed to start {self.image}: {(proc.stderr or proc.stdout).strip()}")
        self._started = True
        try:
            self._exec(["test", "-d", f"{self.root}/.git"])
            # Published PR images are already prepared. Some intentionally contain
            # dependency-lock changes made while the image was built; checkpoint
            # that state so it cannot leak into the model patch.
            self._exec(["git", "config", "user.email", "benchmark@localhost"], cwd=self.root)
            self._exec(["git", "config", "user.name", "Coder Review Benchmark"], cwd=self.root)
            self._exec(["git", "add", "-A"], cwd=self.root)
            self._exec(
                ["git", "commit", "--allow-empty", "--no-verify", "-m", "cbm prepared-image baseline"],
                cwd=self.root,
            )
            if self.hide_harness_files:
                self._exec([
                    "rm", "-f", "/home/fix.patch", "/home/test.patch",
                    "/home/fix-run.sh", "/home/test-run.sh",
                ])
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._started:
            self._docker(["rm", "-f", self.container_name], timeout=30)
            self._started = False

    @staticmethod
    def _relative_path(value: str) -> str:
        path = PurePosixPath(value or ".")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("path escapes task workspace")
        return str(path)

    @staticmethod
    def _output(proc: subprocess.CompletedProcess[str], limit: int = 30000) -> str:
        return (proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))[-limit:]

    def execute(self, name: str, args: dict[str, Any]) -> str:
        if name == "list_files":
            path = self._relative_path(str(args.get("path", ".")))
            proc = self._exec(
                [
                    "find", path,
                    "(", "-path", "*/.git", "-o", "-path", "*/node_modules", ")",
                    "-prune", "-o", "-type", "f", "-print",
                ],
                cwd=self.root,
            )
            return "\n".join(proc.stdout.splitlines()[:1000])[:12000]
        if name == "search_code":
            path = self._relative_path(str(args.get("path", ".")))
            pattern = str(args["pattern"])
            proc = self._exec(
                ["git", "grep", "-n", "-i", "-F", "--", pattern, "--", path],
                cwd=self.root,
                check=False,
            )
            if proc.returncode not in (0, 1):
                raise RuntimeError(self._output(proc, 12000))
            return "\n".join(proc.stdout.splitlines()[:200])[:30000]
        if name == "read_file":
            path = self._relative_path(str(args["path"]))
            start = max(1, int(args.get("start_line", 1)))
            end = max(start, int(args.get("end_line", start + 399)))
            script = 'NR >= start && NR <= end {printf "%d: %s\\n", NR, $0}'
            proc = self._exec(
                ["awk", "-v", f"start={start}", "-v", f"end={end}", script, path],
                cwd=self.root,
            )
            return proc.stdout[:30000]
        if name == "write_file":
            path = self._relative_path(str(args["path"]))
            parent = str(PurePosixPath(path).parent)
            self._exec(["mkdir", "-p", parent], cwd=self.root)
            self._exec(["tee", path], cwd=self.root, input_text=str(args["content"]))
            return f"wrote {path}"
        if name == "run_command":
            proc = self._exec(
                ["/bin/bash", "-lc", str(args["command"])],
                cwd=self.root,
                timeout=self.command_timeout,
                check=False,
            )
            return f"exit_code={proc.returncode}\n{self._output(proc, 12000)}"
        raise ValueError(f"unknown tool: {name}")

    def diff(self) -> str:
        # Make newly-created files visible to `git diff` without staging their
        # contents. This preserves legitimate fixes that add files.
        self._exec(["git", "add", "--intent-to-add", "--all"], cwd=self.root)
        proc = self._exec(
            ["git", "-c", "core.fileMode=false", "diff", "--binary"],
            cwd=self.root,
        )
        return proc.stdout


def evaluate_patch_in_image(image: str, repo: str, patch: str, timeout: int = 1800) -> dict[str, Any]:
    """Run an agent patch with the image's official test script in a fresh container."""
    started = __import__("time").perf_counter()
    if not patch.strip():
        return {
            "method": "official_image_tests_no_uploads",
            "resolved": False,
            "error": "model produced an empty patch",
            "wall_seconds": __import__("time").perf_counter() - started,
        }
    try:
        with DockerWorkspace(image, repo, command_timeout=timeout, hide_harness_files=False) as workspace:
            workspace._exec(["tee", "/home/fix.patch"], input_text=patch)
            # Coverage uploaders are not part of correctness and commonly fail
            # in offline/internal environments. Keep patch application and tests,
            # but remove a trailing codecov upload from a disposable script copy.
            workspace._exec(
                [
                    "/bin/sed", "-E", "-i.cbm",
                    r"s/[[:space:]]*&&[[:space:]]*codecov([[:space:]].*)?$//; /^[[:space:]]*codecov([[:space:]].*)?$/d",
                    "/home/fix-run.sh",
                ]
            )
            proc = workspace._exec(
                ["/bin/bash", "/home/fix-run.sh"],
                cwd="/home",
                timeout=timeout,
                check=False,
            )
            output = workspace._output(proc, 50000)
            return {
                "method": "official_image_tests_no_uploads",
                "resolved": proc.returncode == 0,
                "exit_code": proc.returncode,
                "output": output,
                "wall_seconds": __import__("time").perf_counter() - started,
            }
    except Exception as exc:
        return {
            "method": "official_image_tests_no_uploads",
            "resolved": False,
            "error": str(exc),
            "wall_seconds": __import__("time").perf_counter() - started,
        }
