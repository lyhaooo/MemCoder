"""Best-effort local isolation for generated Python code."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from memcoder.domain import ExecutionResult

_RUNNER = """\
import importlib.util
import pathlib
import sys

workspace = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("solution", workspace / "solution.py")
module = importlib.util.module_from_spec(spec)
sys.modules["solution"] = module
spec.loader.exec_module(module)
test_path = workspace / "test_solution.py"
exec(compile(test_path.read_text(encoding="utf-8"), str(test_path), "exec"), {})
"""


def _limit_resources() -> None:
    """Apply Unix resource limits in the child process when supported."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
    except (ImportError, OSError, ValueError):
        return


class LocalPythonSandbox:
    """Run a solution and assertions in a disposable local directory.

    This blocks inherited secrets and constrains time/resources. It does not
    provide kernel-level network or filesystem isolation. Use a container for
    arbitrary untrusted code.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_output_chars: int = 12_000,
        python_executable: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.python_executable = python_executable or sys.executable

    def execute(self, code: str, tests: str) -> ExecutionResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="memcoder-run-") as directory:
            workspace = Path(directory)
            (workspace / "solution.py").write_text(code, encoding="utf-8")
            (workspace / "test_solution.py").write_text(tests, encoding="utf-8")
            (workspace / "runner.py").write_text(_RUNNER, encoding="utf-8")

            env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONIOENCODING": "utf-8",
                "PYTHONHASHSEED": "0",
                "HOME": str(workspace),
                "TMPDIR": str(workspace),
            }
            try:
                process = subprocess.Popen(
                    [self.python_executable, "-I", "runner.py"],
                    cwd=workspace,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    preexec_fn=_limit_resources if os.name == "posix" else None,
                )
                stdout, stderr = process.communicate(timeout=self.timeout)
                duration = time.perf_counter() - started
                return ExecutionResult(
                    success=process.returncode == 0,
                    return_code=process.returncode,
                    stdout=self._truncate(stdout),
                    stderr=self._truncate(stderr),
                    duration_seconds=duration,
                )
            except subprocess.TimeoutExpired as exc:
                if os.name == "posix":
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                stdout, stderr = process.communicate()
                duration = time.perf_counter() - started
                return ExecutionResult(
                    success=False,
                    return_code=124,
                    stdout=self._truncate(stdout or self._decode_timeout_output(exc.stdout)),
                    stderr=self._truncate(
                        stderr
                        or self._decode_timeout_output(exc.stderr)
                        or "Execution timed out"
                    ),
                    duration_seconds=duration,
                    timed_out=True,
                )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        omitted = len(text) - self.max_output_chars
        return f"{text[: self.max_output_chars]}\n... <{omitted} chars omitted>"

    @staticmethod
    def _decode_timeout_output(value: str | bytes | None) -> str:
        if value is None:
            return ""
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
