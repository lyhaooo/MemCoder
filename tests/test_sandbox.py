from memcoder.execution.sandbox import LocalPythonSandbox


def test_sandbox_executes_solution_and_assertions() -> None:
    result = LocalPythonSandbox(timeout=2).execute(
        "def add(a, b):\n    return a + b\n",
        "from solution import add\nassert add(2, 3) == 5\n",
    )
    assert result.success
    assert result.return_code == 0


def test_sandbox_captures_failure() -> None:
    result = LocalPythonSandbox(timeout=2).execute(
        "def add(a, b):\n    return a - b\n",
        "from solution import add\nassert add(2, 3) == 5\n",
    )
    assert not result.success
    assert "AssertionError" in result.stderr


def test_sandbox_times_out() -> None:
    result = LocalPythonSandbox(timeout=0.2).execute(
        "while True:\n    pass\n",
        "pass\n",
    )
    assert not result.success
    assert result.timed_out
    assert result.return_code == 124

