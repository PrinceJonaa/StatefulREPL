"""Tests for the LoomSandbox restricted execution environment."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stateful_repl.loom_state import LoomREPL
from stateful_repl.sandbox import LoomSandbox


@pytest.fixture
def sandbox(tmp_path):
    loom = LoomREPL(str(tmp_path / "sandbox_state.md"))
    return LoomSandbox(loom)


class TestSandboxExecution:
    def test_basic_loom_operations(self, sandbox):
        code = """
loom['update_l1']('goal', 'sandbox test')
state = loom['read_state']('L1')
print(state['goal'])
"""
        result = sandbox.execute(code)
        assert result["status"] == "success"

    def test_append_l2(self, sandbox):
        code = "loom['append']('L2', 'sandbox entry')"
        result = sandbox.execute(code)
        assert result["status"] == "success"

    def test_validate_state(self, sandbox):
        code = "result = loom['validate_state']()"
        result = sandbox.execute(code)
        assert result["status"] == "success"

    def test_syntax_error_caught(self, sandbox):
        result = sandbox.execute("this is not python")
        assert result["status"] == "error"

    def test_runtime_error_caught(self, sandbox):
        result = sandbox.execute("x = 1 / 0")
        assert result["status"] == "error"
        assert "ZeroDivisionError" in result["message"]


class TestSandboxSecurity:
    def test_no_os_access(self, sandbox):
        result = sandbox.execute("import os")
        assert result["status"] == "error"

    def test_no_subprocess(self, sandbox):
        result = sandbox.execute("import subprocess")
        assert result["status"] == "error"

    def test_no_open(self, sandbox):
        result = sandbox.execute("f = open('/etc/passwd')")
        assert result["status"] == "error"

    def test_no_eval(self, sandbox):
        result = sandbox.execute("eval('1+1')")
        assert result["status"] == "error"

    def test_safe_builtins_available(self, sandbox):
        code = """
x = len([1, 2, 3])
y = str(42)
z = sorted([3, 1, 2])
print(x, y, z)
"""
        result = sandbox.execute(code)
        assert result["status"] == "success"
