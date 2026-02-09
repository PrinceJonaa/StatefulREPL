"""
LoomSandbox: Restricted execution environment for AI-generated code.

Exposes only whitelisted LoomREPL operations — no file system access,
no network, no dangerous builtins.
"""

from typing import Any, Dict

from stateful_repl.loom_state import LoomREPL


class LoomSandbox:
    """
    Execute AI-generated Python code in a restricted namespace.

    Only approved LoomREPL methods are exposed.  The AI sees a `loom`
    dict of callable functions — nothing else.

    Security constraints:
      - No os, subprocess, sys, importlib
      - No open(), exec() (nested), eval() (nested)
      - Whitelisted builtins only
      - 30 s timeout enforced at the caller level (not here)
    """

    SAFE_BUILTINS = {
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "sorted": sorted,
        "enumerate": enumerate,
        "range": range,
        "zip": zip,
        "map": map,
        "filter": filter,
        "isinstance": isinstance,
        "type": type,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "True": True,
        "False": False,
        "None": None,
    }

    def __init__(self, loom: LoomREPL):
        self.loom = loom
        self.allowed_functions: Dict[str, Any] = {
            "read_state": loom.read_state,
            "print_state": loom.print_state,
            "append": loom.append,
            "update_l1": loom.update_l1,
            "clear_l1": loom.clear_l1,
            "consolidate_l1_to_l2": loom.consolidate_l1_to_l2,
            "consolidate_l2_to_l3": loom.consolidate_l2_to_l3,
            "validate_state": loom.validate_state,
            "get_event_log": loom.get_event_log,
        }

    def execute(self, code: str) -> Dict[str, Any]:
        """
        Run *code* in a namespace that only contains safe builtins
        and the approved loom operations.

        Returns {"status": "success"} or {"status": "error", "message": ...}.
        """
        safe_globals: Dict[str, Any] = {
            "__builtins__": dict(self.SAFE_BUILTINS),
            "loom": dict(self.allowed_functions),
        }

        try:
            exec(code, safe_globals)  # noqa: S102 — intentionally sandboxed
            return {"status": "success"}
        except Exception as exc:
            return {"status": "error", "message": f"{type(exc).__name__}: {exc}"}
