"""
Multi-agent orchestration — Phase 3 stub.

Provides the Saga transaction pattern and role definitions for
Coordinator / Builder / Verifier / Distiller Looms.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple


class SagaStep:
    """One compensable step in a Saga transaction."""

    def __init__(
        self,
        name: str,
        action: Callable[[], Any],
        compensation: Callable[[Any], None],
    ):
        self.name = name
        self.action = action
        self.compensation = compensation


class SagaTransaction:
    """
    Ensures consistency across distributed agent operations.

    If any step fails, previously completed steps are rolled back
    in reverse order via their compensation functions.
    """

    def __init__(self) -> None:
        self.steps: List[SagaStep] = []

    def add_step(
        self,
        name: str,
        action: Callable[[], Any],
        compensation: Callable[[Any], None],
    ) -> None:
        self.steps.append(SagaStep(name, action, compensation))

    def execute(self) -> List[Tuple[str, Any]]:
        """
        Run all steps in order.  On failure, compensate in reverse.
        Returns list of (step_name, result) tuples.
        """
        completed: List[Tuple[str, Any]] = []
        try:
            for step in self.steps:
                result = step.action()
                completed.append((step.name, result))
        except Exception:
            # Compensate in reverse
            for step_name, result in reversed(completed):
                matching = [s for s in self.steps if s.name == step_name]
                if matching:
                    matching[0].compensation(result)
            raise
        return completed
