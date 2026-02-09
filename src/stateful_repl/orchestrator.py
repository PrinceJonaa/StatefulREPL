"""
Multi-Agent Orchestration — Phase 3.

Full Saga transaction manager with:
  - Async execution with timeout & retry policies
  - Distributed state tracking per saga
  - Compensation (rollback) on failure
  - Event emission at every lifecycle point
  - Integration with the message bus

Also provides the backward-compatible sync SagaStep/SagaTransaction
from Phase 1.

Usage:
    bus = InProcessBus()
    saga = AsyncSagaManager(bus=bus)

    saga_id = await saga.start("deploy-pipeline", steps=[
        SagaStepDef(name="build", action=build_fn, compensation=cleanup_fn),
        SagaStepDef(name="test", action=test_fn, compensation=revert_fn),
        SagaStepDef(name="deploy", action=deploy_fn, compensation=rollback_fn),
    ])

    status = await saga.wait(saga_id, timeout=60.0)
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from stateful_repl.message_bus import Message, MessageBus, Topics


# ─────────────────────────────────────────────────────────
# Phase 1 compat — sync Saga (unchanged API)
# ─────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────
# Phase 3 — Async Saga
# ─────────────────────────────────────────────────────────

class SagaStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COMPENSATING = "compensating"
    COMPENSATED = "compensated"
    SKIPPED = "skipped"


ActionFn = Union[Callable[[], Any], Callable[[], Awaitable[Any]]]
CompensationFn = Union[Callable[[Any], None], Callable[[Any], Awaitable[None]]]


@dataclass
class RetryPolicy:
    """Retry configuration for a saga step."""
    max_retries: int = 0
    delay: float = 1.0           # seconds between retries
    backoff_factor: float = 2.0  # exponential backoff multiplier
    retry_on: Optional[Tuple[type, ...]] = None  # exception types to retry on (None = all)


@dataclass
class SagaStepDef:
    """Definition for an async saga step."""
    name: str
    action: ActionFn
    compensation: CompensationFn
    timeout: Optional[float] = None  # per-step timeout in seconds
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    depends_on: List[str] = field(default_factory=list)  # step dependencies


@dataclass
class StepRecord:
    """Runtime record of a step's execution."""
    name: str
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0


@dataclass
class SagaRecord:
    """Runtime record of a saga's execution."""
    saga_id: str
    name: str
    status: SagaStatus = SagaStatus.PENDING
    steps: Dict[str, StepRecord] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "saga_id": self.saga_id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "metadata": self.metadata,
            "steps": {
                name: {
                    "status": sr.status.value,
                    "attempts": sr.attempts,
                    "error": sr.error,
                    "started_at": sr.started_at,
                    "completed_at": sr.completed_at,
                }
                for name, sr in self.steps.items()
            },
        }


class AsyncSagaManager:
    """
    Full async saga orchestrator with retry, timeout, compensation,
    and event bus integration.

    Each saga is a sequence of steps that either all complete
    or are rolled back (compensated) in reverse order.
    """

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        default_timeout: float = 300.0,  # 5 minutes default
    ):
        self._bus = bus
        self._default_timeout = default_timeout
        self._sagas: Dict[str, SagaRecord] = {}
        self._step_defs: Dict[str, List[SagaStepDef]] = {}  # saga_id → step defs
        self._completion_events: Dict[str, asyncio.Event] = {}
        self._tasks: Dict[str, asyncio.Task] = {}  # saga_id → background task

    async def _emit(self, topic: str, saga_id: str, **extra: Any) -> None:
        """Emit an event to the message bus (if connected)."""
        if self._bus is not None:
            await self._bus.publish(Message(
                sender=f"saga:{saga_id}",
                topic=topic,
                payload={"saga_id": saga_id, **extra},
            ))

    async def _run_action(self, fn: ActionFn) -> Any:
        """Run an action that may be sync or async."""
        result = fn()
        if inspect.isawaitable(result):
            return await result
        return result

    async def _run_compensation(self, fn: CompensationFn, result: Any) -> None:
        """Run a compensation that may be sync or async."""
        comp_result = fn(result)
        if inspect.isawaitable(comp_result):
            await comp_result

    async def _execute_step(
        self,
        saga_id: str,
        step_def: SagaStepDef,
        record: StepRecord,
    ) -> Any:
        """Execute a single step with retry and timeout."""
        retry = step_def.retry
        timeout = step_def.timeout or self._default_timeout
        last_error: Optional[Exception] = None

        for attempt in range(1 + retry.max_retries):
            record.attempts = attempt + 1
            record.status = StepStatus.RUNNING
            record.started_at = datetime.now().isoformat()

            try:
                if timeout > 0:
                    result = await asyncio.wait_for(
                        self._run_action(step_def.action),
                        timeout=timeout,
                    )
                else:
                    result = await self._run_action(step_def.action)

                record.status = StepStatus.COMPLETED
                record.result = result
                record.completed_at = datetime.now().isoformat()

                await self._emit(
                    Topics.SAGA_CHECKPOINT,
                    saga_id,
                    step=step_def.name,
                    status="completed",
                    attempt=attempt + 1,
                )
                return result

            except Exception as exc:
                last_error = exc
                should_retry = (
                    attempt < retry.max_retries
                    and (retry.retry_on is None or isinstance(exc, retry.retry_on))
                )
                if should_retry:
                    delay = retry.delay * (retry.backoff_factor ** attempt)
                    await asyncio.sleep(delay)
                    continue
                break

        # All retries exhausted
        record.status = StepStatus.FAILED
        record.error = str(last_error)
        record.completed_at = datetime.now().isoformat()
        raise last_error  # type: ignore[misc]

    async def start(
        self,
        name: str,
        steps: List[SagaStepDef],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new saga. Returns the saga_id.

        The saga runs in the background. Use wait() or get_status()
        to track progress.
        """
        saga_id = uuid.uuid4().hex[:12]
        record = SagaRecord(
            saga_id=saga_id,
            name=name,
            metadata=metadata or {},
        )
        for step_def in steps:
            record.steps[step_def.name] = StepRecord(name=step_def.name)

        self._sagas[saga_id] = record
        self._step_defs[saga_id] = steps
        self._completion_events[saga_id] = asyncio.Event()

        # Launch execution in background (store reference to prevent GC)
        task = asyncio.create_task(self._run_saga(saga_id))
        self._tasks[saga_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(saga_id, None))

        await self._emit(Topics.SAGA_START, saga_id, name=name)
        return saga_id

    async def _run_saga(self, saga_id: str) -> None:
        """Execute all saga steps in order, compensating on failure."""
        record = self._sagas[saga_id]
        step_defs = self._step_defs[saga_id]
        record.status = SagaStatus.RUNNING
        completed_steps: List[Tuple[SagaStepDef, Any]] = []

        try:
            for step_def in step_defs:
                step_record = record.steps[step_def.name]

                # Check dependencies
                for dep in step_def.depends_on:
                    dep_record = record.steps.get(dep)
                    if dep_record is None or dep_record.status != StepStatus.COMPLETED:
                        step_record.status = StepStatus.SKIPPED
                        step_record.error = f"Dependency '{dep}' not completed"
                        raise RuntimeError(
                            f"Step '{step_def.name}' dependency '{dep}' not met"
                        )

                result = await self._execute_step(saga_id, step_def, step_record)
                completed_steps.append((step_def, result))

            record.status = SagaStatus.COMPLETED
            record.completed_at = datetime.now().isoformat()
            await self._emit(Topics.SAGA_COMPLETE, saga_id, name=record.name)

        except Exception as exc:
            record.status = SagaStatus.COMPENSATING
            record.error = str(exc)
            await self._emit(
                Topics.SAGA_FAILED,
                saga_id,
                name=record.name,
                error=str(exc),
            )

            # Compensate in reverse order
            compensation_failed = False
            for step_def, result in reversed(completed_steps):
                step_record = record.steps[step_def.name]
                step_record.status = StepStatus.COMPENSATING
                try:
                    await self._run_compensation(step_def.compensation, result)
                    step_record.status = StepStatus.COMPENSATED
                    await self._emit(
                        Topics.SAGA_COMPENSATE,
                        saga_id,
                        step=step_def.name,
                        status="compensated",
                    )
                except Exception as comp_exc:
                    compensation_failed = True
                    step_record.status = StepStatus.FAILED
                    step_record.error = (
                        f"Action failed: {exc}; Compensation also failed: {comp_exc}"
                    )

            record.status = (
                SagaStatus.FAILED if compensation_failed
                else SagaStatus.COMPENSATED
            )
            record.completed_at = datetime.now().isoformat()

        finally:
            event = self._completion_events.get(saga_id)
            if event:
                event.set()

    async def wait(self, saga_id: str, timeout: Optional[float] = None) -> SagaRecord:
        """Wait for a saga to complete (or fail). Returns the SagaRecord."""
        event = self._completion_events.get(saga_id)
        if event is None:
            raise KeyError(f"Unknown saga: {saga_id}")
        if timeout:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        else:
            await event.wait()
        return self._sagas[saga_id]

    def get_status(self, saga_id: str) -> SagaRecord:
        """Get the current status of a saga."""
        record = self._sagas.get(saga_id)
        if record is None:
            raise KeyError(f"Unknown saga: {saga_id}")
        return record

    def list_sagas(
        self,
        status: Optional[SagaStatus] = None,
    ) -> List[SagaRecord]:
        """List all sagas, optionally filtered by status."""
        sagas = list(self._sagas.values())
        if status is not None:
            sagas = [s for s in sagas if s.status == status]
        return sagas

    def clear(self) -> None:
        """Clear all saga records and cancel pending tasks."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        self._sagas.clear()
        self._step_defs.clear()
        self._completion_events.clear()
        self._tasks.clear()
