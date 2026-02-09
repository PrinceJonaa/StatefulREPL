"""
HALO Hierarchical Planner — Phase 3.

Decomposes high-level goals into Directed Acyclic Graphs (DAGs) of
subtasks with dependency tracking, priority ordering, and execution
scheduling.

HALO = Hierarchical Agent-Led Orchestration

Features:
  - Goal → subtask decomposition (manual or model-assisted)
  - DAG validation (cycle detection, dependency resolution)
  - Topological execution ordering
  - Priority-based scheduling within parallel tiers
  - Progress tracking with percentage completion

Usage (manual):
    planner = TaskPlanner()
    plan = planner.create_plan("Build feature X")
    plan.add_task(TaskNode(id="design", name="Design API", role="builder"))
    plan.add_task(TaskNode(id="impl", name="Implement", role="builder", depends_on=["design"]))
    plan.add_task(TaskNode(id="test", name="Test", role="verifier", depends_on=["impl"]))

    schedule = planner.schedule(plan)
    # Returns: [["design"], ["impl"], ["test"]]  — 3 sequential tiers

Usage (model-assisted):
    planner = TaskPlanner(model=adapter)
    plan = await planner.decompose("Add user authentication with OAuth2")
"""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from stateful_repl.models import ModelAdapter


# ─────────────────────────────────────────────────────────
# Task Types
# ─────────────────────────────────────────────────────────

class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"       # All dependencies met, can be dispatched
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"   # Dependency failed
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskNode:
    """A single task in the execution DAG."""

    id: str
    name: str
    description: str = ""
    role: str = "builder"  # coordinator, builder, verifier, distiller
    priority: TaskPriority = TaskPriority.NORMAL
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None  # agent_id
    result: Any = None
    error: Optional[str] = None
    estimated_duration: Optional[float] = None  # seconds
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "role": self.role,
            "priority": self.priority.name,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "assigned_to": self.assigned_to,
            "error": self.error,
            "estimated_duration": self.estimated_duration,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class TaskPlan:
    """A DAG of tasks representing a decomposed goal."""

    plan_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str = ""
    tasks: Dict[str, TaskNode] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_task(self, task: TaskNode) -> None:
        """Add a task to the plan."""
        self.tasks[task.id] = task

    def remove_task(self, task_id: str) -> None:
        """Remove a task and clean up references."""
        self.tasks.pop(task_id, None)
        for task in self.tasks.values():
            if task_id in task.depends_on:
                task.depends_on.remove(task_id)

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self.tasks.get(task_id)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def completed_count(self) -> int:
        return sum(
            1 for t in self.tasks.values()
            if t.status == TaskStatus.COMPLETED
        )

    @property
    def progress(self) -> float:
        """Completion percentage (0.0–1.0)."""
        if not self.tasks:
            return 1.0
        return self.completed_count / self.task_count

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for t in self.tasks.values()
        )

    @property
    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "progress": self.progress,
            "task_count": self.task_count,
            "completed_count": self.completed_count,
            "is_complete": self.is_complete,
            "has_failures": self.has_failures,
            "created_at": self.created_at,
            "tasks": {tid: t.to_dict() for tid, t in self.tasks.items()},
        }


# ─────────────────────────────────────────────────────────
# DAG Validation
# ─────────────────────────────────────────────────────────

class PlanValidationError(Exception):
    """Raised when a plan has structural issues."""
    pass


def validate_dag(plan: TaskPlan) -> List[str]:
    """
    Validate the task DAG.

    Returns a list of issues (empty = valid).
    Checks:
      - No cycles (would cause deadlock)
      - All dependency references exist
      - No self-dependencies
    """
    issues: List[str] = []
    task_ids = set(plan.tasks.keys())

    for task in plan.tasks.values():
        # Self-dependency
        if task.id in task.depends_on:
            issues.append(f"Task '{task.id}' depends on itself")

        # Missing dependency
        for dep in task.depends_on:
            if dep not in task_ids:
                issues.append(
                    f"Task '{task.id}' depends on unknown task '{dep}'"
                )

    # Cycle detection via Kahn's algorithm
    if not issues:
        in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
        adj: Dict[str, List[str]] = defaultdict(list)
        for task in plan.tasks.values():
            for dep in task.depends_on:
                adj[dep].append(task.id)
                in_degree[task.id] += 1

        queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited < len(task_ids):
            issues.append("Cycle detected in task dependencies")

    return issues


# ─────────────────────────────────────────────────────────
# Topological Scheduling
# ─────────────────────────────────────────────────────────

def topological_sort(plan: TaskPlan) -> List[List[str]]:
    """
    Compute execution tiers via topological sort.

    Returns a list of tiers. Tasks within each tier can run in parallel.
    Tasks in tier N+1 depend on tasks in tiers 0..N.

    Example:
        [["design"], ["impl_a", "impl_b"], ["test"], ["deploy"]]
        → Tier 0: design (no deps)
        → Tier 1: impl_a, impl_b (both depend on design)
        → Tier 2: test (depends on impl_a, impl_b)
        → Tier 3: deploy (depends on test)
    """
    issues = validate_dag(plan)
    if issues:
        raise PlanValidationError("; ".join(issues))

    task_ids = set(plan.tasks.keys())
    if not task_ids:
        return []

    # Build adjacency and in-degree
    in_degree: Dict[str, int] = {tid: 0 for tid in task_ids}
    adj: Dict[str, List[str]] = defaultdict(list)
    for task in plan.tasks.values():
        for dep in task.depends_on:
            adj[dep].append(task.id)
            in_degree[task.id] += 1

    # BFS layer by layer
    tiers: List[List[str]] = []
    current = [tid for tid, deg in in_degree.items() if deg == 0]

    while current:
        # Sort within tier by priority (highest first)
        current.sort(
            key=lambda tid: plan.tasks[tid].priority.value,
            reverse=True,
        )
        tiers.append(current)

        next_tier: List[str] = []
        for tid in current:
            for neighbor in adj[tid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    next_tier.append(neighbor)
        current = next_tier

    return tiers


# ─────────────────────────────────────────────────────────
# Task Planner
# ─────────────────────────────────────────────────────────

class TaskPlanner:
    """
    High-level planner that creates and manages TaskPlans.

    Can operate in two modes:
      1. Manual: caller builds the plan via add_task()
      2. Model-assisted: decompose() uses an LLM to break down goals
    """

    def __init__(self, model: Optional["ModelAdapter"] = None):
        self._model = model
        self._plans: Dict[str, TaskPlan] = {}

    def create_plan(self, goal: str, **metadata: Any) -> TaskPlan:
        """Create a new empty plan for a goal."""
        plan = TaskPlan(goal=goal, metadata=metadata)
        self._plans[plan.plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        return self._plans.get(plan_id)

    def list_plans(self) -> List[TaskPlan]:
        return list(self._plans.values())

    def schedule(self, plan: TaskPlan) -> List[List[str]]:
        """Compute execution schedule (topological tiers) for a plan."""
        return topological_sort(plan)

    def get_ready_tasks(self, plan: TaskPlan) -> List[TaskNode]:
        """
        Get all tasks whose dependencies are met and haven't started.

        A task is ready if:
          - status is PENDING or READY
          - all depends_on tasks are COMPLETED
        """
        ready: List[TaskNode] = []
        for task in plan.tasks.values():
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            deps_met = all(
                dep in plan.tasks and plan.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )
            if deps_met:
                task.status = TaskStatus.READY
                ready.append(task)

        # Sort by priority (highest first)
        ready.sort(key=lambda t: t.priority.value, reverse=True)
        return ready

    def mark_running(self, plan: TaskPlan, task_id: str, agent_id: str) -> None:
        """Mark a task as running, assigned to an agent.

        Raises:
            KeyError: If task_id does not exist in the plan.
        """
        task = plan.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found in plan '{plan.plan_id}'")
        task.status = TaskStatus.RUNNING
        task.assigned_to = agent_id
        task.started_at = datetime.now().isoformat()

    def mark_completed(self, plan: TaskPlan, task_id: str, result: Any = None) -> None:
        """Mark a task as completed.

        Raises:
            KeyError: If task_id does not exist in the plan.
        """
        task = plan.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found in plan '{plan.plan_id}'")
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = datetime.now().isoformat()

    def mark_failed(self, plan: TaskPlan, task_id: str, error: str) -> None:
        """Mark a task as failed and block dependents.

        Raises:
            KeyError: If task_id does not exist in the plan.
        """
        task = plan.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found in plan '{plan.plan_id}'")
        task.status = TaskStatus.FAILED
        task.error = error
        task.completed_at = datetime.now().isoformat()

        # Block all downstream tasks
        self._propagate_block(plan, task_id)

    def _propagate_block(self, plan: TaskPlan, failed_id: str) -> None:
        """Block all tasks that transitively depend on a failed task."""
        blocked: Set[str] = set()
        queue = deque([failed_id])

        while queue:
            current = queue.popleft()
            for task in plan.tasks.values():
                if current in task.depends_on and task.id not in blocked:
                    if task.status in (TaskStatus.PENDING, TaskStatus.READY):
                        task.status = TaskStatus.BLOCKED
                        task.error = f"Blocked by failed task '{failed_id}'"
                        blocked.add(task.id)
                        queue.append(task.id)

    async def decompose(
        self,
        goal: str,
        context: Optional[str] = None,
        max_tasks: int = 10,
    ) -> TaskPlan:
        """
        Use an LLM to decompose a goal into a task plan.

        Requires a model adapter. Falls back to a single-task plan
        if no model is available.
        """
        plan = self.create_plan(goal)

        if self._model is None:
            # No model: create a single task
            plan.add_task(TaskNode(
                id="task-1",
                name=goal,
                description=f"Complete: {goal}",
                role="builder",
            ))
            return plan

        # Model-assisted decomposition
        system_prompt = (
            "You are a task planner. Decompose the given goal into subtasks.\n"
            "Output one task per line in this exact format:\n"
            "TASK:<id>|<name>|<role>|<depends_on_csv>|<priority>\n\n"
            "Rules:\n"
            "- id: short kebab-case identifier\n"
            "- role: one of coordinator, builder, verifier, distiller\n"
            "- depends_on_csv: comma-separated task IDs (or 'none')\n"
            "- priority: LOW, NORMAL, HIGH, or CRITICAL\n"
            f"- Maximum {max_tasks} tasks\n"
            "- Ensure the DAG is valid (no cycles)\n"
            "- Include a verification/testing task\n"
        )

        prompt = f"Goal: {goal}"
        if context:
            prompt += f"\n\nContext:\n{context}"

        response = self._model.complete(
            prompt,
            system=system_prompt,
            temperature=0.2,
            max_tokens=2048,
        )

        # Parse response
        priority_map = {
            "LOW": TaskPriority.LOW,
            "NORMAL": TaskPriority.NORMAL,
            "HIGH": TaskPriority.HIGH,
            "CRITICAL": TaskPriority.CRITICAL,
        }

        for line in response.text.strip().splitlines():
            line = line.strip()
            if not line.startswith("TASK:"):
                continue
            parts = line[5:].split("|")
            if len(parts) < 4:
                continue

            task_id = parts[0].strip()
            name = parts[1].strip()
            role = parts[2].strip().lower()
            deps_str = parts[3].strip()
            priority_str = parts[4].strip().upper() if len(parts) > 4 else "NORMAL"

            deps = (
                [d.strip() for d in deps_str.split(",") if d.strip() != "none"]
                if deps_str.lower() != "none"
                else []
            )

            plan.add_task(TaskNode(
                id=task_id,
                name=name,
                role=role if role in ("coordinator", "builder", "verifier", "distiller") else "builder",
                depends_on=deps,
                priority=priority_map.get(priority_str, TaskPriority.NORMAL),
            ))

        # Validate — if invalid, fall back to single task
        issues = validate_dag(plan)
        if issues:
            plan.tasks.clear()
            plan.add_task(TaskNode(
                id="task-1",
                name=goal,
                description=f"Decomposition failed ({'; '.join(issues)}). Running as single task.",
                role="builder",
            ))

        return plan
