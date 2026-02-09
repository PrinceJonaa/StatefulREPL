"""
Loom Agent Roles — Phase 3.

Concrete implementations of the four Loom agent roles:
  - CoordinatorLoom: decomposes tasks, manages sagas, routes work
  - BuilderLoom: executes subtasks (code/artifacts) in sandbox
  - VerifierLoom: runs quality + hallucination checks
  - DistillerLoom: integrates multiple outputs via MCRD

All agents share a base class (LoomAgent) providing:
  - Unique agent ID and role name
  - Message bus integration
  - Event emission
  - L1/L2/L3 state access

Usage:
    bus = InProcessBus()
    coordinator = CoordinatorLoom(bus=bus)
    builder = BuilderLoom(bus=bus)
    verifier = VerifierLoom(bus=bus)

    # Coordinator decomposes a goal and dispatches
    await coordinator.handle_goal("Add OAuth2 authentication")
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from stateful_repl.loom_state import LoomREPL
from stateful_repl.message_bus import InProcessBus, Message, MessageBus, Topics
from stateful_repl.planner import (
    TaskNode,
    TaskPlan,
    TaskPlanner,
    TaskPriority,
    TaskStatus,
)

if TYPE_CHECKING:
    from stateful_repl.models import ModelAdapter


# ─────────────────────────────────────────────────────────
# Agent Status
# ─────────────────────────────────────────────────────────

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"


# ─────────────────────────────────────────────────────────
# Capability Declaration
# ─────────────────────────────────────────────────────────

@dataclass
class AgentCapability:
    """Declares what an agent can do for capability-based routing."""
    role: str                          # coordinator, builder, verifier, distiller
    skills: List[str] = field(default_factory=list)  # e.g. ["python", "testing", "api"]
    max_concurrent_tasks: int = 1
    has_model: bool = False
    has_sandbox: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# Task Result
# ─────────────────────────────────────────────────────────

@dataclass
class TaskResult:
    """Result from an agent completing a task."""
    task_id: str
    agent_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)  # produced artifact IDs
    quality_score: Optional[float] = None
    duration_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "success": self.success,
            "output": str(self.output) if self.output is not None else None,
            "error": self.error,
            "artifacts": self.artifacts,
            "quality_score": self.quality_score,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────
# Base Agent
# ─────────────────────────────────────────────────────────

class LoomAgent(ABC):
    """
    Base class for all Loom agents.

    Provides:
      - Identity (agent_id, role)
      - Message bus integration
      - Status tracking
      - Event emission
      - State access via LoomREPL
    """

    role: str = "agent"

    def __init__(
        self,
        bus: Optional[MessageBus] = None,
        model: Optional["ModelAdapter"] = None,
        loom: Optional[LoomREPL] = None,
        agent_id: Optional[str] = None,
    ):
        self.agent_id = agent_id or f"{self.role}-{uuid.uuid4().hex[:6]}"
        self.bus = bus
        self.model = model
        self.loom = loom
        self.status = AgentStatus.IDLE
        self._active_tasks: Dict[str, TaskNode] = {}
        self._results: List[TaskResult] = []

    @property
    def capability(self) -> AgentCapability:
        """Declare this agent's capabilities."""
        return AgentCapability(
            role=self.role,
            has_model=self.model is not None,
        )

    async def _emit(self, topic: str, **payload: Any) -> None:
        """Publish a message on the bus."""
        if self.bus:
            await self.bus.publish(Message(
                sender=self.agent_id,
                topic=topic,
                payload=payload,
            ))

    async def _send(self, target: str, topic: str, **payload: Any) -> None:
        """Send a direct message to another agent."""
        if self.bus:
            await self.bus.send(target, Message(
                sender=self.agent_id,
                topic=topic,
                payload=payload,
                reply_to=self.agent_id,
            ))

    async def _receive(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive next message from inbox."""
        if self.bus:
            return await self.bus.receive(self.agent_id, timeout=timeout)
        return None

    async def start(self) -> None:
        """Register on the bus and announce readiness."""
        if self.bus:
            await self.bus.subscribe(self.agent_id, self._default_topics())
            await self._emit(Topics.AGENT_READY, role=self.role)

    async def stop(self) -> None:
        """Unsubscribe and shut down."""
        self.status = AgentStatus.STOPPED
        if self.bus:
            await self.bus.unsubscribe(self.agent_id)

    def _default_topics(self) -> List[str]:
        """Topics this agent subscribes to by default."""
        return [
            f"task.assign.{self.role}",
            f"direct.{self.agent_id}",
            "agent.*",
        ]

    @abstractmethod
    async def handle_task(self, task: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """Execute a single task. Subclasses must implement."""
        ...

    async def run_loop(self, timeout: float = 1.0) -> None:
        """
        Main event loop: receive messages, dispatch tasks.

        Runs until stop() is called or agent is stopped externally.
        """
        self.status = AgentStatus.IDLE
        while self.status != AgentStatus.STOPPED:
            msg = await self._receive(timeout=timeout)
            if msg is None:
                continue

            if msg.topic.startswith("task.assign"):
                await self._on_task_assign(msg)
            elif msg.topic == Topics.AGENT_READY:
                pass  # Could track peers

    async def _on_task_assign(self, msg: Message) -> None:
        """Handle an incoming task assignment."""
        task_data = msg.payload
        task = TaskNode(
            id=task_data.get("task_id", uuid.uuid4().hex[:8]),
            name=task_data.get("name", "unnamed"),
            description=task_data.get("description", ""),
            role=task_data.get("role", self.role),
            metadata=task_data.get("metadata", {}),
        )

        self.status = AgentStatus.BUSY
        self._active_tasks[task.id] = task
        await self._emit(Topics.AGENT_BUSY, task_id=task.id)

        try:
            result = await self.handle_task(task, task_data.get("context", {}))
            self._results.append(result)

            await self._emit(
                Topics.TASK_COMPLETE,
                task_id=task.id,
                result=result.to_dict(),
            )
        except Exception as exc:
            result = TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(exc),
            )
            self._results.append(result)
            await self._emit(
                Topics.TASK_FAILED,
                task_id=task.id,
                error=str(exc),
            )
        finally:
            self._active_tasks.pop(task.id, None)
            self.status = AgentStatus.IDLE


# ─────────────────────────────────────────────────────────
# Coordinator Loom
# ─────────────────────────────────────────────────────────

class CoordinatorLoom(LoomAgent):
    """
    Decomposes goals into task DAGs and dispatches to other agents.

    Responsibilities:
      - Goal decomposition via TaskPlanner (manual or model-assisted)
      - Task scheduling (topological ordering)
      - Dispatching ready tasks to appropriate agents
      - Tracking progress and handling failures
      - Saga management for multi-step workflows
    """

    role = "coordinator"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.planner = TaskPlanner(model=self.model)
        self._active_plans: Dict[str, TaskPlan] = {}
        self._agent_registry: Dict[str, AgentCapability] = {}

    def _default_topics(self) -> List[str]:
        return [
            "task.assign.coordinator",
            f"direct.{self.agent_id}",
            "task.complete",
            "task.failed",
            "agent.ready",
            "quality.result",
        ]

    def register_agent(self, agent_id: str, capability: AgentCapability) -> None:
        """Register an agent's capabilities for routing."""
        self._agent_registry[agent_id] = capability

    async def handle_goal(
        self,
        goal: str,
        context: Optional[str] = None,
    ) -> TaskPlan:
        """
        Decompose a goal and begin dispatching tasks.

        Returns the TaskPlan (tasks start executing in background).
        """
        if self.model:
            plan = await self.planner.decompose(goal, context=context)
        else:
            plan = self.planner.create_plan(goal)
            plan.add_task(TaskNode(
                id="task-1",
                name=goal,
                role="builder",
            ))

        self._active_plans[plan.plan_id] = plan
        await self._dispatch_ready(plan)
        return plan

    async def _dispatch_ready(self, plan: TaskPlan) -> None:
        """Find and dispatch all ready tasks in a plan."""
        ready = self.planner.get_ready_tasks(plan)
        for task in ready:
            target = self._find_agent_for(task)
            if target and self.bus:
                self.planner.mark_running(plan, task.id, target)
                await self._send(
                    target,
                    f"task.assign.{task.role}",
                    task_id=task.id,
                    name=task.name,
                    description=task.description,
                    role=task.role,
                    context={"plan_id": plan.plan_id},
                    metadata=task.metadata,
                )

    def _find_agent_for(self, task: TaskNode) -> Optional[str]:
        """Find a registered agent capable of handling this task."""
        for agent_id, cap in self._agent_registry.items():
            if cap.role == task.role:
                return agent_id
        return None

    async def handle_task(self, task: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """Coordinator's own task handler (for meta-tasks like planning)."""
        plan = await self.handle_goal(task.name)
        return TaskResult(
            task_id=task.id,
            agent_id=self.agent_id,
            success=True,
            output={"plan_id": plan.plan_id, "task_count": plan.task_count},
        )

    async def on_task_complete(self, task_id: str, result: Dict[str, Any]) -> None:
        """Handle a task completion notification from a worker agent."""
        for plan in self._active_plans.values():
            task = plan.get_task(task_id)
            if task:
                self.planner.mark_completed(plan, task_id, result)
                # Dispatch next ready tasks
                await self._dispatch_ready(plan)

                if plan.is_complete:
                    await self._emit(
                        "plan.complete",
                        plan_id=plan.plan_id,
                        goal=plan.goal,
                    )
                break

    async def on_task_failed(self, task_id: str, error: str) -> None:
        """Handle a task failure notification."""
        for plan in self._active_plans.values():
            task = plan.get_task(task_id)
            if task:
                self.planner.mark_failed(plan, task_id, error)
                if plan.has_failures:
                    await self._emit(
                        "plan.degraded",
                        plan_id=plan.plan_id,
                        failed_task=task_id,
                        error=error,
                    )
                break

    async def run_loop(self, timeout: float = 1.0) -> None:
        """Extended event loop that handles completions and failures."""
        self.status = AgentStatus.IDLE
        while self.status != AgentStatus.STOPPED:
            msg = await self._receive(timeout=timeout)
            if msg is None:
                continue

            if msg.topic.startswith("task.assign"):
                await self._on_task_assign(msg)
            elif msg.topic == Topics.TASK_COMPLETE:
                task_id = msg.payload.get("task_id", "")
                result = msg.payload.get("result", {})
                await self.on_task_complete(task_id, result)
            elif msg.topic == Topics.TASK_FAILED:
                task_id = msg.payload.get("task_id", "")
                error = msg.payload.get("error", "unknown")
                await self.on_task_failed(task_id, error)
            elif msg.topic == Topics.AGENT_READY:
                agent_id = msg.sender
                role = msg.payload.get("role", "")
                if role:
                    self.register_agent(agent_id, AgentCapability(role=role))


# ─────────────────────────────────────────────────────────
# Builder Loom
# ─────────────────────────────────────────────────────────

class BuilderLoom(LoomAgent):
    """
    Executes vertical slices — produces code, artifacts, and outputs.

    Can run code in sandboxed environment, call LLMs for generation,
    or perform direct computations.
    """

    role = "builder"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._sandbox = None

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            role="builder",
            skills=["code", "artifact"],
            has_model=self.model is not None,
            has_sandbox=True,
        )

    def _get_sandbox(self):
        """Lazy-load sandbox to avoid import cycle."""
        if self._sandbox is None:
            import tempfile
            from stateful_repl.sandbox import LoomSandbox
            if self.loom is None:
                # Create a temporary LoomREPL to avoid disk side effects
                tmp = tempfile.mktemp(suffix=".md")
                self.loom = LoomREPL(state_file=tmp, enable_events=False)
            self._sandbox = LoomSandbox(loom=self.loom)
        return self._sandbox

    async def handle_task(self, task: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """
        Execute a builder task.

        Strategy:
          1. If model available → generate via LLM
          2. If code in metadata → run in sandbox
          3. Otherwise → return task description as output
        """
        start = datetime.now()

        try:
            output = None
            artifacts: List[str] = []

            # Strategy 1: Model-assisted generation
            if self.model and task.description:
                system = (
                    "You are a builder agent. Produce the requested output. "
                    "Be precise and complete. Output only the result."
                )
                response = self.model.complete(
                    task.description,
                    system=system,
                    temperature=0.1,
                )
                output = response.text
                artifacts.append(f"builder-output-{task.id}")

            # Strategy 2: Sandbox execution
            elif "code" in task.metadata:
                sandbox = self._get_sandbox()
                result = sandbox.execute(task.metadata["code"])
                output = result
                if not isinstance(result, dict) or "error" not in result:
                    artifacts.append(f"sandbox-output-{task.id}")

            # Strategy 3: Pass-through
            else:
                output = f"Task '{task.name}' acknowledged. No model or code provided."

            elapsed = (datetime.now() - start).total_seconds() * 1000

            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=True,
                output=output,
                artifacts=artifacts,
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )


# ─────────────────────────────────────────────────────────
# Verifier Loom
# ─────────────────────────────────────────────────────────

class VerifierLoom(LoomAgent):
    """
    Runs quality checks, hallucination detection, and oracle validation.

    Verifies builder outputs against:
      - 7D quality vector (structural analysis)
      - Hallucination detection (4-method ensemble)
      - Claim validation (against sources)
    """

    role = "verifier"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._quality_evaluator = None
        self._hallucination_detector = None

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            role="verifier",
            skills=["quality", "hallucination", "validation"],
            has_model=self.model is not None,
        )

    def _get_quality_evaluator(self):
        if self._quality_evaluator is None:
            from stateful_repl.quality import QualityEvaluator
            self._quality_evaluator = QualityEvaluator(model=self.model)
        return self._quality_evaluator

    def _get_hallucination_detector(self):
        if self._hallucination_detector is None:
            from stateful_repl.hallucination import HallucinationDetector
            self._hallucination_detector = HallucinationDetector(model=self.model)
        return self._hallucination_detector

    def _default_topics(self) -> List[str]:
        return [
            "task.assign.verifier",
            f"direct.{self.agent_id}",
            "quality.request",
            "agent.*",
        ]

    async def handle_task(self, task: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """
        Run verification on the provided input.

        Expects context to contain:
          - text: the content to verify (required)
          - sources: reference sources (optional)
          - state: Loom state dict for quality eval (optional)
        """
        start = datetime.now()
        output: Dict[str, Any] = {}

        try:
            text = context.get("text", task.description)
            sources = context.get("sources", [])
            state = context.get("state")

            # Run hallucination detection
            detector = self._get_hallucination_detector()
            hal_score = detector.check_structural(text, sources=sources)
            output["hallucination"] = hal_score.to_dict()

            # Run quality evaluation if state provided
            if state:
                evaluator = self._get_quality_evaluator()
                quality = evaluator.evaluate(state)
                output["quality"] = quality.to_dict()
                output["quality_score"] = quality.aggregate_score

            # Determine pass/fail
            passed = not hal_score.is_hallucinated
            if state and output.get("quality_score", 1.0) < 0.3:
                passed = False

            elapsed = (datetime.now() - start).total_seconds() * 1000

            await self._emit(
                Topics.QUALITY_RESULT,
                task_id=task.id,
                passed=passed,
                hallucination_risk=hal_score.risk,
            )

            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=passed,
                output=output,
                quality_score=output.get("quality_score"),
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )

    async def verify_output(
        self,
        text: str,
        sources: Optional[List[str]] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> TaskResult:
        """Convenience: run verification directly without task wrapping."""
        task = TaskNode(id=f"verify-{uuid.uuid4().hex[:6]}", name="direct-verify")
        context = {"text": text, "sources": sources or [], "state": state}
        return await self.handle_task(task, context)


# ─────────────────────────────────────────────────────────
# Distiller Loom
# ─────────────────────────────────────────────────────────

class DistillerLoom(LoomAgent):
    """
    Integrates multiple outputs into a unified artifact via MCRD
    (Multi-source Cross-Reference Distillation).

    Steps:
      1. Profile each input (entities, claims, tensions)
      2. Cross-mirror for invariants, complements, contradictions
      3. Hold paradoxes explicitly
      4. Compose unified artifact
      5. Validate coverage + coherence
    """

    role = "distiller"

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            role="distiller",
            skills=["integration", "mcrd", "synthesis"],
            has_model=self.model is not None,
        )

    async def handle_task(self, task: TaskNode, context: Dict[str, Any]) -> TaskResult:
        """
        Integrate multiple sources into a unified output.

        Expects context to contain:
          - sources: list of text outputs to integrate
          - mode: "structural" (rule-based) or "model" (LLM-assisted)
        """
        start = datetime.now()

        try:
            sources = context.get("sources", [])
            mode = context.get("mode", "structural")

            if not sources:
                return TaskResult(
                    task_id=task.id,
                    agent_id=self.agent_id,
                    success=False,
                    error="No sources provided for distillation",
                )

            if mode == "model" and self.model:
                result = await self._model_distill(sources)
            else:
                result = self._structural_distill(sources)

            elapsed = (datetime.now() - start).total_seconds() * 1000

            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=True,
                output=result,
                artifacts=[f"distilled-{task.id}"],
                duration_ms=elapsed,
            )

        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return TaskResult(
                task_id=task.id,
                agent_id=self.agent_id,
                success=False,
                error=str(exc),
                duration_ms=elapsed,
            )

    def _structural_distill(self, sources: List[str]) -> Dict[str, Any]:
        """
        Rule-based MCRD (no model needed).

        Profiles each source, finds shared claims, identifies
        unique contributions and contradictions.
        """
        profiles = []
        for i, source in enumerate(sources):
            profile = self._profile_source(source, f"source-{i+1}")
            profiles.append(profile)

        # Cross-reference: find invariants and contradictions
        all_claims: Dict[str, List[int]] = {}  # claim → source indices
        for i, p in enumerate(profiles):
            for claim in p["claims"]:
                normalized = claim.lower().strip()
                all_claims.setdefault(normalized, []).append(i)

        invariants = [c for c, srcs in all_claims.items() if len(srcs) > 1]
        unique = {
            f"source-{i+1}": [c for c, srcs in all_claims.items() if srcs == [i]]
            for i in range(len(sources))
        }

        return {
            "source_count": len(sources),
            "profiles": profiles,
            "invariants": invariants,
            "unique_contributions": unique,
            "unified_text": self._compose_unified(profiles, invariants),
        }

    def _profile_source(self, text: str, label: str) -> Dict[str, Any]:
        """Extract a structural profile from text."""
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if s.strip()]
        words = text.lower().split()
        word_set = set(words)

        return {
            "label": label,
            "length": len(text),
            "sentence_count": len(sentences),
            "claims": sentences[:20],  # Treat sentences as claims
            "key_terms": sorted(
                {w for w in word_set if len(w) > 4},
                key=lambda w: words.count(w),
                reverse=True,
            )[:15],
        }

    def _compose_unified(
        self,
        profiles: List[Dict[str, Any]],
        invariants: List[str],
    ) -> str:
        """Compose a unified text from profiles."""
        sections = []

        if invariants:
            sections.append("## Shared Findings")
            for claim in invariants[:10]:
                sections.append(f"- {claim}")

        for p in profiles:
            sections.append(f"\n## From {p['label']}")
            unique_claims = [c for c in p["claims"] if c.lower().strip() not in invariants]
            for claim in unique_claims[:5]:
                sections.append(f"- {claim}")

        return "\n".join(sections) if sections else "No content to distill."

    async def _model_distill(self, sources: List[str]) -> Dict[str, Any]:
        """
        LLM-assisted MCRD.

        Uses the model to produce a high-quality integration.
        """
        if not self.model:
            return self._structural_distill(sources)

        source_block = "\n\n---\n\n".join(
            f"SOURCE {i+1}:\n{s}" for i, s in enumerate(sources)
        )

        system = (
            "You are a Distiller agent performing MCRD "
            "(Multi-source Cross-Reference Distillation).\n\n"
            "Steps:\n"
            "1. Profile each source (key claims, entities, unique contribution)\n"
            "2. Identify invariants (shared across sources)\n"
            "3. Identify contradictions — hold as paradoxes, do NOT erase either side\n"
            "4. Compose a unified output that preserves all unique contributions\n"
            "5. Note coverage: which source contributed what\n\n"
            "Output a clear, structured synthesis."
        )

        response = self.model.complete(
            source_block,
            system=system,
            temperature=0.2,
            max_tokens=4096,
        )

        return {
            "source_count": len(sources),
            "unified_text": response.text,
            "model_used": self.model.model,
        }

    async def distill(
        self,
        sources: List[str],
        mode: str = "structural",
    ) -> TaskResult:
        """Convenience: run distillation directly without task wrapping."""
        task = TaskNode(id=f"distill-{uuid.uuid4().hex[:6]}", name="direct-distill")
        context = {"sources": sources, "mode": mode}
        return await self.handle_task(task, context)
