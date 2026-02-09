"""
Dynamic Task Router — Phase 3.

Capability-based routing that matches task requirements to agent
capabilities. Supports:

  - Capability matching (role, skills, model access)
  - Load balancing (round-robin, least-busy)
  - Agent health tracking
  - Routing rules and overrides

Usage:
    router = TaskRouter()
    router.register("builder-1", AgentCapability(role="builder", skills=["python"]))
    router.register("builder-2", AgentCapability(role="builder", skills=["js"]))
    router.register("verifier-1", AgentCapability(role="verifier"))

    agent_id = router.route(TaskNode(id="t1", name="Write Python code", role="builder"))
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from stateful_repl.agents import AgentCapability, AgentStatus
from stateful_repl.planner import TaskNode


# ─────────────────────────────────────────────────────────
# Routing Strategy
# ─────────────────────────────────────────────────────────

class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_BUSY = "least_busy"
    CAPABILITY_MATCH = "capability_match"
    RANDOM = "random"


# ─────────────────────────────────────────────────────────
# Agent Health
# ─────────────────────────────────────────────────────────

@dataclass
class AgentHealth:
    """Tracks an agent's health and workload."""
    agent_id: str
    capability: AgentCapability
    status: AgentStatus = AgentStatus.IDLE
    active_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    last_heartbeat: str = field(default_factory=lambda: datetime.now().isoformat())
    avg_task_duration_ms: float = 0.0
    error_rate: float = 0.0

    @property
    def is_available(self) -> bool:
        """Agent is available if idle or has capacity."""
        if self.status in (AgentStatus.ERROR, AgentStatus.STOPPED):
            return False
        return self.active_tasks < self.capability.max_concurrent_tasks

    def record_completion(self, duration_ms: float) -> None:
        """Update stats after a task completes."""
        self.total_completed += 1
        self.active_tasks = max(0, self.active_tasks - 1)
        # Running average
        total = self.total_completed + self.total_failed
        self.avg_task_duration_ms = (
            (self.avg_task_duration_ms * (total - 1) + duration_ms) / total
        )
        self._update_error_rate()

    def record_failure(self) -> None:
        """Update stats after a task fails."""
        self.total_failed += 1
        self.active_tasks = max(0, self.active_tasks - 1)
        self._update_error_rate()

    def _update_error_rate(self) -> None:
        total = self.total_completed + self.total_failed
        self.error_rate = self.total_failed / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────
# Routing Rule
# ─────────────────────────────────────────────────────────

@dataclass
class RoutingRule:
    """Custom routing rule that overrides default capability matching."""
    name: str
    match_role: Optional[str] = None
    match_skills: Optional[List[str]] = None  # task must require these
    prefer_agent: Optional[str] = None         # route to this specific agent
    require_model: bool = False
    priority: int = 0  # Higher = checked first

    def matches(self, task: TaskNode) -> bool:
        """Check if this rule applies to the given task."""
        if self.match_role and task.role != self.match_role:
            return False
        if self.match_skills:
            task_skills = task.metadata.get("skills", [])
            if not all(s in task_skills for s in self.match_skills):
                return False
        return True


# ─────────────────────────────────────────────────────────
# Task Router
# ─────────────────────────────────────────────────────────

class TaskRouter:
    """
    Routes tasks to agents based on capabilities, load, and rules.

    Registration:
        router.register(agent_id, capability)

    Routing:
        agent_id = router.route(task)  # returns best agent or None

    Strategies:
        - ROUND_ROBIN: cycle through available agents
        - LEAST_BUSY: pick agent with fewest active tasks
        - CAPABILITY_MATCH: best skill overlap (default)
    """

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH):
        self.strategy = strategy
        self._agents: Dict[str, AgentHealth] = {}
        self._rules: List[RoutingRule] = []
        self._round_robin_idx: Dict[str, int] = defaultdict(int)  # role → index

    def register(self, agent_id: str, capability: AgentCapability) -> None:
        """Register an agent with its capabilities."""
        self._agents[agent_id] = AgentHealth(
            agent_id=agent_id,
            capability=capability,
        )

    def unregister(self, agent_id: str) -> None:
        """Remove an agent from the router."""
        self._agents.pop(agent_id, None)

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a custom routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def update_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update an agent's status."""
        health = self._agents.get(agent_id)
        if health:
            health.status = status
            health.last_heartbeat = datetime.now().isoformat()

    def record_task_start(self, agent_id: str) -> None:
        """Record that a task has been assigned to an agent."""
        health = self._agents.get(agent_id)
        if health:
            health.active_tasks += 1

    def record_task_complete(self, agent_id: str, duration_ms: float) -> None:
        """Record a task completion."""
        health = self._agents.get(agent_id)
        if health:
            health.record_completion(duration_ms)

    def record_task_failure(self, agent_id: str) -> None:
        """Record a task failure."""
        health = self._agents.get(agent_id)
        if health:
            health.record_failure()

    def route(self, task: TaskNode) -> Optional[str]:
        """
        Find the best agent to handle a task.

        Returns agent_id or None if no suitable agent is available.

        Resolution order:
          1. Check custom routing rules
          2. Filter by role match
          3. Filter by availability
          4. Apply strategy (round-robin, least-busy, or capability-match)
        """
        # 1. Check custom rules first
        for rule in self._rules:
            if rule.matches(task) and rule.prefer_agent:
                health = self._agents.get(rule.prefer_agent)
                if health and health.is_available:
                    return rule.prefer_agent

        # 2. Filter candidates by role
        candidates = [
            h for h in self._agents.values()
            if h.capability.role == task.role and h.is_available
        ]

        if not candidates:
            return None

        # 3. Apply additional filters from rules
        for rule in self._rules:
            if rule.matches(task) and rule.require_model:
                candidates = [c for c in candidates if c.capability.has_model]

        if not candidates:
            return None

        # 4. Apply routing strategy
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._route_round_robin(task.role, candidates)
        elif self.strategy == RoutingStrategy.LEAST_BUSY:
            return self._route_least_busy(candidates)
        elif self.strategy == RoutingStrategy.CAPABILITY_MATCH:
            return self._route_capability_match(task, candidates)
        else:
            return candidates[0].agent_id if candidates else None

    def _route_round_robin(self, role: str, candidates: List[AgentHealth]) -> str:
        """Cycle through available agents."""
        idx = self._round_robin_idx[role] % len(candidates)
        self._round_robin_idx[role] = idx + 1
        return candidates[idx].agent_id

    def _route_least_busy(self, candidates: List[AgentHealth]) -> str:
        """Pick the agent with fewest active tasks."""
        return min(candidates, key=lambda h: h.active_tasks).agent_id

    def _route_capability_match(
        self,
        task: TaskNode,
        candidates: List[AgentHealth],
    ) -> str:
        """Pick the agent with the best skill overlap."""
        required_skills = set(task.metadata.get("skills", []))

        if not required_skills:
            # No skill requirements — fall back to least-busy
            return self._route_least_busy(candidates)

        def score(h: AgentHealth) -> float:
            agent_skills = set(h.capability.skills)
            overlap = len(required_skills & agent_skills)
            # Penalize for high error rate
            reliability = 1.0 - h.error_rate
            return overlap * reliability

        best = max(candidates, key=score)
        return best.agent_id

    def get_health(self, agent_id: str) -> Optional[AgentHealth]:
        """Get health info for an agent."""
        return self._agents.get(agent_id)

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """Get health info for all agents."""
        return dict(self._agents)

    def get_available_agents(self, role: Optional[str] = None) -> List[str]:
        """List all available agents, optionally filtered by role."""
        agents = [
            h.agent_id for h in self._agents.values()
            if h.is_available
        ]
        if role:
            agents = [
                a for a in agents
                if self._agents[a].capability.role == role
            ]
        return agents

    def summary(self) -> Dict[str, Any]:
        """Get a summary of the router's state."""
        by_role: Dict[str, int] = defaultdict(int)
        for h in self._agents.values():
            by_role[h.capability.role] += 1

        return {
            "total_agents": len(self._agents),
            "available_agents": len(self.get_available_agents()),
            "agents_by_role": dict(by_role),
            "routing_strategy": self.strategy.value,
            "routing_rules": len(self._rules),
        }
