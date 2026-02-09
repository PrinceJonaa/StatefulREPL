"""
Inter-Agent Message Bus — Phase 3.

Provides async message passing between Loom agents using
asyncio.Queue under the hood. Supports:

  - Point-to-point messaging (agent → agent)
  - Broadcast messaging (agent → all)
  - Topic-based pub/sub (agent → subscribers of topic)
  - Message history & replay

Upgrade path: swap asyncio.Queue for Redis Pub/Sub by
implementing the same MessageBus interface.

Usage:
    bus = InProcessBus()
    await bus.subscribe("verifier-1", topics=["task.complete", "quality.*"])
    await bus.publish(Message(sender="coordinator", topic="task.assign", payload={...}))
    msg = await bus.receive("verifier-1", timeout=5.0)
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


# ─────────────────────────────────────────────────────────
# Message Types
# ─────────────────────────────────────────────────────────

class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    """A single message on the bus."""

    sender: str
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: Optional[str] = None  # Links request → response
    reply_to: Optional[str] = None        # Agent to reply to
    ttl: Optional[float] = None           # Time-to-live in seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender": self.sender,
            "topic": self.topic,
            "payload": self.payload,
            "priority": self.priority.name,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
        }

    def reply(self, payload: Dict[str, Any], topic: Optional[str] = None) -> "Message":
        """Create a reply message to this message."""
        return Message(
            sender=self.reply_to or "",
            topic=topic or f"{self.topic}.reply",
            payload=payload,
            correlation_id=self.correlation_id or self.message_id,
            reply_to=self.sender,
        )


# ─────────────────────────────────────────────────────────
# Bus Protocol
# ─────────────────────────────────────────────────────────

class MessageBus:
    """Interface for all message bus implementations."""

    async def subscribe(self, agent_id: str, topics: List[str]) -> None: ...
    async def unsubscribe(self, agent_id: str, topics: Optional[List[str]] = None) -> None: ...
    async def publish(self, message: Message) -> None: ...
    async def send(self, agent_id: str, message: Message) -> None: ...
    async def receive(self, agent_id: str, timeout: Optional[float] = None) -> Optional[Message]: ...
    async def broadcast(self, message: Message) -> None: ...
    def get_history(self, topic: Optional[str] = None, limit: int = 100) -> List[Message]: ...


# ─────────────────────────────────────────────────────────
# In-Process Implementation
# ─────────────────────────────────────────────────────────

class InProcessBus(MessageBus):
    """
    Async in-process message bus backed by asyncio.Queue.

    Each agent gets its own inbox (Queue). Messages are routed
    by topic pattern matching (supports wildcards: task.* matches
    task.assign, task.complete, etc.).

    Thread-safe for asyncio tasks. Not suitable for multi-process.
    For multi-process, swap for RedisBus (same interface).
    """

    def __init__(self, max_queue_size: int = 1000, history_limit: int = 5000):
        self._inboxes: Dict[str, asyncio.Queue[Message]] = {}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # agent → topic patterns
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)  # topic pattern → agents
        self._max_queue_size = max_queue_size
        self._history: List[Message] = []
        self._history_limit = history_limit
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)  # topic → callbacks
        self._lock = asyncio.Lock()
        self._last_handler_error: Optional[Exception] = None

    def _ensure_inbox(self, agent_id: str) -> asyncio.Queue[Message]:
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = asyncio.Queue(maxsize=self._max_queue_size)
        return self._inboxes[agent_id]

    async def subscribe(self, agent_id: str, topics: List[str]) -> None:
        """Subscribe an agent to one or more topic patterns (supports glob wildcards)."""
        async with self._lock:
            self._ensure_inbox(agent_id)
            for topic in topics:
                self._subscriptions[agent_id].add(topic)
                self._topic_subscribers[topic].add(agent_id)

    async def unsubscribe(self, agent_id: str, topics: Optional[List[str]] = None) -> None:
        """Unsubscribe an agent from topics. If topics=None, unsubscribe from all."""
        async with self._lock:
            if topics is None:
                patterns = list(self._subscriptions.get(agent_id, set()))
            else:
                patterns = topics
            for pattern in patterns:
                self._subscriptions[agent_id].discard(pattern)
                self._topic_subscribers[pattern].discard(agent_id)

    def _match_subscribers(self, topic: str) -> Set[str]:
        """Find all agents subscribed to patterns matching this topic."""
        matched: Set[str] = set()
        for pattern, agents in self._topic_subscribers.items():
            if fnmatch.fnmatch(topic, pattern):
                matched.update(agents)
        return matched

    async def publish(self, message: Message) -> None:
        """Publish a message to all subscribers matching its topic."""
        # Record history
        self._history.append(message)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        subscribers = self._match_subscribers(message.topic)
        for agent_id in subscribers:
            if agent_id == message.sender:
                continue  # Don't send to self
            inbox = self._ensure_inbox(agent_id)
            try:
                inbox.put_nowait(message)
            except asyncio.QueueFull:
                # Drop oldest message to make room
                try:
                    inbox.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                inbox.put_nowait(message)

        # Fire registered handlers
        for pattern, handlers in self._handlers.items():
            if fnmatch.fnmatch(message.topic, pattern):
                for handler in handlers:
                    try:
                        result = handler(message)
                        if asyncio.iscoroutine(result):
                            await result
                    except Exception as handler_exc:
                        # Handlers must not crash the bus, but record the error
                        self._last_handler_error = handler_exc

    async def send(self, agent_id: str, message: Message) -> None:
        """Send a message directly to a specific agent's inbox."""
        self._history.append(message)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        inbox = self._ensure_inbox(agent_id)
        try:
            inbox.put_nowait(message)
        except asyncio.QueueFull:
            try:
                inbox.get_nowait()
            except asyncio.QueueEmpty:
                pass
            inbox.put_nowait(message)

    async def broadcast(self, message: Message) -> None:
        """Send a message to ALL agents (regardless of subscription)."""
        self._history.append(message)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]

        for agent_id, inbox in self._inboxes.items():
            if agent_id == message.sender:
                continue
            try:
                inbox.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    inbox.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                inbox.put_nowait(message)

    def _is_expired(self, message: Message) -> bool:
        """Check if a message has exceeded its TTL."""
        if message.ttl is None:
            return False
        try:
            sent = datetime.fromisoformat(message.timestamp)
            elapsed = (datetime.now() - sent).total_seconds()
            return elapsed > message.ttl
        except (ValueError, TypeError):
            return False

    async def receive(
        self,
        agent_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Message]:
        """
        Receive the next message from an agent's inbox.

        Returns None on timeout (if timeout is set).
        Blocks indefinitely if timeout is None.
        Expired messages (past TTL) are silently discarded.
        """
        inbox = self._ensure_inbox(agent_id)
        try:
            if timeout is not None:
                msg = await asyncio.wait_for(inbox.get(), timeout=timeout)
            else:
                msg = await inbox.get()
            # Discard expired messages and try again non-blocking
            if self._is_expired(msg):
                return await self.receive_nowait(agent_id)
            return msg
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

    async def receive_nowait(self, agent_id: str) -> Optional[Message]:
        """Non-blocking receive. Returns None if inbox is empty."""
        inbox = self._ensure_inbox(agent_id)
        try:
            return inbox.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def on(self, topic_pattern: str, handler: Callable) -> None:
        """Register a callback for messages matching a topic pattern."""
        self._handlers[topic_pattern].append(handler)

    def get_history(
        self,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> List[Message]:
        """Get recent message history, optionally filtered by topic pattern."""
        if topic is None:
            return self._history[-limit:]
        return [
            m for m in self._history
            if fnmatch.fnmatch(m.topic, topic)
        ][-limit:]

    def pending_count(self, agent_id: str) -> int:
        """Number of messages waiting in an agent's inbox."""
        inbox = self._inboxes.get(agent_id)
        return inbox.qsize() if inbox else 0

    @property
    def agent_ids(self) -> List[str]:
        """All registered agent IDs."""
        return list(self._inboxes.keys())

    def clear(self) -> None:
        """Clear all inboxes, subscriptions, and history."""
        self._inboxes.clear()
        self._subscriptions.clear()
        self._topic_subscribers.clear()
        self._history.clear()
        self._handlers.clear()


# ─────────────────────────────────────────────────────────
# Standard Topics
# ─────────────────────────────────────────────────────────

class Topics:
    """Well-known topic constants for inter-agent communication."""

    # Task lifecycle
    TASK_ASSIGN = "task.assign"
    TASK_ACCEPT = "task.accept"
    TASK_PROGRESS = "task.progress"
    TASK_COMPLETE = "task.complete"
    TASK_FAILED = "task.failed"

    # Saga lifecycle
    SAGA_START = "saga.start"
    SAGA_CHECKPOINT = "saga.checkpoint"
    SAGA_COMPENSATE = "saga.compensate"
    SAGA_COMPLETE = "saga.complete"
    SAGA_FAILED = "saga.failed"

    # Quality & verification
    QUALITY_REQUEST = "quality.request"
    QUALITY_RESULT = "quality.result"
    HALLUCINATION_ALERT = "quality.hallucination"

    # Agent lifecycle
    AGENT_READY = "agent.ready"
    AGENT_BUSY = "agent.busy"
    AGENT_ERROR = "agent.error"

    # State changes
    STATE_UPDATED = "state.updated"
    STATE_CONSOLIDATED = "state.consolidated"
