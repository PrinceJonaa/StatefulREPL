"""
StatefulREPL — The Loom's stateful memory engine.
"""

from stateful_repl.loom_state import LoomREPL
from stateful_repl.sandbox import LoomSandbox
from stateful_repl.quality import QualityEvaluator, QualityVector
from stateful_repl.hallucination import HallucinationDetector, HallucinationScore
from stateful_repl.events import InMemoryEventStore, SQLiteEventStore, create_event_store
from stateful_repl.models import create_adapter, CompletionResponse
from stateful_repl.message_bus import InProcessBus, Message, MessageBus, Topics
from stateful_repl.orchestrator import (
    SagaTransaction,
    AsyncSagaManager,
    SagaStepDef,
    SagaStatus,
    RetryPolicy,
)
from stateful_repl.planner import TaskPlanner, TaskPlan, TaskNode, TaskStatus, TaskPriority
from stateful_repl.agents import (
    LoomAgent,
    CoordinatorLoom,
    BuilderLoom,
    VerifierLoom,
    DistillerLoom,
    AgentCapability,
    TaskResult,
)
from stateful_repl.router import TaskRouter, RoutingStrategy
from stateful_repl.compression import ExtractiveCompressor, CompressionResult
from stateful_repl.prefetch import PredictivePrefetchEngine, PrefetchCandidate
from stateful_repl.calibration import CalibrationLearner, CalibrationSample, CalibrationReport

__all__ = [
    # Phase 1
    "LoomREPL",
    "LoomSandbox",
    # Phase 2
    "QualityEvaluator",
    "QualityVector",
    "HallucinationDetector",
    "HallucinationScore",
    "InMemoryEventStore",
    "SQLiteEventStore",
    "create_event_store",
    "create_adapter",
    "CompletionResponse",
    # Phase 3 — Message Bus
    "InProcessBus",
    "Message",
    "MessageBus",
    "Topics",
    # Phase 3 — Orchestration
    "SagaTransaction",
    "AsyncSagaManager",
    "SagaStepDef",
    "SagaStatus",
    "RetryPolicy",
    # Phase 3 — Planner
    "TaskPlanner",
    "TaskPlan",
    "TaskNode",
    "TaskStatus",
    "TaskPriority",
    # Phase 3 — Agents
    "LoomAgent",
    "CoordinatorLoom",
    "BuilderLoom",
    "VerifierLoom",
    "DistillerLoom",
    "AgentCapability",
    "TaskResult",
    # Phase 3 — Router
    "TaskRouter",
    "RoutingStrategy",
    # Phase 4 — Compression
    "ExtractiveCompressor",
    "CompressionResult",
    # Phase 4 — Prefetch
    "PredictivePrefetchEngine",
    "PrefetchCandidate",
    # Phase 4 — Calibration
    "CalibrationLearner",
    "CalibrationSample",
    "CalibrationReport",
]
__version__ = "0.4.0"
