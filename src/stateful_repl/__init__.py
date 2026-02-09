"""
StatefulREPL — The Loom's stateful memory engine.
"""

from stateful_repl.loom_state import LoomREPL
from stateful_repl.sandbox import LoomSandbox
from stateful_repl.quality import QualityEvaluator, QualityVector
from stateful_repl.hallucination import HallucinationDetector, HallucinationScore
from stateful_repl.events import InMemoryEventStore, SQLiteEventStore, create_event_store
from stateful_repl.models import create_adapter, CompletionResponse

__all__ = [
    "LoomREPL",
    "LoomSandbox",
    "QualityEvaluator",
    "QualityVector",
    "HallucinationDetector",
    "HallucinationScore",
    "InMemoryEventStore",
    "SQLiteEventStore",
    "create_event_store",
    "create_adapter",
    "CompletionResponse",
]
__version__ = "0.2.0"
