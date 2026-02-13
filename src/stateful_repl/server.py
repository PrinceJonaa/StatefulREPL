"""
Web API & SSE Server — Phase 2.

FastAPI application providing:
  - REST endpoints for Loom state operations
  - SSE (Server-Sent Events) stream for real-time quality vector changes
  - Quality vector visualization endpoint (JSON for radar chart)
  - Event history with time-travel replay

Run:
    uvicorn stateful_repl.server:app --reload

Or via CLI:
    loom-repl serve --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise ImportError(
        "FastAPI dependencies not installed. "
        "Install with: pip install stateful-repl[server]"
    ) from exc

from stateful_repl.loom_state import LoomREPL, StateEvent
from stateful_repl.quality import QualityEvaluator, QualityVector
from stateful_repl.events import create_event_store, SQLiteEventStore
from stateful_repl.compression import ExtractiveCompressor
from stateful_repl.prefetch import PredictivePrefetchEngine
from stateful_repl.calibration import CalibrationLearner, CalibrationSample
from stateful_repl.loom_writeback import LoomWriteback, WritebackPacket


# ─────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────

class GoalUpdate(BaseModel):
    goal: str


class ArtifactAdd(BaseModel):
    artifact: str


class ConstraintAdd(BaseModel):
    constraint: str


class QuestionAdd(BaseModel):
    question: str


class LogEntry(BaseModel):
    content: str
    phase: Optional[str] = None


class RuleAdd(BaseModel):
    rule_id: str
    when: str
    then: str
    why: str


class ScarAdd(BaseModel):
    scar: str
    boon: str
    newrule: str = ""
    glyphstamp: str = ""
    userfeedback: str = ""


class StateResponse(BaseModel):
    L1: Dict[str, Any]
    L2: list
    L3: Dict[str, Any]


class QualityResponse(BaseModel):
    aggregate_score: float
    aggregate_confidence: float
    timestamp: str
    dimensions: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    version: str
    state_file: str
    event_count: int


class CompressionRequest(BaseModel):
    text: Optional[str] = ""
    target_ratio: float = Field(default=0.3, ge=0.05, le=1.0)
    use_state: bool = False


class PrefetchRecordRequest(BaseModel):
    key: str = Field(min_length=1)


class PrefetchPredictRequest(BaseModel):
    current_keys: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class CalibrationSampleRequest(BaseModel):
    predicted: float = Field(ge=0.0, le=1.0)
    observed: int = Field(ge=0, le=1)


class CalibrationFitRequest(BaseModel):
    samples: list[CalibrationSampleRequest]


class CompressionMetricsRequest(BaseModel):
    text: str
    target_ratio: float = Field(default=0.3, ge=0.05, le=1.0)
    required_terms: list[str] = Field(default_factory=list)


class PrefetchMetricsRequest(BaseModel):
    trace: list[str] = Field(default_factory=list)
    k: int = Field(default=3, ge=1, le=20)
    warmup: int = Field(default=2, ge=1, le=50)


class CalibrationMetricsRequest(BaseModel):
    samples: list[CalibrationSampleRequest]
    holdout_ratio: float = Field(default=0.3, ge=0.1, le=0.5)


class ReplayResponse(BaseModel):
    event_count: int
    up_to: Optional[int]
    state: Dict[str, Any]


class LoomWritebackRequest(BaseModel):
    artifact_name: str
    artifact_type: str
    artifact_path: str
    claim_statement: str
    claim_scope: str = "module"
    claim_confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    claim_falsifies: str
    oracle_name: str
    oracle_method: str = "test"
    oracle_command: str
    oracle_expected: str
    scar: str = ""
    boon: str = ""
    newrule: str = ""
    glyphstamp: str = ""
    owner: str = "runtime"


# ─────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────

STATE_FILE = os.environ.get("LOOM_STATE_FILE", "loom_state.md")
EVENT_BACKEND = os.environ.get("LOOM_EVENT_BACKEND", "sqlite")
EVENT_PATH = os.environ.get("LOOM_EVENT_PATH", "events.db")
WORKSPACE_ROOT = os.environ.get("LOOM_WORKSPACE_ROOT", str(Path.cwd()))

_repl: Optional[LoomREPL] = None
_evaluator: Optional[QualityEvaluator] = None
_event_store = None
_sse_subscribers: list = []
_compressor: Optional[ExtractiveCompressor] = None
_prefetcher: Optional[PredictivePrefetchEngine] = None
_calibration: Optional[CalibrationLearner] = None
_loom_writeback: Optional[LoomWriteback] = None


def _get_repl() -> LoomREPL:
    global _repl
    if _repl is None:
        _repl = LoomREPL(state_file=STATE_FILE)
    return _repl


def _get_evaluator() -> QualityEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = QualityEvaluator()
    return _evaluator


def _get_event_store():
    global _event_store
    if _event_store is None:
        _event_store = create_event_store(backend=EVENT_BACKEND, path=EVENT_PATH)
    return _event_store


def _get_compressor() -> ExtractiveCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ExtractiveCompressor()
    return _compressor


def _get_prefetcher() -> PredictivePrefetchEngine:
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = PredictivePrefetchEngine()
    return _prefetcher


def _get_calibration() -> CalibrationLearner:
    global _calibration
    if _calibration is None:
        _calibration = CalibrationLearner()
    return _calibration


def _get_loom_writeback() -> LoomWriteback:
    global _loom_writeback
    if _loom_writeback is None:
        _loom_writeback = LoomWriteback(workspace_root=WORKSPACE_ROOT)
    return _loom_writeback


async def _broadcast_sse(event_type: str, data: dict) -> None:
    """Push event to all SSE subscribers."""
    payload = json.dumps({"type": event_type, "data": data, "timestamp": datetime.now().isoformat()})
    dead: list = []
    for q in _sse_subscribers:
        try:
            await q.put(payload)
        except Exception:
            dead.append(q)
    for d in dead:
        _sse_subscribers.remove(d)


def _save_and_broadcast(repl: LoomREPL, event_type: str, data: dict) -> None:
    """Save state, then schedule SSE broadcast."""
    repl.save_state()
    # Fire-and-forget broadcast
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast_sse(event_type, data))
    except RuntimeError:
        pass


def _record_event(layer: str, operation: str, data: dict[str, Any]) -> None:
    """Persist one event into configured event store."""
    store = _get_event_store()
    store.emit(
        StateEvent(
            timestamp=datetime.now().isoformat(),
            layer=layer,
            operation=operation,
            data=data,
        )
    )


# ─────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load state on startup, save on shutdown."""
    _get_repl()
    _get_event_store()
    _get_compressor()
    _get_prefetcher()
    _get_calibration()
    _get_loom_writeback()
    yield
    if _repl:
        _repl.save_state()


app = FastAPI(
    title="StatefulREPL — The Loom",
    description="Stateful AI memory & orchestration API",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    store = _get_event_store()
    return HealthResponse(
        status="ok",
        version="0.4.0",
        state_file=STATE_FILE,
        event_count=store.count(),
    )


# ─────────────────────────────────────────────────────────
# State CRUD
# ─────────────────────────────────────────────────────────

@app.get("/state", response_model=StateResponse)
def read_state(layer: Optional[str] = Query(None, pattern="^(L1|L2|L3|ALL)$")):
    repl = _get_repl()
    state = repl.read_state(layer or "ALL")
    return StateResponse(
        L1=state.get("L1", {}),
        L2=state.get("L2", []),
        L3=state.get("L3", {}),
    )


@app.post("/state/goal")
def update_goal(body: GoalUpdate):
    repl = _get_repl()
    repl.update_l1("goal", body.goal)
    _record_event("L1", "update.goal", {"new": body.goal})
    _get_prefetcher().record_access("goal")
    _save_and_broadcast(repl, "goal_updated", {"goal": body.goal})
    return {"ok": True, "goal": body.goal}


@app.post("/state/artifact")
def add_artifact(body: ArtifactAdd):
    repl = _get_repl()
    repl.update_l1("artifacts", body.artifact)
    _record_event("L1", "update.artifacts", {"new": body.artifact})
    _get_prefetcher().record_access(str(body.artifact))
    _save_and_broadcast(repl, "artifact_added", {"artifact": body.artifact})
    return {"ok": True}


@app.post("/state/constraint")
def add_constraint(body: ConstraintAdd):
    repl = _get_repl()
    repl.update_l1("constraints", body.constraint)
    _record_event("L1", "update.constraints", {"new": body.constraint})
    _get_prefetcher().record_access("constraints")
    _save_and_broadcast(repl, "constraint_added", {"constraint": body.constraint})
    return {"ok": True}


@app.post("/state/question")
def add_question(body: QuestionAdd):
    repl = _get_repl()
    repl.update_l1("open_questions", body.question)
    _record_event("L1", "update.open_questions", {"new": body.question})
    _get_prefetcher().record_access("open_questions")
    _save_and_broadcast(repl, "question_added", {"question": body.question})
    return {"ok": True}


@app.post("/state/log")
def add_log(body: LogEntry):
    repl = _get_repl()
    repl.append("L2", body.content)
    _record_event("L2", "append", {"content": body.content})
    _get_prefetcher().record_access("L2")
    _save_and_broadcast(repl, "log_added", {"content": body.content})
    return {"ok": True}


@app.post("/state/rule")
def add_rule(body: RuleAdd):
    repl = _get_repl()
    repl.append("L3", {
        "type": "rule",
        "rule_id": body.rule_id,
        "when": body.when,
        "then": body.then,
        "why": body.why,
    })
    _record_event("L3", "append", {
        "rule_id": body.rule_id,
        "when": body.when,
        "then": body.then,
        "why": body.why,
    })
    _save_and_broadcast(repl, "rule_added", {"rule_id": body.rule_id})
    return {"ok": True}


@app.post("/state/scar")
def add_scar(body: ScarAdd):
    repl = _get_repl()
    repl.append("L3", {
        "type": "tracewisdomlog",
        "scar": body.scar,
        "boon": body.boon,
        "newrule": body.newrule,
        "glyphstamp": body.glyphstamp,
        "userfeedback": body.userfeedback,
    })
    _record_event("L3", "append", {
        "scar": body.scar,
        "boon": body.boon,
        "newrule": body.newrule,
        "glyphstamp": body.glyphstamp,
        "userfeedback": body.userfeedback,
    })
    _save_and_broadcast(repl, "scar_added", {"scar": body.scar})
    return {"ok": True}


# ─────────────────────────────────────────────────────────
# Consolidation
# ─────────────────────────────────────────────────────────

@app.post("/state/consolidate/l1-to-l2")
def consolidate_l1_l2():
    repl = _get_repl()
    repl.consolidate_l1_to_l2()
    _record_event("L2", "consolidate_from_l1", {"from": "L1", "to": "L2"})
    _save_and_broadcast(repl, "consolidated", {"from": "L1", "to": "L2"})
    return {"ok": True, "from": "L1", "to": "L2"}


@app.post("/state/consolidate/l2-to-l3")
def consolidate_l2_l3():
    repl = _get_repl()
    repl.consolidate_l2_to_l3()
    _record_event("L3", "consolidate_from_l2", {"from": "L2", "to": "L3"})
    _save_and_broadcast(repl, "consolidated", {"from": "L2", "to": "L3"})
    return {"ok": True, "from": "L2", "to": "L3"}


# ─────────────────────────────────────────────────────────
# Quality
# ─────────────────────────────────────────────────────────

@app.get("/quality", response_model=QualityResponse)
def get_quality():
    repl = _get_repl()
    evaluator = _get_evaluator()
    state = repl.read_state("ALL")
    vector = evaluator.evaluate(state)
    return QualityResponse(**vector.to_dict())


@app.get("/quality/summary")
def get_quality_summary():
    repl = _get_repl()
    evaluator = _get_evaluator()
    state = repl.read_state("ALL")
    vector = evaluator.evaluate(state)
    return {"summary": vector.summary(), **vector.to_dict()}


# ─────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────

@app.get("/validate")
def validate_state():
    repl = _get_repl()
    result = repl.validate_state()
    return {"validation": result}


# ─────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────

@app.get("/events")
def list_events(
    layer: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    store = _get_event_store()
    events = store.get_events(layer=layer, since=since)
    return {"events": events[-limit:], "total": len(events)}


def _replay_reducer(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer used by time-travel replay endpoint."""
    layer = event.get("layer", "")
    operation = event.get("operation", "")
    data = event.get("data", {})

    if layer == "L1" and operation.startswith("update."):
        field = operation.split("update.", 1)[1]
        state.setdefault("L1", {})[field] = data.get("new")
    elif layer == "L1" and operation == "clear":
        state["L1"] = {
            "goal": "",
            "constraints": [],
            "artifacts": [],
            "open_questions": [],
        }
    elif layer == "L2" and operation in {"append", "consolidate_from_l1"}:
        state.setdefault("L2", []).append(data)
    elif layer == "L3" and operation == "append":
        if "rule_id" in data:
            state.setdefault("L3", {}).setdefault("rules", []).append(data)
        elif "concept" in data:
            state.setdefault("L3", {}).setdefault("concepts", []).append(data)
        elif "scar" in data or "boon" in data:
            state.setdefault("L3", {}).setdefault("tracewisdomlog", []).append(data)

    return state


@app.get("/events/replay", response_model=ReplayResponse)
def replay_events(up_to: Optional[int] = Query(None, ge=1)):
    """Time-travel replay to reconstruct state up to event N."""
    store = _get_event_store()
    initial = {
        "L1": {
            "goal": "",
            "constraints": [],
            "artifacts": [],
            "open_questions": [],
        },
        "L2": [],
        "L3": {
            "rules": [],
            "concepts": [],
            "tracewisdomlog": [],
        },
    }
    replayed = store.replay(_replay_reducer, initial=initial, up_to=up_to)
    return ReplayResponse(event_count=store.count(), up_to=up_to, state=replayed)


# ─────────────────────────────────────────────────────────
# SSE — real-time stream
# ─────────────────────────────────────────────────────────

@app.get("/stream")
async def sse_stream():
    """
    Server-Sent Events stream.

    Clients receive real-time updates whenever state changes:
      - goal_updated, artifact_added, log_added, etc.
      - quality vector after each mutation

    Connect with: EventSource('/stream')
    """
    queue: asyncio.Queue = asyncio.Queue()
    _sse_subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Send initial quality snapshot
            repl = _get_repl()
            evaluator = _get_evaluator()
            state = repl.read_state("ALL")
            vector = evaluator.evaluate(state)
            yield f"data: {json.dumps({'type': 'initial_quality', 'data': vector.to_dict()})}\n\n"

            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield f": keepalive\n\n"
        finally:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────────────────
# Phase 3 — Agents & Orchestration
# ─────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    goal: str
    context: Optional[str] = None


class TaskAssignRequest(BaseModel):
    task_id: str
    name: str
    role: str = "builder"
    description: str = ""


@app.get("/agents/status")
def agents_status():
    """Phase 3 agent system overview."""
    return {
        "version": "0.3.0",
        "phase": 3,
        "modules": ["message_bus", "orchestrator", "planner", "agents", "router"],
        "roles": ["coordinator", "builder", "verifier", "distiller"],
    }


@app.post("/agents/plan")
def create_plan(body: PlanRequest):
    """Create a task plan by decomposing a goal."""
    from stateful_repl.planner import TaskPlanner, TaskNode
    planner = TaskPlanner()
    plan = planner.create_plan(body.goal)
    plan.add_task(TaskNode(id="task-1", name=body.goal, role="builder"))
    tiers = planner.schedule(plan)
    return {
        "plan_id": plan.plan_id,
        "goal": plan.goal,
        "task_count": plan.task_count,
        "tiers": tiers,
        "tasks": {tid: t.to_dict() for tid, t in plan.tasks.items()},
    }


@app.get("/agents/router")
def router_summary():
    """Get task router summary."""
    from stateful_repl.router import TaskRouter
    router = TaskRouter()
    return router.summary()


# ─────────────────────────────────────────────────────────
# Phase 4 — Compression, Prefetch, Calibration, Time-travel
# ─────────────────────────────────────────────────────────

@app.get("/phase4/status")
def phase4_status():
    """Phase 4 feature availability overview."""
    return {
        "version": "0.4.0",
        "phase": 4,
        "features": {
            "context_compression": True,
            "predictive_prefetch": True,
            "calibration_learning": True,
            "time_travel_replay": True,
        },
    }


@app.post("/context/compress")
def compress_context(body: CompressionRequest):
    """Compress provided text or current state into compact context."""
    compressor = _get_compressor()
    if body.use_state:
        result = compressor.compress_state(_get_repl().read_state("ALL"), target_ratio=body.target_ratio)
    else:
        text = body.text or ""
        result = compressor.compress_text(text, target_ratio=body.target_ratio)
    return result.to_dict()


@app.post("/prefetch/record")
def prefetch_record(body: PrefetchRecordRequest):
    """Record one access event for prefetch learning."""
    _get_prefetcher().record_access(body.key)
    return {"ok": True, "summary": _get_prefetcher().summary()}


@app.post("/prefetch/predict")
def prefetch_predict(body: PrefetchPredictRequest):
    """Predict likely next context keys/artifacts."""
    candidates = _get_prefetcher().predict_next(body.current_keys, limit=body.limit)
    return {
        "candidates": [c.to_dict() for c in candidates],
        "summary": _get_prefetcher().summary(),
    }


@app.get("/calibration")
def calibration_summary():
    """Get current calibration parameters."""
    return _get_calibration().summary()


@app.post("/calibration/fit")
def calibration_fit(body: CalibrationFitRequest):
    """Fit calibration model from prediction/outcome samples."""
    learner = _get_calibration()
    samples = [CalibrationSample(predicted=s.predicted, observed=s.observed) for s in body.samples]
    report = learner.fit(samples)
    return report.to_dict()


@app.post("/phase4/metrics/compression")
def compression_metrics(body: CompressionMetricsRequest):
    """Measure compression ratio and retention quality."""
    compressor = _get_compressor()
    result = compressor.compress_text(body.text, target_ratio=body.target_ratio)
    quality = compressor.evaluate_retention(
        original_text=body.text,
        compressed_text=result.compressed_text,
        required_terms=body.required_terms,
    )
    return {
        "compression": result.to_dict(),
        "quality": quality.to_dict(),
    }


@app.post("/phase4/metrics/prefetch")
def prefetch_metrics(body: PrefetchMetricsRequest):
    """Measure prefetch hit-rate@k and MRR on a trace."""
    quality = _get_prefetcher().evaluate_trace(
        trace=body.trace,
        k=body.k,
        warmup=body.warmup,
    )
    return quality.to_dict()


@app.post("/phase4/metrics/calibration")
def calibration_metrics(body: CalibrationMetricsRequest):
    """Measure calibration quality on holdout split."""
    learner = _get_calibration()
    samples = [CalibrationSample(predicted=s.predicted, observed=s.observed) for s in body.samples]
    quality = learner.evaluate_holdout(samples, holdout_ratio=body.holdout_ratio)
    return quality.to_dict()


@app.post("/loom/writeback")
def loom_writeback(body: LoomWritebackRequest):
    """Append ART/CLAIM/ORACLE and optional SCAR entries at runtime."""
    packet = WritebackPacket(
        artifact_name=body.artifact_name,
        artifact_type=body.artifact_type,
        artifact_path=body.artifact_path,
        claim_statement=body.claim_statement,
        claim_scope=body.claim_scope,
        claim_confidence=body.claim_confidence,
        claim_falsifies=body.claim_falsifies,
        oracle_name=body.oracle_name,
        oracle_method=body.oracle_method,
        oracle_command=body.oracle_command,
        oracle_expected=body.oracle_expected,
        scar=body.scar,
        boon=body.boon,
        newrule=body.newrule,
        glyphstamp=body.glyphstamp,
    )
    ids = _get_loom_writeback().append_minimum(packet=packet, owner=body.owner)
    return {"ok": True, "ids": ids}
