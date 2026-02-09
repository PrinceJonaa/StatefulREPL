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

from stateful_repl.loom_state import LoomREPL
from stateful_repl.quality import QualityEvaluator, QualityVector
from stateful_repl.events import create_event_store, SQLiteEventStore


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


# ─────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────

STATE_FILE = os.environ.get("LOOM_STATE_FILE", "loom_state.md")
EVENT_BACKEND = os.environ.get("LOOM_EVENT_BACKEND", "sqlite")
EVENT_PATH = os.environ.get("LOOM_EVENT_PATH", "events.db")

_repl: Optional[LoomREPL] = None
_evaluator: Optional[QualityEvaluator] = None
_event_store = None
_sse_subscribers: list = []


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


# ─────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load state on startup, save on shutdown."""
    _get_repl()
    _get_event_store()
    yield
    if _repl:
        _repl.save_state(STATE_FILE)


app = FastAPI(
    title="StatefulREPL — The Loom",
    description="Stateful AI memory & orchestration API",
    version="0.2.0",
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
        version="0.2.0",
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
    _save_and_broadcast(repl, "goal_updated", {"goal": body.goal})
    return {"ok": True, "goal": body.goal}


@app.post("/state/artifact")
def add_artifact(body: ArtifactAdd):
    repl = _get_repl()
    repl.update_l1("artifacts", body.artifact)
    _save_and_broadcast(repl, "artifact_added", {"artifact": body.artifact})
    return {"ok": True}


@app.post("/state/constraint")
def add_constraint(body: ConstraintAdd):
    repl = _get_repl()
    repl.update_l1("constraints", body.constraint)
    _save_and_broadcast(repl, "constraint_added", {"constraint": body.constraint})
    return {"ok": True}


@app.post("/state/question")
def add_question(body: QuestionAdd):
    repl = _get_repl()
    repl.update_l1("open_questions", body.question)
    _save_and_broadcast(repl, "question_added", {"question": body.question})
    return {"ok": True}


@app.post("/state/log")
def add_log(body: LogEntry):
    repl = _get_repl()
    repl.append("L2", body.content)
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
    _save_and_broadcast(repl, "scar_added", {"scar": body.scar})
    return {"ok": True}


# ─────────────────────────────────────────────────────────
# Consolidation
# ─────────────────────────────────────────────────────────

@app.post("/state/consolidate/l1-to-l2")
def consolidate_l1_l2():
    repl = _get_repl()
    repl.consolidate_l1_to_l2()
    _save_and_broadcast(repl, "consolidated", {"from": "L1", "to": "L2"})
    return {"ok": True, "from": "L1", "to": "L2"}


@app.post("/state/consolidate/l2-to-l3")
def consolidate_l2_l3():
    repl = _get_repl()
    repl.consolidate_l2_to_l3()
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
