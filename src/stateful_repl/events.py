"""
Event Sourcing Store — Phase 2.

Provides two backends:

  1. InMemoryEventStore   — JSON file-backed (Phase 1 compat, tests, CLI)
  2. SQLiteEventStore     — SQLite-backed (Phase 2 production)

Both conform to the EventStore protocol so callers are backend-agnostic.

CQRS: writes append to the event log; reads go through ``get_events()``
or ``replay()`` (time-travel debugging).

Production upgrade path: swap SQLite for PostgreSQL + JSONB by
implementing the same EventStore interface.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol

from stateful_repl.loom_state import StateEvent


# ─────────────────────────────────────────────────────────
# Protocol (interface)
# ─────────────────────────────────────────────────────────

class EventStore(Protocol):
    """Common interface for all event stores."""

    def emit(self, event: StateEvent) -> None: ...
    def get_events(
        self, layer: Optional[str] = None, since: Optional[str] = None
    ) -> List[dict]: ...
    def replay(
        self,
        reducer: Callable[[Dict[str, Any], dict], Dict[str, Any]],
        initial: Dict[str, Any],
        up_to: Optional[int] = None,
    ) -> Dict[str, Any]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


# ─────────────────────────────────────────────────────────
# Backend 1: In-Memory (JSON file)
# ─────────────────────────────────────────────────────────

class InMemoryEventStore:
    """
    Minimal event store backed by a JSON file.

    Good for: tests, CLI, small projects.
    """

    def __init__(self, filepath: str = "events.json"):
        self.filepath = Path(filepath)
        self.events: List[StateEvent] = []
        if self.filepath.exists():
            self._load()

    def emit(self, event: StateEvent) -> None:
        self.events.append(event)
        self._save()

    def get_events(
        self,
        layer: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[dict]:
        result = [e.to_dict() for e in self.events]
        if layer:
            result = [e for e in result if e["layer"] == layer]
        if since:
            result = [e for e in result if e["timestamp"] >= since]
        return result

    def replay(
        self,
        reducer: Callable[[Dict[str, Any], dict], Dict[str, Any]],
        initial: Dict[str, Any],
        up_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        events = self.events[:up_to] if up_to else self.events
        state = dict(initial)
        for event in events:
            state = reducer(state, event.to_dict())
        return state

    def count(self) -> int:
        return len(self.events)

    def clear(self) -> None:
        self.events.clear()
        if self.filepath.exists():
            self.filepath.unlink()

    # ── persistence ──────────────────────────────────────

    def _save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self.events], f, indent=2, default=str
            )

    def _load(self) -> None:
        with open(self.filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.events = [
            StateEvent(
                timestamp=e["timestamp"],
                layer=e["layer"],
                operation=e["operation"],
                data=e.get("data", {}),
            )
            for e in raw
        ]


# ─────────────────────────────────────────────────────────
# Backend 2: SQLite
# ─────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT    NOT NULL,
    layer     TEXT    NOT NULL,
    operation TEXT    NOT NULL,
    data      TEXT    NOT NULL DEFAULT '{}'
);
"""

_CREATE_INDEX_LAYER = """
CREATE INDEX IF NOT EXISTS idx_events_layer ON events(layer);
"""

_CREATE_INDEX_TS = """
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
"""


class SQLiteEventStore:
    """
    Production event store backed by SQLite.

    Features:
      - ACID transactions
      - Indexed by layer and timestamp
      - WAL mode for concurrent reads
      - Time-travel replay via ``up_to`` parameter
    """

    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            conn.execute(_CREATE_INDEX_LAYER)
            conn.execute(_CREATE_INDEX_TS)
            conn.execute("PRAGMA journal_mode=WAL;")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def emit(self, event: StateEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO events (timestamp, layer, operation, data) VALUES (?, ?, ?, ?)",
                (event.timestamp, event.layer, event.operation, json.dumps(event.data)),
            )

    def get_events(
        self,
        layer: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[dict]:
        sql = "SELECT timestamp, layer, operation, data FROM events WHERE 1=1"
        params: list = []
        if layer:
            sql += " AND layer = ?"
            params.append(layer)
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY id ASC"

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return [
            {
                "timestamp": row["timestamp"],
                "layer": row["layer"],
                "operation": row["operation"],
                "data": json.loads(row["data"]),
            }
            for row in rows
        ]

    def replay(
        self,
        reducer: Callable[[Dict[str, Any], dict], Dict[str, Any]],
        initial: Dict[str, Any],
        up_to: Optional[int] = None,
    ) -> Dict[str, Any]:
        sql = "SELECT timestamp, layer, operation, data FROM events ORDER BY id ASC"
        if up_to is not None:
            sql += f" LIMIT {int(up_to)}"

        with self._conn() as conn:
            rows = conn.execute(sql).fetchall()

        state = dict(initial)
        for row in rows:
            event_dict = {
                "timestamp": row["timestamp"],
                "layer": row["layer"],
                "operation": row["operation"],
                "data": json.loads(row["data"]),
            }
            state = reducer(state, event_dict)
        return state

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM events").fetchone()
            return row["c"]

    def clear(self) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM events")

    def get_event_by_id(self, event_id: int) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT timestamp, layer, operation, data FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row:
                return {
                    "timestamp": row["timestamp"],
                    "layer": row["layer"],
                    "operation": row["operation"],
                    "data": json.loads(row["data"]),
                }
        return None


# ─────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────

def create_event_store(
    backend: str = "json",
    path: Optional[str] = None,
) -> EventStore:
    """
    Factory to create the appropriate event store backend.

    Args:
        backend: "json" or "sqlite"
        path: File path for the store. Defaults per backend.
    """
    if backend == "sqlite":
        return SQLiteEventStore(db_path=path or "events.db")  # type: ignore[return-value]
    return InMemoryEventStore(filepath=path or "events.json")  # type: ignore[return-value]
