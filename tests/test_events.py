"""Tests for the event store backends — Phase 2."""

import json
import os
import tempfile

import pytest

from stateful_repl.events import (
    InMemoryEventStore,
    SQLiteEventStore,
    create_event_store,
)
from stateful_repl.loom_state import StateEvent


# ─── Helpers ─────────────────────────────────────────

def _event(layer="L1", op="update.goal", data=None):
    from datetime import datetime
    return StateEvent(
        timestamp=datetime.now().isoformat(),
        layer=layer,
        operation=op,
        data=data or {"goal": "test"},
    )


# ─── InMemoryEventStore ─────────────────────────────

class TestInMemoryEventStore:
    def test_emit_and_count(self, tmp_path):
        store = InMemoryEventStore(filepath=str(tmp_path / "ev.json"))
        store.emit(_event())
        assert store.count() == 1

    def test_get_events(self, tmp_path):
        store = InMemoryEventStore(filepath=str(tmp_path / "ev.json"))
        store.emit(_event(layer="L1"))
        store.emit(_event(layer="L2"))
        assert len(store.get_events()) == 2
        assert len(store.get_events(layer="L1")) == 1

    def test_replay(self, tmp_path):
        store = InMemoryEventStore(filepath=str(tmp_path / "ev.json"))
        store.emit(_event(layer="L1", data={"value": 1}))
        store.emit(_event(layer="L1", data={"value": 2}))

        def reducer(state, event):
            state["last"] = event["data"]["value"]
            return state

        result = store.replay(reducer, {})
        assert result["last"] == 2

    def test_replay_time_travel(self, tmp_path):
        store = InMemoryEventStore(filepath=str(tmp_path / "ev.json"))
        store.emit(_event(data={"v": 1}))
        store.emit(_event(data={"v": 2}))
        store.emit(_event(data={"v": 3}))

        def reducer(state, event):
            state["v"] = event["data"]["v"]
            return state

        result = store.replay(reducer, {}, up_to=2)
        assert result["v"] == 2

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "ev.json")
        store1 = InMemoryEventStore(filepath=path)
        store1.emit(_event())
        store1.emit(_event())

        store2 = InMemoryEventStore(filepath=path)
        assert store2.count() == 2

    def test_clear(self, tmp_path):
        store = InMemoryEventStore(filepath=str(tmp_path / "ev.json"))
        store.emit(_event())
        store.clear()
        assert store.count() == 0


# ─── SQLiteEventStore ───────────────────────────────

class TestSQLiteEventStore:
    def test_emit_and_count(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        store.emit(_event())
        assert store.count() == 1

    def test_get_events(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        store.emit(_event(layer="L1"))
        store.emit(_event(layer="L2"))
        store.emit(_event(layer="L3"))
        assert len(store.get_events()) == 3
        assert len(store.get_events(layer="L2")) == 1

    def test_get_events_since(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        e1 = StateEvent(timestamp="2024-01-01T00:00:00", layer="L1", operation="op", data={})
        e2 = StateEvent(timestamp="2024-06-01T00:00:00", layer="L1", operation="op", data={})
        store.emit(e1)
        store.emit(e2)
        assert len(store.get_events(since="2024-03-01")) == 1

    def test_replay(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        store.emit(_event(data={"n": 1}))
        store.emit(_event(data={"n": 2}))

        def reducer(state, event):
            state["sum"] = state.get("sum", 0) + event["data"]["n"]
            return state

        result = store.replay(reducer, {})
        assert result["sum"] == 3

    def test_replay_time_travel(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        for i in range(5):
            store.emit(_event(data={"i": i}))

        def reducer(state, event):
            state["last"] = event["data"]["i"]
            return state

        result = store.replay(reducer, {}, up_to=3)
        assert result["last"] == 2  # indices 0, 1, 2

    def test_clear(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        store.emit(_event())
        store.emit(_event())
        store.clear()
        assert store.count() == 0

    def test_get_event_by_id(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        store.emit(_event(data={"tag": "first"}))
        store.emit(_event(data={"tag": "second"}))
        e = store.get_event_by_id(1)
        assert e is not None
        assert e["data"]["tag"] == "first"

    def test_get_event_by_id_missing(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        assert store.get_event_by_id(999) is None

    def test_data_preserved(self, tmp_path):
        store = SQLiteEventStore(db_path=str(tmp_path / "ev.db"))
        nested = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
        store.emit(_event(data=nested))
        events = store.get_events()
        assert events[0]["data"] == nested


# ─── Factory ────────────────────────────────────────

class TestCreateEventStore:
    def test_json_backend(self, tmp_path):
        store = create_event_store("json", str(tmp_path / "ev.json"))
        assert isinstance(store, InMemoryEventStore)

    def test_sqlite_backend(self, tmp_path):
        store = create_event_store("sqlite", str(tmp_path / "ev.db"))
        assert isinstance(store, SQLiteEventStore)

    def test_default_is_json(self):
        store = create_event_store()
        assert isinstance(store, InMemoryEventStore)
