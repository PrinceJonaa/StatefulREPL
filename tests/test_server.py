"""Tests for server endpoints — Phase 2.

Tests use FastAPI's TestClient (httpx under the hood) so no real
server process is needed.
"""

import json
import os
import tempfile

import pytest

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(
    not HAS_FASTAPI,
    reason="FastAPI not installed — install with: pip install stateful-repl[server]",
)


@pytest.fixture(autouse=True)
def clean_server_state(tmp_path, monkeypatch):
    """Give each test a fresh state file and event store."""
    state_file = str(tmp_path / "test_state.md")
    event_db = str(tmp_path / "test_events.db")
    monkeypatch.setenv("LOOM_STATE_FILE", state_file)
    monkeypatch.setenv("LOOM_EVENT_BACKEND", "sqlite")
    monkeypatch.setenv("LOOM_EVENT_PATH", event_db)

    # Reset module-level state
    import stateful_repl.server as srv
    srv._repl = None
    srv._evaluator = None
    srv._event_store = None
    srv._sse_subscribers.clear()
    srv.STATE_FILE = state_file
    srv.EVENT_BACKEND = "sqlite"
    srv.EVENT_PATH = event_db
    yield


@pytest.fixture
def client():
    from stateful_repl.server import app
    return TestClient(app)


class TestHealth:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestStateCRUD:
    def test_read_empty(self, client):
        r = client.get("/state")
        assert r.status_code == 200
        data = r.json()
        assert "L1" in data

    def test_set_goal(self, client):
        r = client.post("/state/goal", json={"goal": "Build the Loom"})
        assert r.status_code == 200
        state = client.get("/state").json()
        assert state["L1"]["goal"] == "Build the Loom"

    def test_add_artifact(self, client):
        client.post("/state/goal", json={"goal": "test"})
        r = client.post("/state/artifact", json={"artifact": "agents.md"})
        assert r.status_code == 200

    def test_add_constraint(self, client):
        r = client.post("/state/constraint", json={"constraint": "No network"})
        assert r.status_code == 200

    def test_add_question(self, client):
        r = client.post("/state/question", json={"question": "Why?"})
        assert r.status_code == 200

    def test_add_log(self, client):
        r = client.post("/state/log", json={"content": "Started testing"})
        assert r.status_code == 200

    def test_add_rule(self, client):
        r = client.post("/state/rule", json={
            "rule_id": "R1", "when": "err", "then": "fix", "why": "reliability",
        })
        assert r.status_code == 200

    def test_add_scar(self, client):
        r = client.post("/state/scar", json={
            "scar": "crash", "boon": "learned", "newrule": "R1",
            "glyphstamp": "crash-fix",
        })
        assert r.status_code == 200


class TestConsolidation:
    def test_l1_to_l2(self, client):
        client.post("/state/goal", json={"goal": "test consolidation"})
        r = client.post("/state/consolidate/l1-to-l2")
        assert r.status_code == 200

    def test_l2_to_l3(self, client):
        client.post("/state/log", json={"content": "entry1"})
        r = client.post("/state/consolidate/l2-to-l3")
        assert r.status_code == 200


class TestQuality:
    def test_quality_endpoint(self, client):
        r = client.get("/quality")
        assert r.status_code == 200
        data = r.json()
        assert "aggregate_score" in data
        assert "dimensions" in data

    def test_quality_summary(self, client):
        r = client.get("/quality/summary")
        assert r.status_code == 200
        assert "summary" in r.json()


class TestValidation:
    def test_validate(self, client):
        r = client.get("/validate")
        assert r.status_code == 200
        assert "validation" in r.json()


class TestEvents:
    def test_list_events(self, client):
        r = client.get("/events")
        assert r.status_code == 200
        assert "events" in r.json()
