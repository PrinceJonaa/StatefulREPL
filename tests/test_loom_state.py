"""Tests for the core LoomREPL state engine."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stateful_repl.loom_state import LoomREPL


@pytest.fixture
def loom(tmp_path):
    """Create a LoomREPL with a temporary state file."""
    state_file = tmp_path / "test_state.md"
    return LoomREPL(str(state_file))


# ── L1 operations ────────────────────────────────────────


class TestL1:
    def test_update_goal(self, loom):
        loom.update_l1("goal", "test goal")
        assert loom.read_state("L1")["goal"] == "test goal"

    def test_update_constraints(self, loom):
        loom.update_l1("constraints", ["a", "b"])
        assert loom.read_state("L1")["constraints"] == ["a", "b"]

    def test_update_artifacts(self, loom):
        loom.update_l1("artifacts", ["doc.md"])
        assert loom.read_state("L1")["artifacts"] == ["doc.md"]

    def test_update_invalid_field_raises(self, loom):
        with pytest.raises(ValueError, match="Invalid L1 field"):
            loom.update_l1("nonexistent", "value")

    def test_clear_l1(self, loom):
        loom.update_l1("goal", "something")
        loom.clear_l1()
        assert loom.read_state("L1")["goal"] == ""
        assert loom.read_state("L1")["constraints"] == []


# ── L2 operations ────────────────────────────────────────


class TestL2:
    def test_append_l2(self, loom):
        loom.append("L2", "first entry")
        entries = loom.read_state("L2")["L2"]
        assert len(entries) == 1
        assert entries[0]["content"] == "first entry"

    def test_append_l2_multiple(self, loom):
        loom.append("L2", "entry 1")
        loom.append("L2", "entry 2")
        entries = loom.read_state("L2")["L2"]
        assert len(entries) == 2

    def test_append_l1_raises(self, loom):
        with pytest.raises(ValueError, match="requires UPDATE"):
            loom.append("L1", "bad")

    def test_append_invalid_layer_raises(self, loom):
        with pytest.raises(ValueError, match="Invalid layer"):
            loom.append("L4", "bad")


# ── L3 operations ────────────────────────────────────────


class TestL3:
    def test_append_rule(self, loom):
        loom.append("L3", {
            "rule_id": "test-rule",
            "when": "testing",
            "then": "pass",
            "why": "verification",
        })
        rules = loom.read_state("L3")["rules"]
        assert len(rules) == 1
        assert rules[0]["rule_id"] == "test-rule"

    def test_append_concept(self, loom):
        loom.append("L3", {
            "concept": "coherence",
            "definition": "internal consistency",
        })
        concepts = loom.read_state("L3")["concepts"]
        assert len(concepts) == 1

    def test_append_scar(self, loom):
        loom.append("L3", {
            "scar": "failed",
            "boon": "learned",
            "newrule": "do-better",
            "glyphstamp": "Phoenix",
        })
        log = loom.read_state("L3")["tracewisdomlog"]
        assert len(log) == 1
        assert log[0]["glyphstamp"] == "Phoenix"

    def test_append_l3_bad_dict_raises(self, loom):
        with pytest.raises(ValueError, match="must contain"):
            loom.append("L3", {"random_key": "value"})

    def test_append_l3_non_dict_raises(self, loom):
        with pytest.raises(TypeError, match="must be a dict"):
            loom.append("L3", "string value")


# ── Consolidation ────────────────────────────────────────


class TestConsolidation:
    def test_consolidate_l1_to_l2(self, loom):
        loom.update_l1("goal", "test")
        loom.update_l1("artifacts", ["a.md", "b.md"])
        entry = loom.consolidate_l1_to_l2()

        assert entry["phase"] == "consolidation"
        assert "test" in entry["summary"]
        # L1 should be cleared
        assert loom.read_state("L1")["goal"] == ""

    def test_consolidate_l1_custom_summary(self, loom):
        entry = loom.consolidate_l1_to_l2("custom summary")
        assert entry["summary"] == "custom summary"

    def test_consolidate_l2_to_l3_empty(self, loom):
        result = loom.consolidate_l2_to_l3()
        assert "empty" in result["message"].lower()

    def test_consolidate_l2_to_l3_with_entries(self, loom, tmp_path):
        for _ in range(5):
            loom.append("L2", "some work")
        result = loom.consolidate_l2_to_l3()
        assert isinstance(result, dict)
        # L2 should be cleared after archival
        assert len(loom.state["L2"]) == 0
        # Archive file should exist
        archive_dir = tmp_path / "loom_archives"
        assert archive_dir.exists()


# ── Validation ───────────────────────────────────────────


class TestValidation:
    def test_valid_state(self, loom):
        result = loom.validate_state()
        assert result["status"] == "VALID"

    def test_warn_l1_no_goal(self, loom):
        loom.update_l1("constraints", ["something"])
        result = loom.validate_state()
        assert result["status"] == "WARNINGS"
        assert any("no goal" in i.lower() for i in result["issues"])

    def test_warn_l3_rule_no_why(self, loom):
        loom.state["L3"]["rules"].append({"rule_id": "no-why"})
        loom.save_state()
        result = loom.validate_state()
        assert any("no 'why'" in i for i in result["issues"])


# ── Persistence ──────────────────────────────────────────


class TestPersistence:
    def test_save_and_load(self, loom, tmp_path):
        loom.update_l1("goal", "persist me")
        loom.append("L2", "session entry")

        # Create new instance from same file
        loom2 = LoomREPL(str(tmp_path / "test_state.md"))
        assert loom2.read_state("L1")["goal"] == "persist me"

    def test_state_file_is_markdown(self, loom):
        content = loom.state_file.read_text()
        assert content.startswith("# Loom State")
        assert "## L1: Working Pad" in content
        assert "## L2: Session Log" in content
        assert "## L3: Wisdom Base" in content

    def test_event_log(self, loom):
        loom.update_l1("goal", "test")
        loom.append("L2", "entry")
        events = loom.get_event_log()
        assert len(events) >= 2
        assert events[0]["layer"] == "L1"

    def test_save_event_log(self, loom, tmp_path):
        loom.update_l1("goal", "test")
        path = loom.save_event_log(str(tmp_path / "events.json"))
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert isinstance(data, list)


# ── Read state layer routing ─────────────────────────────


class TestReadState:
    def test_read_all(self, loom):
        state = loom.read_state("ALL")
        assert "L1" in state and "L2" in state and "L3" in state

    def test_read_none_returns_all(self, loom):
        state = loom.read_state(None)
        assert "metadata" in state

    def test_read_metadata(self, loom):
        meta = loom.read_state("METADATA")
        assert "session_id" in meta

    def test_read_invalid_raises(self, loom):
        with pytest.raises(ValueError, match="Invalid layer"):
            loom.read_state("L4")
