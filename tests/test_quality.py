"""Tests for the 7D quality vector — Phase 2."""

import pytest
from stateful_repl.quality import (
    QualityEvaluator,
    QualityVector,
    DimensionResult,
    InternalConsistency,
    ExternalCorrespondence,
    TemporalStability,
    CausalValidity,
    RelationalIntegrity,
    MultiscaleAlignment,
    PredictiveAccuracy,
)
from datetime import datetime, timedelta


# ─── Helper: build a realistic state ──────────────────

def _make_state(
    goal="Build the Loom",
    artifacts=None,
    constraints=None,
    questions=None,
    l2_entries=None,
    rules=None,
    scars=None,
    concepts=None,
):
    now = datetime.now()
    return {
        "L1": {
            "goal": goal,
            "artifacts": artifacts or ["agents.md", "loom_state.py"],
            "constraints": constraints or ["No side effects"],
            "open_questions": questions or ["How to test quality?"],
        },
        "L2": l2_entries or [
            {"timestamp": (now - timedelta(minutes=5)).isoformat(), "content": "Started agents.md", "summary": ""},
            {"timestamp": (now - timedelta(minutes=3)).isoformat(), "content": "Built loom_state.py", "summary": ""},
            {"timestamp": now.isoformat(), "content": "Testing quality", "summary": ""},
        ],
        "L3": {
            "rules": rules or [
                {"rule_id": "R1", "when": "file already exists", "then": "delete first", "why": "create_file fails"},
            ],
            "concepts": concepts or [
                {"concept": "event sourcing", "definition": "append-only log"},
            ],
            "tracewisdomlog": scars or [
                {"scar": "crashed on import", "boon": "fixed path", "newrule": "R1", "glyphstamp": "path-fix", "userfeedback": "agreed"},
            ],
        },
    }


# ─── Evaluator integration ────────────────────────────

class TestQualityEvaluator:
    def test_evaluate_returns_vector(self):
        evaluator = QualityEvaluator()
        vector = evaluator.evaluate(_make_state())
        assert isinstance(vector, QualityVector)
        assert 0.0 <= vector.aggregate_score <= 1.0
        assert 0.0 <= vector.aggregate_confidence <= 1.0
        assert len(vector.dimensions) == 7

    def test_all_dimensions_scored(self):
        vector = QualityEvaluator().evaluate(_make_state())
        expected = {
            "internal_consistency", "external_correspondence",
            "temporal_stability", "causal_validity",
            "relational_integrity", "multiscale_alignment",
            "predictive_accuracy",
        }
        assert set(vector.dimensions.keys()) == expected

    def test_to_dict_roundtrip(self):
        vector = QualityEvaluator().evaluate(_make_state())
        d = vector.to_dict()
        assert "aggregate_score" in d
        assert "dimensions" in d
        assert len(d["dimensions"]) == 7

    def test_summary_string(self):
        vector = QualityEvaluator().evaluate(_make_state())
        s = vector.summary()
        assert "Quality Vector" in s
        assert "internal_consistency" in s

    def test_empty_state(self):
        state = {"L1": {"goal": "", "artifacts": [], "constraints": [], "open_questions": []}, "L2": [], "L3": {"rules": [], "concepts": [], "tracewisdomlog": []}}
        vector = QualityEvaluator().evaluate(state)
        assert isinstance(vector, QualityVector)
        # Empty state may have high scores (no contradictions) but low confidence
        assert 0.0 <= vector.aggregate_score <= 1.0


# ─── Individual dimensions ────────────────────────────

class TestInternalConsistency:
    def test_good_state(self):
        r = InternalConsistency().compute(_make_state())
        assert r.score >= 0.7
        assert r.name == "internal_consistency"

    def test_artifacts_without_goal(self):
        r = InternalConsistency().compute(_make_state(goal=""))
        assert r.score < 1.0
        assert any("no goal" in e for e in r.evidence)

    def test_duplicate_rules_penalized(self):
        rules = [
            {"rule_id": "R1", "when": "x", "then": "y", "why": "z"},
            {"rule_id": "R1", "when": "a", "then": "b", "why": "c"},
        ]
        r = InternalConsistency().compute(_make_state(rules=rules))
        assert r.score < 1.0


class TestExternalCorrespondence:
    def test_with_references(self):
        r = ExternalCorrespondence().compute(_make_state())
        assert r.score > 0.5

    def test_no_artifacts(self):
        r = ExternalCorrespondence().compute(_make_state(artifacts=[]))
        assert r.score > 0.0
        assert r.name == "external_correspondence"


class TestTemporalStability:
    def test_ordered_entries(self):
        r = TemporalStability().compute(_make_state())
        assert r.score >= 0.8

    def test_few_entries(self):
        r = TemporalStability().compute(_make_state(l2_entries=[{"timestamp": datetime.now().isoformat(), "content": "solo"}]))
        assert r.score == 1.0
        assert r.confidence == 0.5


class TestCausalValidity:
    def test_complete_rules(self):
        r = CausalValidity().compute(_make_state())
        assert r.score >= 0.7

    def test_incomplete_rules(self):
        rules = [{"rule_id": "R1", "when": "", "then": "x", "why": ""}]
        r = CausalValidity().compute(_make_state(rules=rules))
        assert r.score < 1.0


class TestRelationalIntegrity:
    def test_artifacts_in_l2(self):
        r = RelationalIntegrity().compute(_make_state())
        assert r.score > 0.3

    def test_no_artifacts(self):
        r = RelationalIntegrity().compute(_make_state(artifacts=[]))
        assert r.score > 0.0


class TestMultiscaleAlignment:
    def test_well_formed(self):
        r = MultiscaleAlignment().compute(_make_state())
        assert r.score >= 0.7

    def test_empty_l2(self):
        r = MultiscaleAlignment().compute(_make_state(l2_entries=[]))
        assert r.score == 1.0


class TestPredictiveAccuracy:
    def test_testable_questions(self):
        r = PredictiveAccuracy().compute(_make_state(questions=["How does quality work?"]))
        assert r.score > 0.5

    def test_no_questions(self):
        r = PredictiveAccuracy().compute(_make_state(questions=[]))
        assert r.score >= 0.7
