"""Tests for the hallucination detection ensemble — Phase 2."""

import pytest
from stateful_repl.hallucination import (
    HallucinationDetector,
    HallucinationScore,
    TokenProbabilityMethod,
    SamplingConsistencyMethod,
    SelfEvaluationMethod,
    RetrievalVerificationMethod,
)


# ─── Token Probability ───────────────────────────────

class TestTokenProbability:
    def test_no_logprobs(self):
        r = TokenProbabilityMethod().analyze(None)
        assert r.risk_score == 0.5
        assert r.confidence == 0.2

    def test_all_high(self):
        r = TokenProbabilityMethod().analyze([-0.1, -0.2, -0.3, -0.1])
        assert r.risk_score < 0.3

    def test_many_low(self):
        r = TokenProbabilityMethod().analyze([-3.0, -4.0, -5.0, -3.5, -0.1])
        assert r.risk_score > 0.3

    def test_long_low_span(self):
        logprobs = [-0.1] * 5 + [-3.0] * 5 + [-0.1] * 5
        r = TokenProbabilityMethod().analyze(logprobs)
        assert r.raw["max_span"] == 5


# ─── Sampling Consistency ────────────────────────────

class TestSamplingConsistency:
    def test_identical_responses(self):
        r = SamplingConsistencyMethod().analyze_responses([
            "The capital is Paris", "The capital is Paris", "The capital is Paris"
        ])
        assert r.risk_score < 0.3

    def test_inconsistent(self):
        r = SamplingConsistencyMethod().analyze_responses([
            "The capital is Paris",
            "The capital is Lyon",
            "The capital is Marseille",
        ])
        assert r.risk_score > 0.2

    def test_single_response(self):
        r = SamplingConsistencyMethod().analyze_responses(["solo"])
        assert r.confidence < 0.5


# ─── Self Evaluation ─────────────────────────────────

class TestSelfEvaluation:
    def test_confident_text(self):
        r = SelfEvaluationMethod().analyze_text("The answer is definitely 42. It is confirmed.")
        assert r.risk_score < 0.3

    def test_hedging_text(self):
        r = SelfEvaluationMethod().analyze_text("I think probably it might be something, I'm not sure.")
        assert r.risk_score > 0.3

    def test_neutral(self):
        r = SelfEvaluationMethod().analyze_text("Hello world.")
        assert r.risk_score == 0.3
        assert r.confidence == 0.4


# ─── Retrieval Verification ──────────────────────────

class TestRetrievalVerification:
    def test_matching_source(self):
        r = RetrievalVerificationMethod().analyze(
            "Paris is the capital of France",
            ["Paris is the capital of France"],
        )
        assert r.risk_score < 0.2

    def test_no_sources(self):
        r = RetrievalVerificationMethod().analyze("Some claim", [])
        assert r.risk_score == 0.5
        assert r.confidence == 0.3

    def test_contradicting_source(self):
        r = RetrievalVerificationMethod().analyze(
            "Lions are the fastest land animals",
            ["Cheetahs are the fastest land animals"],
        )
        assert r.risk_score > 0.1


# ─── Ensemble ────────────────────────────────────────

class TestHallucinationDetector:
    def test_structural_check_grounded(self):
        detector = HallucinationDetector(threshold=0.5)
        result = detector.check_structural(
            "Paris is the capital of France",
            sources=["Paris is the capital of France"],
            logprobs=[-0.1, -0.2, -0.1, -0.3],
            alternate_responses=["Paris is the capital of France"],
        )
        assert isinstance(result, HallucinationScore)
        assert not result.is_hallucinated
        assert result.risk < 0.5

    def test_structural_check_hallucinated(self):
        detector = HallucinationDetector(threshold=0.3)
        result = detector.check_structural(
            "I think maybe probably the thing is unclear",
            sources=["Completely unrelated text about quantum physics"],
            logprobs=[-4.0, -5.0, -3.0, -6.0],
        )
        assert result.risk > 0.3

    def test_no_data(self):
        detector = HallucinationDetector()
        result = detector.check_structural("hello world")
        assert isinstance(result, HallucinationScore)
        # Should still produce a score even with no data
        assert 0.0 <= result.risk <= 1.0

    def test_to_dict(self):
        detector = HallucinationDetector()
        result = detector.check_structural("test")
        d = result.to_dict()
        assert "risk" in d
        assert "methods" in d
        assert len(d["methods"]) == 4

    def test_summary(self):
        detector = HallucinationDetector()
        result = detector.check_structural("test", sources=["test"])
        s = result.summary()
        assert "Hallucination Check" in s

    def test_update_weights(self):
        detector = HallucinationDetector()
        detector.update_weights({
            "token_probability": 0.1,
            "sampling_consistency": 0.5,
            "self_evaluation": 0.3,
            "retrieval_verification": 0.2,
        })
        # After update, weights should still sum to ~1.0
        total = sum(detector.weights.values())
        assert 0.95 <= total <= 1.05
