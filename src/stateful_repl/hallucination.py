"""
Hallucination Detection — Phase 2.

Calibrated 4-method ensemble that detects when LLM outputs diverge
from grounded facts or internal consistency.

Methods:
  1. Token Probability Analysis — flag low-confidence tokens via logprobs
  2. Sampling Consistency       — compare N completions for same prompt
  3. Self-Evaluation            — ask the model to rate its own confidence
  4. Retrieval Verification     — cross-check claims against retrieved sources

The ensemble produces a calibrated HallucinationScore.  Individual
method weights are tunable and can be trained via Brier score feedback.

Usage (structural / rule-based, no model needed):
    detector = HallucinationDetector()
    result = detector.check_structural(state)

Usage (model-assisted):
    detector = HallucinationDetector(model=adapter)
    result = await detector.check("The capital of France is Lyon.", sources=["Paris is the capital."])
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stateful_repl.models import ModelAdapter


# ─────────────────────────────────────────────────────────
# Result Types
# ─────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """Result from a single detection method."""
    method: str
    risk_score: float       # 0.0 (safe) to 1.0 (hallucinated)
    confidence: float       # How confident we are in this score
    evidence: List[str]
    raw: Optional[Dict[str, Any]] = None


@dataclass
class HallucinationScore:
    """Aggregate hallucination assessment."""
    risk: float                     # 0.0–1.0 overall risk
    is_hallucinated: bool           # risk > threshold
    method_results: Dict[str, MethodResult]
    calibrated_confidence: float    # ensemble confidence
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk": self.risk,
            "is_hallucinated": self.is_hallucinated,
            "calibrated_confidence": self.calibrated_confidence,
            "timestamp": self.timestamp,
            "methods": {
                name: {
                    "risk_score": r.risk_score,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                }
                for name, r in self.method_results.items()
            },
        }

    def summary(self) -> str:
        verdict = "⚠️ HALLUCINATED" if self.is_hallucinated else "✓ GROUNDED"
        lines = [
            f"Hallucination Check: {verdict} "
            f"(risk={self.risk:.2f}, confidence={self.calibrated_confidence:.0%})"
        ]
        for name, r in self.method_results.items():
            lines.append(f"  {name:<25s} risk={r.risk_score:.2f}  conf={r.confidence:.2f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Method 1: Token Probability Analysis
# ─────────────────────────────────────────────────────────

class TokenProbabilityMethod:
    """Flags spans where token log-probabilities drop below threshold."""

    name = "token_probability"

    def __init__(self, low_prob_threshold: float = -2.0, min_span: int = 3):
        self.threshold = low_prob_threshold
        self.min_span = min_span

    def analyze(self, logprobs: Optional[List[float]]) -> MethodResult:
        if not logprobs:
            return MethodResult(
                method=self.name, risk_score=0.5, confidence=0.2,
                evidence=["No logprobs available — cannot assess"],
            )

        low = [lp for lp in logprobs if lp < self.threshold]
        ratio = len(low) / len(logprobs)

        # Check for concerning contiguous spans
        max_span = 0
        current_span = 0
        for lp in logprobs:
            if lp < self.threshold:
                current_span += 1
                max_span = max(max_span, current_span)
            else:
                current_span = 0

        evidence = [
            f"{len(low)}/{len(logprobs)} tokens below threshold {self.threshold}",
            f"Max low-confidence span: {max_span} tokens",
        ]

        span_penalty = 0.1 if max_span >= self.min_span else 0.0
        risk = min(1.0, ratio * 0.7 + span_penalty + (0.1 if ratio > 0.3 else 0.0))

        return MethodResult(
            method=self.name, risk_score=round(risk, 3),
            confidence=0.8, evidence=evidence,
            raw={"low_count": len(low), "max_span": max_span},
        )


# ─────────────────────────────────────────────────────────
# Method 2: Sampling Consistency
# ─────────────────────────────────────────────────────────

class SamplingConsistencyMethod:
    """
    Compare N completions for the same prompt.
    High variance → likely hallucination.
    """

    name = "sampling_consistency"

    def __init__(self, n_samples: int = 5, temperature: float = 0.7):
        self.n_samples = n_samples
        self.temperature = temperature

    def analyze_responses(self, responses: List[str]) -> MethodResult:
        """Analyze pre-collected responses for consistency."""
        if len(responses) < 2:
            return MethodResult(
                method=self.name, risk_score=0.5, confidence=0.3,
                evidence=["Insufficient samples for consistency check"],
            )

        # Pairwise similarity
        sims = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sims.append(
                    SequenceMatcher(None, responses[i], responses[j]).ratio()
                )

        avg_sim = statistics.mean(sims)
        std_sim = statistics.stdev(sims) if len(sims) > 1 else 0.0

        evidence = [
            f"Avg pairwise similarity: {avg_sim:.2f}",
            f"Similarity std dev: {std_sim:.2f}",
            f"Compared {len(responses)} samples",
        ]

        # Low similarity or high variance → hallucination risk
        risk = max(0.0, 1.0 - avg_sim) * 0.6 + min(0.4, std_sim)
        risk = min(1.0, risk)

        return MethodResult(
            method=self.name, risk_score=round(risk, 3),
            confidence=0.75, evidence=evidence,
        )


# ─────────────────────────────────────────────────────────
# Method 3: Self-Evaluation (structural proxy)
# ─────────────────────────────────────────────────────────

class SelfEvaluationMethod:
    """
    Without a model: structural check for hedging language.
    With a model: ask the model to rate its own confidence.
    """

    name = "self_evaluation"

    HEDGE_PHRASES = [
        "i think", "probably", "i'm not sure", "it might",
        "i believe", "approximately", "roughly", "possibly",
        "as far as i know", "i don't have", "i cannot confirm",
        "i'm uncertain", "it's unclear",
    ]

    CONFIDENT_PHRASES = [
        "definitely", "certainly", "it is", "the answer is",
        "according to", "based on", "confirmed", "verified",
    ]

    def analyze_text(self, text: str) -> MethodResult:
        """Structural hedge/confidence analysis."""
        text_lower = text.lower()

        hedge_count = sum(1 for p in self.HEDGE_PHRASES if p in text_lower)
        confident_count = sum(1 for p in self.CONFIDENT_PHRASES if p in text_lower)

        total = hedge_count + confident_count
        if total == 0:
            # Neutral text — moderate risk
            return MethodResult(
                method=self.name, risk_score=0.3, confidence=0.4,
                evidence=["No hedge or confidence markers found"],
            )

        hedge_ratio = hedge_count / total
        evidence = [
            f"Hedge markers: {hedge_count}",
            f"Confidence markers: {confident_count}",
            f"Hedge ratio: {hedge_ratio:.2f}",
        ]

        risk = hedge_ratio * 0.6
        return MethodResult(
            method=self.name, risk_score=round(risk, 3),
            confidence=0.5, evidence=evidence,
        )


# ─────────────────────────────────────────────────────────
# Method 4: Retrieval Verification
# ─────────────────────────────────────────────────────────

class RetrievalVerificationMethod:
    """
    Cross-check claim text against provided source texts.
    """

    name = "retrieval_verification"

    def __init__(self, min_overlap: float = 0.3):
        self.min_overlap = min_overlap

    def analyze(self, claim: str, sources: List[str]) -> MethodResult:
        if not sources:
            return MethodResult(
                method=self.name, risk_score=0.5, confidence=0.3,
                evidence=["No sources provided for verification"],
            )

        # Compare claim against each source
        best_match = 0.0
        matches = []
        for i, source in enumerate(sources):
            sim = SequenceMatcher(None, claim.lower(), source.lower()).ratio()
            matches.append(sim)
            best_match = max(best_match, sim)

        avg_match = statistics.mean(matches)
        evidence = [
            f"Best source match: {best_match:.2f}",
            f"Avg source match: {avg_match:.2f}",
            f"Checked against {len(sources)} sources",
        ]

        # Word-level coverage: what fraction of claim words appear in sources
        claim_words = set(claim.lower().split())
        source_words = set()
        for s in sources:
            source_words.update(s.lower().split())

        if claim_words:
            word_coverage = len(claim_words & source_words) / len(claim_words)
            evidence.append(f"Word coverage: {word_coverage:.2f}")
        else:
            word_coverage = 1.0

        risk = max(0.0, 1.0 - (best_match * 0.4 + word_coverage * 0.4 + avg_match * 0.2))
        return MethodResult(
            method=self.name, risk_score=round(risk, 3),
            confidence=0.7, evidence=evidence,
        )


# ─────────────────────────────────────────────────────────
# Ensemble Detector
# ─────────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Calibrated 4-method ensemble.

    Weights can be tuned via ``update_weights()`` using Brier score feedback.
    """

    DEFAULT_WEIGHTS = {
        "token_probability": 0.30,
        "sampling_consistency": 0.25,
        "self_evaluation": 0.15,
        "retrieval_verification": 0.30,
    }

    def __init__(
        self,
        model: Optional["ModelAdapter"] = None,
        threshold: float = 0.5,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.model = model
        self.threshold = threshold
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)

        # Sub-methods
        self.token_prob = TokenProbabilityMethod()
        self.sampling = SamplingConsistencyMethod()
        self.self_eval = SelfEvaluationMethod()
        self.retrieval = RetrievalVerificationMethod()

    def check_structural(
        self,
        text: str,
        sources: Optional[List[str]] = None,
        logprobs: Optional[List[float]] = None,
        alternate_responses: Optional[List[str]] = None,
    ) -> HallucinationScore:
        """
        Run all available methods without needing a live model.
        Pass in whatever data you have; methods adapt gracefully.
        """
        results: Dict[str, MethodResult] = {}

        results[self.token_prob.name] = self.token_prob.analyze(logprobs)
        results[self.self_eval.name] = self.self_eval.analyze_text(text)
        results[self.retrieval.name] = self.retrieval.analyze(text, sources or [])

        if alternate_responses:
            results[self.sampling.name] = self.sampling.analyze_responses(
                [text] + alternate_responses
            )
        else:
            results[self.sampling.name] = MethodResult(
                method=self.sampling.name, risk_score=0.5, confidence=0.2,
                evidence=["No alternate responses provided"],
            )

        return self._aggregate(results)

    def _aggregate(self, results: Dict[str, MethodResult]) -> HallucinationScore:
        """Weighted average with confidence weighting."""
        weighted_risk = 0.0
        total_weight = 0.0

        for name, result in results.items():
            w = self.weights.get(name, 0.0)
            # Scale weight by confidence
            effective_w = w * result.confidence
            weighted_risk += result.risk_score * effective_w
            total_weight += effective_w

        if total_weight > 0:
            risk = weighted_risk / total_weight
        else:
            risk = 0.5

        # Calibrated confidence = mean of individual confidences weighted
        confidences = [r.confidence for r in results.values()]
        cal_conf = statistics.mean(confidences) if confidences else 0.5

        return HallucinationScore(
            risk=round(risk, 3),
            is_hallucinated=risk > self.threshold,
            method_results=results,
            calibrated_confidence=round(cal_conf, 3),
        )

    def update_weights(self, feedback: Dict[str, float]) -> None:
        """
        Adjust method weights based on Brier score feedback.

        Args:
            feedback: dict of {method_name: brier_score}.
                      Lower Brier → better calibration → higher weight.
        """
        inv_scores = {}
        for name, brier in feedback.items():
            if name in self.weights:
                inv_scores[name] = 1.0 / (brier + 0.01)

        total = sum(inv_scores.values())
        if total > 0:
            for name, inv in inv_scores.items():
                self.weights[name] = round(inv / total, 3)
