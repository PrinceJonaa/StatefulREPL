"""Predictive context prefetching for Phase 4.

Uses lightweight transition probabilities and recency weighting to
suggest likely next context keys/artifacts.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class PrefetchCandidate:
    """A predicted next key/artifact."""

    key: str
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "score": round(self.score, 4),
            "reason": self.reason,
        }


@dataclass
class PrefetchQuality:
    """Quality metrics for prefetch predictions."""

    total_predictions: int
    hit_rate_at_k: float
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_predictions": self.total_predictions,
            "hit_rate_at_k": round(self.hit_rate_at_k, 3),
            "mrr": round(self.mrr, 3),
        }


class PredictivePrefetchEngine:
    """Markov-style prefetch engine with recency bias.

    CLAIM-402: "Transition + recency hybrid is sufficient for first-pass
    prefetch quality in single-session workflows." [scope: module]
    [confidence: 0.77]
    [falsifies: "Top-3 recall below 0.4 on representative session logs"]
    """

    def __init__(self, window_size: int = 200, recency_weight: float = 0.25):
        if window_size < 2:
            raise ValueError(f"Invalid window_size {window_size}: must be >= 2")
        if not 0.0 <= recency_weight <= 1.0:
            raise ValueError(
                f"Invalid recency_weight {recency_weight}: must be 0.0-1.0"
            )

        self.window_size = window_size
        self.recency_weight = recency_weight
        self._history: deque[str] = deque(maxlen=window_size)
        self._transitions: dict[str, Counter[str]] = defaultdict(Counter)
        self._last_seen: dict[str, str] = {}

    def record_access(self, key: str, timestamp: str | None = None) -> None:
        """Record an access event for a key/artifact."""
        cleaned = key.strip()
        if not cleaned:
            raise ValueError("key must be non-empty")

        if self._history:
            prev = self._history[-1]
            self._transitions[prev][cleaned] += 1

        self._history.append(cleaned)
        self._last_seen[cleaned] = timestamp or datetime.now().isoformat()

    def fit_from_sequence(self, keys: list[str]) -> None:
        """Train from a historical sequence of accesses."""
        for key in keys:
            self.record_access(key)

    def predict_next(self, current_keys: list[str] | None = None, limit: int = 5) -> list[PrefetchCandidate]:
        """Predict likely next keys.

        Args:
            current_keys: Recent key context; uses latest history if omitted.
            limit: Max number of candidates.

        Returns:
            Ranked prefetch candidates.
        """
        if limit < 1:
            raise ValueError(f"Invalid limit {limit}: must be >= 1")

        anchors = [k for k in (current_keys or []) if k]
        if not anchors and self._history:
            anchors = [self._history[-1]]
        if not anchors:
            return []

        combined_scores: Counter[str] = Counter()
        for anchor in anchors:
            transitions = self._transitions.get(anchor, Counter())
            total = sum(transitions.values())
            if total == 0:
                continue
            for nxt, count in transitions.items():
                combined_scores[nxt] += count / total

        if not combined_scores:
            return []

        recency_scores = self._recency_scores()
        candidates: list[PrefetchCandidate] = []
        for key, score in combined_scores.items():
            recency = recency_scores.get(key, 0.0)
            final_score = (1.0 - self.recency_weight) * score + self.recency_weight * recency
            candidates.append(
                PrefetchCandidate(
                    key=key,
                    score=final_score,
                    reason="transition+recency",
                )
            )

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]

    def _recency_scores(self) -> dict[str, float]:
        ordered = list(self._history)
        if not ordered:
            return {}

        scores: dict[str, float] = {}
        n = len(ordered)
        for idx, key in enumerate(ordered):
            # Later indices => higher scores.
            scores[key] = max(scores.get(key, 0.0), (idx + 1) / n)
        return scores

    def summary(self) -> dict[str, Any]:
        """Return engine summary stats."""
        return {
            "history_size": len(self._history),
            "tracked_keys": len(self._last_seen),
            "transition_nodes": len(self._transitions),
        }

    def evaluate_trace(self, trace: list[str], k: int = 3, warmup: int = 2) -> PrefetchQuality:
        """Evaluate prefetch quality on a sequential trace.

        Args:
            trace: Ordered key access sequence.
            k: Top-k cutoff for hit-rate.
            warmup: Number of initial events before evaluating.

        Returns:
            Prefetch quality metrics.
        """
        if k < 1:
            raise ValueError(f"Invalid k {k}: must be >= 1")
        if warmup < 1:
            raise ValueError(f"Invalid warmup {warmup}: must be >= 1")
        if len(trace) <= warmup:
            return PrefetchQuality(total_predictions=0, hit_rate_at_k=0.0, mrr=0.0)

        sim = PredictivePrefetchEngine(
            window_size=self.window_size,
            recency_weight=self.recency_weight,
        )

        for key in trace[:warmup]:
            sim.record_access(key)

        total = 0
        hits = 0
        reciprocal_rank_sum = 0.0

        for i in range(warmup, len(trace)):
            actual = trace[i]
            anchor = trace[i - 1]
            preds = sim.predict_next([anchor], limit=k)
            ranked = [p.key for p in preds]

            total += 1
            if actual in ranked:
                hits += 1
                rank = ranked.index(actual) + 1
                reciprocal_rank_sum += 1.0 / rank

            sim.record_access(actual)

        hit_rate = hits / total if total else 0.0
        mrr = reciprocal_rank_sum / total if total else 0.0
        return PrefetchQuality(total_predictions=total, hit_rate_at_k=hit_rate, mrr=mrr)
