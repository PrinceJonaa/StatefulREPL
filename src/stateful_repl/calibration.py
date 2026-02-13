"""Calibration learning utilities for Phase 4.

Provides a lightweight post-hoc calibrator for confidence scores and
quality weighting recommendations.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CalibrationSample:
    """One prediction/outcome pair.

    Attributes:
        predicted: Predicted probability in range $[0, 1]$.
        observed: Ground-truth outcome, either 0 or 1.
    """

    predicted: float
    observed: int


@dataclass
class CalibrationReport:
    """Result from calibration fitting."""

    sample_count: int
    brier_before: float
    brier_after: float
    offset: float
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "brier_before": round(self.brier_before, 4),
            "brier_after": round(self.brier_after, 4),
            "offset": round(self.offset, 4),
            "scale": round(self.scale, 4),
        }


@dataclass
class CalibrationQuality:
    """Holdout quality metrics for calibration."""

    train_size: int
    holdout_size: int
    holdout_brier_before: float
    holdout_brier_after: float
    improvement: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_size": self.train_size,
            "holdout_size": self.holdout_size,
            "holdout_brier_before": round(self.holdout_brier_before, 4),
            "holdout_brier_after": round(self.holdout_brier_after, 4),
            "improvement": round(self.improvement, 4),
        }


class CalibrationLearner:
    """Learns and persists a simple affine calibration function.

    Calibrated probability:
            $p' = clip(offset + scale * p, 0, 1)$
    """

    def __init__(self, weights_path: str = "calibration_weights.json"):
        self.weights_path = Path(weights_path)
        self.offset = 0.0
        self.scale = 1.0
        if self.weights_path.exists():
            self.load()

    def fit(self, samples: list[CalibrationSample]) -> CalibrationReport:
        """Fit calibration parameters using least-squares affine mapping."""
        if len(samples) < 3:
            raise ValueError("Need at least 3 samples for calibration")

        offset, scale, before, after = self._fit_params(samples)

        self.offset = float(offset)
        self.scale = float(scale)
        self.save()

        return CalibrationReport(
            sample_count=len(samples),
            brier_before=before,
            brier_after=after,
            offset=self.offset,
            scale=self.scale,
        )

    def evaluate_holdout(
        self,
        samples: list[CalibrationSample],
        holdout_ratio: float = 0.3,
    ) -> CalibrationQuality:
        """Evaluate calibration quality using chronological holdout split."""
        if len(samples) < 6:
            raise ValueError("Need at least 6 samples for holdout evaluation")
        if not 0.1 <= holdout_ratio <= 0.5:
            raise ValueError(
                f"Invalid holdout_ratio {holdout_ratio}: must be 0.1-0.5"
            )

        split = max(3, int(round(len(samples) * (1.0 - holdout_ratio))))
        split = min(split, len(samples) - 3)

        train = samples[:split]
        holdout = samples[split:]

        offset, scale, _, _ = self._fit_params(train)
        hold_preds = [self._clamp01(s.predicted) for s in holdout]
        hold_obs = [1 if s.observed else 0 for s in holdout]

        brier_before = statistics.mean((p - o) ** 2 for p, o in zip(hold_preds, hold_obs))
        brier_after = statistics.mean(
            (self._clamp01(offset + scale * p) - o) ** 2 for p, o in zip(hold_preds, hold_obs)
        )

        return CalibrationQuality(
            train_size=len(train),
            holdout_size=len(holdout),
            holdout_brier_before=brier_before,
            holdout_brier_after=brier_after,
            improvement=brier_before - brier_after,
        )

    def calibrate_probability(self, predicted: float) -> float:
        """Apply learned calibration to a probability."""
        p = self._clamp01(predicted)
        return self._clamp01(self.offset + self.scale * p)

    def recommend_quality_weights(self, outcomes: list[dict[str, Any]]) -> dict[str, float]:
        """Recommend dimension weights from outcome records.

        Expected record shape:
            {
              "dimension": "internal_consistency",
              "predicted": 0.7,
              "observed": 1
            }
        """
        by_dim: dict[str, list[float]] = {}
        for row in outcomes:
            dim = str(row.get("dimension", "")).strip()
            if not dim:
                continue
            pred = self._clamp01(float(row.get("predicted", 0.0)))
            obs = 1.0 if row.get("observed", 0) else 0.0
            err = abs(pred - obs)
            by_dim.setdefault(dim, []).append(err)

        if not by_dim:
            return {}

        # Lower error => higher weight.
        raw_weights: dict[str, float] = {}
        for dim, errs in by_dim.items():
            mae = statistics.mean(errs)
            raw_weights[dim] = max(0.05, 1.0 - mae)

        total = sum(raw_weights.values()) or 1.0
        return {dim: round(w / total, 4) for dim, w in raw_weights.items()}

    def save(self) -> None:
        """Persist calibration params to disk."""
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"offset": self.offset, "scale": self.scale}
        self.weights_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load calibration params from disk."""
        try:
            text = self.weights_path.read_text(encoding="utf-8").strip()
            if not text:
                return
            raw = json.loads(text)
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return

        self.offset = float(raw.get("offset", 0.0))
        self.scale = float(raw.get("scale", 1.0))

    def summary(self) -> dict[str, Any]:
        """Get current calibration parameters."""
        return {
            "offset": round(self.offset, 4),
            "scale": round(self.scale, 4),
            "weights_path": str(self.weights_path),
        }

    def _clamp01(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _fit_params(self, samples: list[CalibrationSample]) -> tuple[float, float, float, float]:
        preds = [self._clamp01(s.predicted) for s in samples]
        obs = [1 if s.observed else 0 for s in samples]

        mean_p = statistics.mean(preds)
        mean_o = statistics.mean(obs)

        var_p = statistics.mean((p - mean_p) ** 2 for p in preds)
        if var_p == 0:
            scale = 1.0
        else:
            cov = statistics.mean((p - mean_p) * (o - mean_o) for p, o in zip(preds, obs))
            scale = cov / var_p

        offset = mean_o - scale * mean_p
        before = statistics.mean((p - o) ** 2 for p, o in zip(preds, obs))
        after = statistics.mean((self._clamp01(offset + scale * p) - o) ** 2 for p, o in zip(preds, obs))
        return float(offset), float(scale), float(before), float(after)
