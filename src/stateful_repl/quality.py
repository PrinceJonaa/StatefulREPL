"""
Quality Vector Computation — Phase 2.

Implements the 7D quality vector that scores every operation across:
  1. Internal Consistency   — agreement across rephrased queries
  2. External Correspondence — alignment with retrieved facts
  3. Temporal Stability     — consistency of answers over time
  4. Causal Validity        — logical cause-effect reasoning
  5. Relational Integrity   — entity relationships preserved
  6. Multiscale Alignment   — micro/macro coherence
  7. Predictive Accuracy    — verifiable forward predictions

Each dimension is a Strategy object conforming to QualityDimension.
QualityEvaluator computes the full vector in a single call.

Usage without a model (rule-based / structural analysis):
    evaluator = QualityEvaluator()
    vector = evaluator.evaluate(state)

Usage with a model (LLM-assisted quality measurement):
    evaluator = QualityEvaluator(model=adapter)
    vector = evaluator.evaluate(state)
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from stateful_repl.models import ModelAdapter


# ─────────────────────────────────────────────────────────
# Protocol
# ─────────────────────────────────────────────────────────

class QualityDimension(ABC):
    """Strategy interface for a single quality axis."""

    name: str
    weight: float = 1.0

    @abstractmethod
    def compute(
        self,
        state: Dict[str, Any],
        model: Optional["ModelAdapter"] = None,
    ) -> "DimensionResult":
        ...


@dataclass
class DimensionResult:
    """Score + explanation for one quality dimension."""

    name: str
    score: float          # 0.0–1.0
    confidence: float     # how confident we are in this score
    evidence: List[str]   # what contributed to the score
    suggestions: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Dimension 1: Internal Consistency
# ─────────────────────────────────────────────────────────

class InternalConsistency(QualityDimension):
    """
    Do the parts of the state agree with each other?

    Checks: L1 goal alignment, L3 rule uniqueness, scar→rule linkage,
    L2 chronological ordering.
    """

    name = "internal_consistency"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        penalties: List[float] = []

        l1 = state.get("L1", {})
        l2 = state.get("L2", [])
        l3 = state.get("L3", {})

        if l1.get("artifacts") and not l1.get("goal"):
            penalties.append(0.15)
            evidence.append("L1 has artifacts but no goal")
        elif l1.get("constraints") and not l1.get("goal"):
            penalties.append(0.10)
            evidence.append("L1 has constraints but no goal")
        elif l1.get("goal") and l1.get("artifacts"):
            evidence.append(f"L1 has goal + {len(l1['artifacts'])} artifacts")

        rules = l3.get("rules", [])
        rule_ids = [r.get("rule_id", "") for r in rules]
        dupes = [rid for rid, cnt in Counter(rule_ids).items() if cnt > 1]
        if dupes:
            penalties.append(0.10 * len(dupes))
            evidence.append(f"Duplicate rule IDs: {dupes}")
        else:
            evidence.append(f"{len(rules)} unique rules")

        incomplete = [
            r.get("rule_id", "?")
            for r in rules
            if not all(r.get(k) for k in ("when", "then", "why"))
        ]
        if incomplete:
            penalties.append(0.05 * len(incomplete))
            evidence.append(f"Incomplete rules: {incomplete}")

        scars = l3.get("tracewisdomlog", [])
        scar_rules = {s.get("newrule") for s in scars if s.get("newrule")}
        orphaned = scar_rules - set(rule_ids) - {""}
        if orphaned:
            penalties.append(0.05 * len(orphaned))
            evidence.append(f"Scars reference missing rules: {orphaned}")

        if len(l2) >= 2:
            timestamps = [e.get("timestamp", "") for e in l2 if e.get("timestamp")]
            if timestamps == sorted(timestamps):
                evidence.append("L2 chronologically ordered")
            else:
                penalties.append(0.10)
                evidence.append("L2 timestamps out of order")

        score = max(0.0, 1.0 - sum(penalties))
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.9, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Dimension 2: External Correspondence
# ─────────────────────────────────────────────────────────

class ExternalCorrespondence(QualityDimension):
    """
    Do claims correspond to verifiable external facts?

    Checks artifact references in L2, rule justifications, scar user feedback.
    """

    name = "external_correspondence"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l1 = state.get("L1", {})
        l2 = state.get("L2", [])
        l3 = state.get("L3", {})

        artifacts = l1.get("artifacts", [])
        l2_texts = [str(e.get("content", "") or e.get("summary", "")) for e in l2]
        if artifacts:
            refs = sum(1 for t in l2_texts if any(a in t for a in artifacts))
            ref_ratio = refs / max(len(l2_texts), 1)
            evidence.append(f"{refs}/{len(l2_texts)} L2 entries reference artifacts")
        else:
            ref_ratio = 0.5
            evidence.append("No artifacts to cross-reference")

        rules = l3.get("rules", [])
        grounded = sum(1 for r in rules if r.get("why"))
        rule_ratio = grounded / len(rules) if rules else 1.0
        if rules:
            evidence.append(f"{grounded}/{len(rules)} rules have justifications")

        scars = l3.get("tracewisdomlog", [])
        validated = sum(1 for s in scars if s.get("userfeedback"))
        scar_ratio = validated / len(scars) if scars else 1.0
        if scars:
            evidence.append(f"{validated}/{len(scars)} scars have user feedback")

        score = ref_ratio * 0.3 + rule_ratio * 0.4 + scar_ratio * 0.3
        return DimensionResult(
            name=self.name,
            score=round(min(1.0, score), 3),
            confidence=0.7 if model is None else 0.85,
            evidence=evidence,
        )


# ─────────────────────────────────────────────────────────
# Dimension 3: Temporal Stability
# ─────────────────────────────────────────────────────────

class TemporalStability(QualityDimension):
    """
    Is the state consistent over time?

    Measures goal drift, entry regularity, consolidation health.
    """

    name = "temporal_stability"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l2 = state.get("L2", [])

        if len(l2) < 2:
            return DimensionResult(
                name=self.name, score=1.0, confidence=0.5,
                evidence=["Too few L2 entries to assess temporal stability"],
            )

        snapshots = [e for e in l2 if e.get("l1_snapshot")]
        goals = [s["l1_snapshot"].get("goal", "") for s in snapshots if s.get("l1_snapshot")]
        if len(goals) >= 2:
            similarities = [
                SequenceMatcher(None, goals[i], goals[i + 1]).ratio()
                for i in range(len(goals) - 1)
            ]
            avg_sim = statistics.mean(similarities)
            evidence.append(f"Goal stability across {len(goals)} snapshots: {avg_sim:.2f}")
        else:
            avg_sim = 1.0
            evidence.append("Stable (single goal snapshot)")

        timestamps = []
        for e in l2:
            ts = e.get("timestamp", "")
            if ts:
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except (ValueError, TypeError):
                    pass

        if len(timestamps) >= 3:
            gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds() for i in range(len(timestamps) - 1)]
            mean_gap = statistics.mean(gaps) if gaps else 1
            if mean_gap > 0:
                cv = statistics.stdev(gaps) / mean_gap
                regularity = max(0.0, 1.0 - cv * 0.3)
            else:
                regularity = 1.0
            evidence.append(f"Entry regularity: {regularity:.2f}")
        else:
            regularity = 1.0

        n_raw = len([e for e in l2 if not e.get("phase")])
        consolidation_health = 0.7 if n_raw > 15 else 1.0
        evidence.append(f"{n_raw} unconsolidated entries")

        score = avg_sim * 0.4 + regularity * 0.3 + consolidation_health * 0.3
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.75, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Dimension 4: Causal Validity
# ─────────────────────────────────────────────────────────

class CausalValidity(QualityDimension):
    """
    Are cause-effect relationships correctly represented?

    Checks rule causal chains (when→then→why), scar chains (scar→boon→newrule),
    trigger specificity.
    """

    name = "causal_validity"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l3 = state.get("L3", {})

        rules = l3.get("rules", [])
        if rules:
            complete = sum(1 for r in rules if r.get("when") and r.get("then") and r.get("why"))
            rule_score = complete / len(rules)
            evidence.append(f"{complete}/{len(rules)} rules have full causal chain")
        else:
            rule_score = 1.0
            evidence.append("No rules to evaluate")

        scars = l3.get("tracewisdomlog", [])
        if scars:
            complete_scars = sum(1 for s in scars if s.get("scar") and s.get("boon"))
            scar_score = complete_scars / len(scars)
            evidence.append(f"{complete_scars}/{len(scars)} scars have scar→boon chain")
            with_rules = sum(1 for s in scars if s.get("newrule"))
            if with_rules:
                evidence.append(f"{with_rules} scars produced new rules")
        else:
            scar_score = 1.0

        generic = sum(
            1 for r in rules
            if r.get("when", "").lower() in ("repeated_action", "always", "")
        )
        specificity = 1.0 - (generic / len(rules)) * 0.3 if rules else 1.0
        if generic:
            evidence.append(f"{generic} rules have generic triggers")

        score = rule_score * 0.4 + scar_score * 0.4 + specificity * 0.2
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.8, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Dimension 5: Relational Integrity
# ─────────────────────────────────────────────────────────

class RelationalIntegrity(QualityDimension):
    """
    Are entity relationships preserved across the state?

    Checks artifact mention coverage, concept↔rule linkage,
    question↔goal relevance.
    """

    name = "relational_integrity"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l1 = state.get("L1", {})
        l2 = state.get("L2", [])
        l3 = state.get("L3", {})

        artifacts = set(l1.get("artifacts", []))
        l2_text = " ".join(
            str(e.get("content", "")) + " " + str(e.get("summary", ""))
            for e in l2
        )
        if artifacts:
            covered = sum(1 for a in artifacts if a in l2_text)
            coverage = covered / len(artifacts)
            evidence.append(f"{covered}/{len(artifacts)} artifacts referenced in L2")
        else:
            coverage = 1.0
            evidence.append("No artifacts to track")

        concepts = l3.get("concepts", [])
        rules = l3.get("rules", [])
        concept_names = [c.get("concept", "").lower() for c in concepts]
        rule_text = " ".join(
            f"{r.get('when', '')} {r.get('then', '')} {r.get('why', '')}" for r in rules
        ).lower()
        if concepts and rules:
            linked = sum(1 for c in concept_names if c and c in rule_text)
            concept_linkage = linked / len(concepts)
            evidence.append(f"{linked}/{len(concepts)} concepts linked to rules")
        else:
            concept_linkage = 1.0

        questions = l1.get("open_questions", [])
        goal = l1.get("goal", "").lower()
        if questions and goal:
            related = sum(
                1 for q in questions
                if any(w in q.lower() for w in goal.split() if len(w) > 3)
            )
            q_relevance = related / len(questions)
            evidence.append(f"{related}/{len(questions)} questions relate to goal")
        else:
            q_relevance = 1.0

        score = coverage * 0.4 + concept_linkage * 0.3 + q_relevance * 0.3
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.75, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Dimension 6: Multiscale Alignment
# ─────────────────────────────────────────────────────────

class MultiscaleAlignment(QualityDimension):
    """
    Consistency at micro (individual entries) and macro (overall arc) levels.
    """

    name = "multiscale_alignment"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l2 = state.get("L2", [])
        l3 = state.get("L3", {})

        if l2:
            well_formed = sum(
                1 for e in l2
                if e.get("timestamp") and (e.get("content") or e.get("summary"))
            )
            micro = well_formed / len(l2)
            evidence.append(f"{well_formed}/{len(l2)} L2 entries well-formed")
        else:
            micro = 1.0
            evidence.append("No L2 entries (micro)")

        consolidations = sum(1 for e in l2 if e.get("phase") == "consolidation")
        total = len(l2)
        if total > 5:
            macro = min(1.0, consolidations / (total / 5))
            evidence.append(f"{consolidations} consolidation points in {total} entries")
        else:
            macro = 1.0

        l3_total = len(l3.get("rules", [])) + len(l3.get("tracewisdomlog", []))
        cross = 0.6 if total > 10 and l3_total == 0 else 1.0
        if total > 10 and l3_total == 0:
            evidence.append("Heavy L2 but no L3 learning")

        score = micro * 0.4 + macro * 0.3 + cross * 0.3
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.7, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Dimension 7: Predictive Accuracy
# ─────────────────────────────────────────────────────────

class PredictiveAccuracy(QualityDimension):
    """
    Can the state make verifiable predictions?

    Checks testability of questions, rule trigger specificity,
    scar boon verifiability.
    """

    name = "predictive_accuracy"

    def compute(self, state: Dict[str, Any], model=None) -> DimensionResult:
        evidence: List[str] = []
        l1 = state.get("L1", {})
        l3 = state.get("L3", {})

        questions = l1.get("open_questions", [])
        if questions:
            markers = ("how", "what", "when", "which", "does", "can", "is", "will")
            testable = sum(
                1 for q in questions
                if any(q.lower().strip().startswith(m) for m in markers)
            )
            q_score = testable / len(questions)
            evidence.append(f"{testable}/{len(questions)} questions testable")
        else:
            q_score = 1.0

        rules = l3.get("rules", [])
        if rules:
            specific = sum(1 for r in rules if len(r.get("when", "").split()) >= 3)
            r_score = specific / len(rules)
            evidence.append(f"{specific}/{len(rules)} rules have specific triggers")
        else:
            r_score = 1.0

        scars = l3.get("tracewisdomlog", [])
        if scars:
            verifiable = sum(
                1 for s in scars if s.get("boon") and len(s.get("boon", "").split()) >= 3
            )
            s_score = verifiable / len(scars)
            evidence.append(f"{verifiable}/{len(scars)} scars have verifiable boons")
        else:
            s_score = 1.0

        score = q_score * 0.3 + r_score * 0.4 + s_score * 0.3
        return DimensionResult(
            name=self.name, score=round(score, 3), confidence=0.65, evidence=evidence
        )


# ─────────────────────────────────────────────────────────
# Evaluator (composes all 7 dimensions)
# ─────────────────────────────────────────────────────────

ALL_DIMENSIONS: List[QualityDimension] = [
    InternalConsistency(),
    ExternalCorrespondence(),
    TemporalStability(),
    CausalValidity(),
    RelationalIntegrity(),
    MultiscaleAlignment(),
    PredictiveAccuracy(),
]


@dataclass
class QualityVector:
    """Complete 7D quality measurement result."""

    dimensions: Dict[str, DimensionResult]
    aggregate_score: float
    aggregate_confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_score": self.aggregate_score,
            "aggregate_confidence": self.aggregate_confidence,
            "timestamp": self.timestamp,
            "dimensions": {
                name: {
                    "score": r.score,
                    "confidence": r.confidence,
                    "evidence": r.evidence,
                    "suggestions": r.suggestions,
                }
                for name, r in self.dimensions.items()
            },
        }

    def summary(self) -> str:
        lines = [
            f"Quality Vector ({self.aggregate_score:.2f} "
            f"@ {self.aggregate_confidence:.0%} confidence)"
        ]
        for name, r in self.dimensions.items():
            filled = int(r.score * 10)
            bar = "█" * filled + "░" * (10 - filled)
            lines.append(f"  {name:<25s} {bar} {r.score:.2f}")
        return "\n".join(lines)


class QualityEvaluator:
    """Computes the full 7D quality vector for a given state."""

    def __init__(
        self,
        dimensions: Optional[List[QualityDimension]] = None,
        model: Optional["ModelAdapter"] = None,
    ):
        self.dimensions = dimensions or list(ALL_DIMENSIONS)
        self.model = model

    def evaluate(self, state: Dict[str, Any]) -> QualityVector:
        results: Dict[str, DimensionResult] = {}
        for dim in self.dimensions:
            results[dim.name] = dim.compute(state, model=self.model)

        scores = [r.score for r in results.values()]
        confidences = [r.confidence for r in results.values()]

        return QualityVector(
            dimensions=results,
            aggregate_score=round(statistics.mean(scores), 3),
            aggregate_confidence=round(statistics.mean(confidences), 3),
        )
