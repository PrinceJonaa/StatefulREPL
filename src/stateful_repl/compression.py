"""Context compression utilities for Phase 4.

Provides an extractive compression strategy that keeps the most
informative sentences based on term frequency scoring.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class CompressionResult:
    """Result of a compression operation.

    Attributes:
        original_tokens: Estimated token count before compression.
        compressed_tokens: Estimated token count after compression.
        compression_ratio: $compressed/original$ in range $[0, 1]$.
        compressed_text: Final compressed output.
        selected_units: Sentences selected by the compressor.
    """

    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compressed_text: str
    selected_units: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result to dictionary."""
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_ratio": round(self.compression_ratio, 3),
            "compressed_text": self.compressed_text,
            "selected_units": self.selected_units,
        }


@dataclass
class CompressionQuality:
    """Quality metrics for compressed text fidelity."""

    retention_score: float
    anchor_coverage: float
    missing_anchors: list[str]
    required_term_coverage: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "retention_score": round(self.retention_score, 3),
            "anchor_coverage": round(self.anchor_coverage, 3),
            "missing_anchors": self.missing_anchors,
            "required_term_coverage": round(self.required_term_coverage, 3),
        }


class ExtractiveCompressor:
    """Simple extractive context compressor.

    CLAIM-401: "Extractive sentence scoring preserves key context while
    reducing size for orchestration prompts." [scope: module] [confidence: 0.82]
    [falsifies: "Compressed output misses critical entities in benchmark tasks"]
    """

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on",
        "for", "with", "as", "is", "are", "was", "were", "be", "this", "that",
        "it", "at", "by", "from", "we", "you", "they", "he", "she",
    }

    def __init__(self, default_target_ratio: float = 0.3, max_sentences: int = 20):
        if not 0.05 <= default_target_ratio <= 1.0:
            raise ValueError(
                f"Invalid default_target_ratio {default_target_ratio}: must be 0.05-1.0"
            )
        self.default_target_ratio = default_target_ratio
        self.max_sentences = max_sentences

    def compress_text(self, text: str, target_ratio: float | None = None) -> CompressionResult:
        """Compress free-form text.

        Args:
            text: Input text to compress.
            target_ratio: Target output ratio in range $[0.05, 1.0]$.

        Returns:
            Compression result with selected sentences.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        ratio = self.default_target_ratio if target_ratio is None else target_ratio
        if not 0.05 <= ratio <= 1.0:
            raise ValueError(f"Invalid target_ratio {ratio}: must be 0.05-1.0")

        normalized = text.strip()
        if not normalized:
            return CompressionResult(0, 0, 0.0, "", [])

        sentences = self._split_sentences(normalized)
        if len(sentences) <= 1:
            tokens = self._estimate_tokens(normalized)
            return CompressionResult(tokens, tokens, 1.0, normalized, sentences)

        selected = self._select_sentences(sentences, ratio)
        compressed_text = " ".join(selected).strip()

        original_tokens = self._estimate_tokens(normalized)
        compressed_tokens = self._estimate_tokens(compressed_text)
        result_ratio = compressed_tokens / max(original_tokens, 1)

        return CompressionResult(
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=result_ratio,
            compressed_text=compressed_text,
            selected_units=selected,
        )

    def compress_state(self, state: dict[str, Any], target_ratio: float | None = None) -> CompressionResult:
        """Compress a Loom state into a compact context payload.

        Args:
            state: Full state dictionary (L1/L2/L3).
            target_ratio: Optional compression ratio override.

        Returns:
            Compression result.
        """
        l1 = state.get("L1", {})
        l2 = state.get("L2", [])

        chunks: list[str] = []
        goal = str(l1.get("goal", "")).strip()
        if goal:
            chunks.append(f"Goal: {goal}.")

        constraints = l1.get("constraints", [])
        if constraints:
            chunks.append("Constraints: " + "; ".join(str(c) for c in constraints) + ".")

        artifacts = l1.get("artifacts", [])
        if artifacts:
            chunks.append("Artifacts: " + ", ".join(str(a) for a in artifacts) + ".")

        open_questions = l1.get("open_questions", [])
        if open_questions:
            chunks.append("Open questions: " + "; ".join(str(q) for q in open_questions) + ".")

        for entry in l2[-30:]:
            summary = entry.get("summary") or entry.get("content")
            if summary:
                chunks.append(str(summary))

        result = self.compress_text("\n".join(chunks), target_ratio=target_ratio)

        # Preserve explicit goal context if it was present in input.
        if goal and "goal:" not in result.compressed_text.lower():
            injected = f"Goal: {goal}. {result.compressed_text}".strip()
            result.compressed_text = injected
            result.selected_units = [f"Goal: {goal}."] + result.selected_units
            result.compressed_tokens = self._estimate_tokens(injected)
            result.compression_ratio = result.compressed_tokens / max(result.original_tokens, 1)

        return result

    def evaluate_retention(
        self,
        original_text: str,
        compressed_text: str,
        required_terms: list[str] | None = None,
    ) -> CompressionQuality:
        """Evaluate fidelity of compressed text.

        Args:
            original_text: Source text before compression.
            compressed_text: Output text after compression.
            required_terms: Optional required terms (case-insensitive).

        Returns:
            Compression quality metrics.
        """
        original = original_text.lower()
        compressed = compressed_text.lower()

        anchors = self._extract_anchors(original)
        missing_anchors = [a for a in anchors if a not in compressed]
        anchor_coverage = 1.0 if not anchors else 1.0 - (len(missing_anchors) / len(anchors))

        terms = [t.strip().lower() for t in (required_terms or []) if t.strip()]
        if not terms:
            terms = self._top_terms(original_text, limit=8)
        covered_terms = [t for t in terms if t in compressed]
        required_term_coverage = 1.0 if not terms else len(covered_terms) / len(terms)

        retention_score = max(0.0, min(1.0, 0.6 * anchor_coverage + 0.4 * required_term_coverage))
        return CompressionQuality(
            retention_score=retention_score,
            anchor_coverage=anchor_coverage,
            missing_anchors=missing_anchors,
            required_term_coverage=required_term_coverage,
        )

    def _split_sentences(self, text: str) -> list[str]:
        candidates = re.split(r"(?<=[.!?])\s+", text)
        return [c.strip() for c in candidates if c.strip()]

    def _select_sentences(self, sentences: list[str], target_ratio: float) -> list[str]:
        tokenized = [self._tokenize(s) for s in sentences]
        corpus_tokens = [tok for sent in tokenized for tok in sent]
        tf = Counter(corpus_tokens)

        if not tf:
            keep = max(1, min(len(sentences), math.ceil(len(sentences) * target_ratio)))
            return sentences[:keep]

        # Score each sentence by informative token frequency.
        scored: list[tuple[float, int]] = []
        for idx, tokens in enumerate(tokenized):
            if not tokens:
                scored.append((0.0, idx))
                continue
            score = sum(tf[t] for t in tokens) / len(tokens)
            scored.append((score, idx))

        scored.sort(reverse=True)
        keep_count = max(1, min(len(sentences), math.ceil(len(sentences) * target_ratio)))
        keep_count = min(keep_count, self.max_sentences)
        chosen_indices = sorted(idx for _, idx in scored[:keep_count])
        return [sentences[i] for i in chosen_indices]

    def _tokenize(self, text: str) -> list[str]:
        words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [w for w in words if w not in self._STOPWORDS and len(w) > 2]

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text.split())) if text.strip() else 0

    def _extract_anchors(self, text: str) -> list[str]:
        anchors: list[str] = []
        for marker in ("goal:", "constraints:", "artifacts:", "open questions:"):
            if marker in text:
                anchors.append(marker)
        return anchors

    def _top_terms(self, text: str, limit: int = 8) -> list[str]:
        tokens = self._tokenize(text)
        if not tokens:
            return []
        counts = Counter(tokens)
        return [term for term, _ in counts.most_common(limit)]
