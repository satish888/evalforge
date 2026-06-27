"""
EvalForge — Benchmark Contamination Detector
=============================================
Uses Bloom-filtered n-gram overlap to flag evaluation tasks whose
answer spans appear in a supplied training corpus.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

# ─────────────────────────────────────────────────────────────────────────────
# Tiny Bloom filter (no external deps)
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import math


class BloomFilter:
    """
    Space-efficient probabilistic set for n-gram membership queries.

    False positive rate: (1 - e^(-k*n/m))^k ≈ target_fpr.
    """

    def __init__(
        self,
        expected_items: int = 10_000_000,
        false_positive_rate: float = 1e-6,
    ) -> None:
        self._m = self._optimal_m(expected_items, false_positive_rate)
        self._k = self._optimal_k(self._m, expected_items)
        self._bits = bytearray(math.ceil(self._m / 8))

    # ── Public API ────────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        for bit in self._hash_bits(item):
            self._bits[bit >> 3] |= 1 << (bit & 7)

    def __contains__(self, item: str) -> bool:
        return all(
            self._bits[bit >> 3] & (1 << (bit & 7))
            for bit in self._hash_bits(item)
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _hash_bits(self, item: str) -> list[int]:
        """Return k bit positions for a given item."""
        encoded = item.encode("utf-8")
        positions = []
        for seed in range(self._k):
            digest = hashlib.sha256(
                seed.to_bytes(4, "little") + encoded
            ).digest()
            bit = int.from_bytes(digest[:8], "little") % self._m
            positions.append(bit)
        return positions

    @staticmethod
    def _optimal_m(n: int, p: float) -> int:
        return max(1, -int(n * math.log(p) / (math.log(2) ** 2)))

    @staticmethod
    def _optimal_k(m: int, n: int) -> int:
        return max(1, round((m / n) * math.log(2)))


# ─────────────────────────────────────────────────────────────────────────────
# N-gram extraction
# ─────────────────────────────────────────────────────────────────────────────

_WHITESPACE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    # Filter empty strings that arise from leading/trailing whitespace
    return [t for t in _WHITESPACE.split(text.strip()) if t]


def extract_ngrams(text: str, n: int) -> list[str]:
    """Return all n-grams of a tokenized text as space-joined strings."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# Contamination detector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ContaminationConfig:
    n: int = 13                        # n-gram length (Lee et al. 2022)
    flag_threshold_score: float = 0.5  # per-example contamination score cutoff
    flag_threshold_frac: float = 0.2   # fraction of examples that triggers task flag
    expected_corpus_ngrams: int = 200_000_000
    bloom_fpr: float = 1e-6


@dataclass
class ContaminationReport:
    task_name: str
    example_scores: list[float]        # per-example contamination score
    mean_score: float
    flagged_fraction: float
    task_flagged: bool
    config: ContaminationConfig = field(repr=False)


class ContaminationDetector:
    """
    Detects n-gram overlap between evaluation answers and a training corpus.

    Workflow::

        detector = ContaminationDetector()
        detector.index_corpus(corpus_texts)          # one-time indexing
        report = detector.check_task(task_name, [(answer, ...), ...])

    Parameters
    ----------
    config:
        Contamination detection configuration.
    """

    def __init__(self, config: ContaminationConfig | None = None) -> None:
        self.config = config or ContaminationConfig()
        self._bloom = BloomFilter(
            expected_items=self.config.expected_corpus_ngrams,
            false_positive_rate=self.config.bloom_fpr,
        )
        self._indexed = False

    # ── Corpus indexing ───────────────────────────────────────────────────

    def index_corpus(self, texts: Iterable[str]) -> None:
        """
        Index all n-grams from the training corpus into the Bloom filter.

        Call once before running check_task(); the index is in-memory.
        """
        for text in texts:
            for ngram in extract_ngrams(text, self.config.n):
                self._bloom.add(ngram)
        self._indexed = True

    # ── Task contamination check ──────────────────────────────────────────

    def check_task(
        self,
        task_name: str,
        answers: list[str],
    ) -> ContaminationReport:
        """
        Compute contamination scores for each answer and produce a report.

        Parameters
        ----------
        task_name:
            Name of the task being checked (for reporting).
        answers:
            List of reference answer strings, one per evaluation example.

        Returns
        -------
        ContaminationReport
            Per-example scores and a task-level flagging decision.
        """
        if not self._indexed:
            raise RuntimeError(
                "Call index_corpus() before check_task()."
            )
        cfg = self.config
        example_scores: list[float] = []

        for answer in answers:
            ngrams = extract_ngrams(answer, cfg.n)
            if not ngrams:
                example_scores.append(0.0)
                continue
            hits = sum(1 for ng in ngrams if ng in self._bloom)
            example_scores.append(hits / len(ngrams))

        mean_score = sum(example_scores) / len(example_scores) if example_scores else 0.0
        flagged_fraction = sum(
            1 for s in example_scores if s > cfg.flag_threshold_score
        ) / max(len(example_scores), 1)
        task_flagged = flagged_fraction > cfg.flag_threshold_frac

        return ContaminationReport(
            task_name=task_name,
            example_scores=example_scores,
            mean_score=mean_score,
            flagged_fraction=flagged_fraction,
            task_flagged=task_flagged,
            config=cfg,
        )

    def adjusted_score(
        self,
        raw_score: float,
        report: ContaminationReport,
    ) -> float:
        """
        Return a contamination-adjusted score.

        Conservative heuristic: reduce raw score proportionally to the
        estimated fraction of examples that may have been memorized.
        """
        if not report.task_flagged:
            return raw_score
        # discount by the flagged fraction
        return raw_score * (1.0 - report.flagged_fraction)
