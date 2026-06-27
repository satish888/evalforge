"""
EvalForge — Enterprise Readiness Index (ERI)
============================================
Computes the weighted harmonic mean of six operational dimensions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from .core import TaskResult

# ─────────────────────────────────────────────────────────────────────────────
# Constants & built-in weight profiles
# ─────────────────────────────────────────────────────────────────────────────

DIMENSIONS = ("accuracy", "calibration", "safety", "latency", "throughput", "cost")
_EPSILON = 1e-6

WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "balanced": {
        "accuracy": 0.25, "calibration": 0.15, "safety": 0.15,
        "latency": 0.15, "throughput": 0.15, "cost": 0.15,
    },
    "realtime": {
        "accuracy": 0.20, "calibration": 0.10, "safety": 0.20,
        "latency": 0.30, "throughput": 0.15, "cost": 0.05,
    },
    "cost_first": {
        "accuracy": 0.20, "calibration": 0.10, "safety": 0.15,
        "latency": 0.10, "throughput": 0.10, "cost": 0.35,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ERIConfig:
    """Configuration for ERI computation."""
    weight_profile: str = "balanced"
    weights: dict[str, float] = field(default_factory=dict)

    # Normalisation thresholds
    latency_budget_ms: float = 2000.0
    throughput_target_rps: float = 50.0
    cost_budget_per_1k_usd: float = 1.0

    # Capability-dimension weights for accuracy aggregation
    capability_weights: dict[str, float] = field(default_factory=dict)

    # Bootstrap CIs
    n_bootstrap: int = 10_000
    ci_level: float = 0.95

    def __post_init__(self) -> None:
        if not self.weights:
            if self.weight_profile not in WEIGHT_PROFILES:
                raise ValueError(
                    f"Unknown weight profile '{self.weight_profile}'. "
                    f"Available: {list(WEIGHT_PROFILES)}"
                )
            self.weights = dict(WEIGHT_PROFILES[self.weight_profile])
        self._validate_weights()

    def _validate_weights(self) -> None:
        unknown = set(self.weights) - set(DIMENSIONS)
        if unknown:
            raise ValueError(
                f"Unknown dimension(s) in weights: {unknown}. "
                f"Valid dimensions: {DIMENSIONS}"
            )
        total = sum(self.weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Dimension weights must sum to 1.0; got {total:.4f}"
            )
        # Subset mode: not all six dimensions are required.
        # Missing dimensions are silently excluded from the harmonic mean.


# ─────────────────────────────────────────────────────────────────────────────
# Dimension normalisers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_accuracy(
    results: list[TaskResult],
    capability_weights: dict[str, float],
) -> float:
    """Weighted average of per-task primary metrics."""
    total_weight = 0.0
    weighted_sum = 0.0
    for r in results:
        w = capability_weights.get(r.task_name, 1.0)
        weighted_sum += w * r.scores[_primary_metric_of(r)]
        total_weight += w
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


def _primary_metric_of(result: TaskResult) -> str:
    """Return the first key in scores as primary (matches Task.primary_metric)."""
    return next(iter(result.scores))


def _normalise_calibration(results: list[TaskResult]) -> float:
    """1 - mean ECE across tasks that expose it."""
    eces = [
        1.0 - r.scores["ece"]
        for r in results
        if "ece" in r.scores
    ]
    return float(np.mean(eces)) if eces else 0.5  # neutral fallback


def _normalise_safety(results: list[TaskResult]) -> float:
    safety_results = [r for r in results if r.task_name.startswith("safety/")]
    if not safety_results:
        return 1.0  # neutral if no safety tasks configured
    return float(np.mean([
        r.scores[_primary_metric_of(r)] for r in safety_results
    ]))


def _normalise_latency(results: list[TaskResult], budget_ms: float) -> float:
    p95 = max((r.p95_latency_ms for r in results), default=0.0)
    return max(0.0, 1.0 - p95 / budget_ms)


def _normalise_throughput(results: list[TaskResult], target_rps: float) -> float:
    # throughput_tokens_per_s averaged; convert to a [0,1] score
    mean_tps = float(np.mean([r.throughput_tokens_per_s for r in results]))
    # Assume 256 tokens per request as a reasonable average output length
    mean_rps = mean_tps / 256.0
    return min(1.0, mean_rps / target_rps)


def _normalise_cost(results: list[TaskResult], budget_per_1k: float) -> float:
    mean_cost = float(np.mean([r.cost_usd_per_1k for r in results]))
    return max(0.0, 1.0 - mean_cost / budget_per_1k)


# ─────────────────────────────────────────────────────────────────────────────
# ERI computation
# ─────────────────────────────────────────────────────────────────────────────

class ERIScore(NamedTuple):
    """Result of ERI computation."""
    eri: float
    ci_lower: float
    ci_upper: float
    dimension_scores: dict[str, float]
    weight_profile: str
    weights: dict[str, float]


def compute_eri(
    results: list[TaskResult],
    config: ERIConfig | None = None,
) -> ERIScore:
    """
    Compute the Enterprise Readiness Index from a list of TaskResults.

    Parameters
    ----------
    results:
        Task-level results produced by EvalRunner.
    config:
        ERI configuration; uses balanced defaults if None.

    Returns
    -------
    ERIScore
        Named tuple with ERI, CI, and per-dimension breakdowns.
    """
    if config is None:
        config = ERIConfig()

    dim_scores = {
        "accuracy": _normalise_accuracy(results, config.capability_weights),
        "calibration": _normalise_calibration(results),
        "safety": _normalise_safety(results),
        "latency": _normalise_latency(results, config.latency_budget_ms),
        "throughput": _normalise_throughput(results, config.throughput_target_rps),
        "cost": _normalise_cost(results, config.cost_budget_per_1k_usd),
    }

    eri = _weighted_harmonic_mean(dim_scores, config.weights)

    # Bootstrap CI over example-level scores for accuracy dimension only.
    # A full multi-dimension bootstrap requires raw per-example operational
    # data; for simplicity we propagate accuracy CI scaled by the
    # harmonic sensitivity derivative.
    acc_scores_flat = [
        s[_primary_metric_of(r)]
        for r in results
        for s in r.example_scores
    ]
    ci_lower, ci_upper = _bootstrap_ci(
        acc_scores_flat,
        config.n_bootstrap,
        config.ci_level,
    )
    # Scale CI to ERI space (first-order approximation)
    delta = abs(eri - dim_scores["accuracy"]) / (dim_scores["accuracy"] + _EPSILON)
    ci_lower_eri = max(0.0, eri - delta * (dim_scores["accuracy"] - ci_lower))
    ci_upper_eri = min(1.0, eri + delta * (ci_upper - dim_scores["accuracy"]))
    # Guarantee ordering is preserved despite floating-point rounding
    ci_lower_eri = min(ci_lower_eri, eri)
    ci_upper_eri = max(ci_upper_eri, eri)

    return ERIScore(
        eri=eri,
        ci_lower=ci_lower_eri,
        ci_upper=ci_upper_eri,
        dimension_scores=dim_scores,
        weight_profile=config.weight_profile,
        weights=dict(config.weights),
    )


def _weighted_harmonic_mean(
    scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Compute the weighted harmonic mean over whichever dimensions are in weights.

    Supports subset mode: if weights only covers {accuracy, latency, cost},
    only those three enter the harmonic mean.  Weights must still sum to 1.0.
    """
    denom = sum(
        weights[dim] / (scores[dim] + _EPSILON)
        for dim in weights          # only active dimensions
    )
    return 1.0 / denom


def compute_eri_from_scores(
    dimension_scores: dict[str, float],
    weights: dict[str, float],
) -> float:
    """
    Compute ERI directly from pre-normalised dimension scores.

    This is the entry point used by the paper's case study and
    ``compute_harmonic_eri.py``.  It supports subset mode — only the
    dimensions present in ``weights`` are included in the harmonic mean.
    Weights must sum to 1.0 and all dimension names must be in DIMENSIONS.

    Parameters
    ----------
    dimension_scores:
        Mapping of dimension name -> normalised score in [0, 1].
        Must contain a key for every dimension listed in ``weights``.
    weights:
        Mapping of dimension name -> weight.  Need not cover all six
        DIMENSIONS; missing ones are excluded from the harmonic mean.

    Returns
    -------
    float
        The weighted harmonic ERI in [0, 1].

    Examples
    --------
    >>> from evalforge.eri import compute_eri_from_scores
    >>> compute_eri_from_scores(
    ...     {"accuracy": 0.77, "latency": 0.554, "cost": 1.0},
    ...     {"accuracy": 0.40, "latency": 0.40, "cost": 0.20},
    ... )
    0.693  # Mistral-Small-3.1 result from paper
    """
    # Re-use ERIConfig validation (checks unknown dims + weight sum)
    cfg = ERIConfig(weights=dict(weights))
    missing = set(cfg.weights) - set(dimension_scores)
    if missing:
        raise ValueError(
            f"dimension_scores is missing keys required by weights: {missing}"
        )
    return _weighted_harmonic_mean(dimension_scores, cfg.weights)


def _bootstrap_ci(
    values: list[float],
    n_resamples: int,
    ci_level: float,
) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean of `values`."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values, dtype=np.float64)
    rng = np.random.default_rng(seed=42)
    indices = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    boot_means = arr[indices].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1.0 - alpha))
    return lower, upper
