"""
Tests for the Enterprise Readiness Index module.
"""
import pytest
from evalforge.core import TaskResult
from evalforge.eri import (
    ERIConfig, WEIGHT_PROFILES, compute_eri,
    _weighted_harmonic_mean, _bootstrap_ci,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(
    task_name: str = "test/task",
    accuracy: float = 0.8,
    latency_ms: float = 300.0,
    p95_latency_ms: float = 500.0,
    throughput_tps: float = 2000.0,
    cost_per_1k: float = 0.10,
    ece: float | None = None,
) -> TaskResult:
    scores: dict = {"accuracy": accuracy}
    if ece is not None:
        scores["ece"] = ece
    return TaskResult(
        task_name=task_name,
        scores=scores,
        ci={"accuracy": (accuracy - 0.02, accuracy + 0.02)},
        example_scores=[{"accuracy": accuracy}] * 100,
        mean_latency_ms=latency_ms,
        p95_latency_ms=p95_latency_ms,
        throughput_tokens_per_s=throughput_tps,
        cost_usd_per_1k=cost_per_1k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ERIConfig tests
# ─────────────────────────────────────────────────────────────────────────────

def test_eri_config_default_profile():
    cfg = ERIConfig()
    assert cfg.weight_profile == "balanced"
    assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9


def test_eri_config_all_profiles_valid():
    for profile in WEIGHT_PROFILES:
        cfg = ERIConfig(weight_profile=profile)
        assert abs(sum(cfg.weights.values()) - 1.0) < 1e-9


def test_eri_config_unknown_profile_raises():
    with pytest.raises(ValueError, match="Unknown weight profile"):
        ERIConfig(weight_profile="nonexistent")


def test_eri_config_custom_weights():
    w = {"accuracy": 0.4, "calibration": 0.1, "safety": 0.1,
         "latency": 0.2, "throughput": 0.1, "cost": 0.1}
    cfg = ERIConfig(weights=w)
    assert cfg.weights == w


def test_eri_config_weights_must_sum_to_one():
    w = {"accuracy": 0.5, "calibration": 0.1, "safety": 0.1,
         "latency": 0.1, "throughput": 0.1, "cost": 0.1}  # sums to 1.0 ✓
    cfg = ERIConfig(weights=w)
    assert cfg.weights == w


def test_eri_config_bad_sum_raises():
    w = {"accuracy": 0.5, "calibration": 0.1, "safety": 0.1,
         "latency": 0.1, "throughput": 0.1, "cost": 0.5}  # sums to 1.4
    with pytest.raises(ValueError, match="sum to 1"):
        ERIConfig(weights=w)


# ─────────────────────────────────────────────────────────────────────────────
# Weighted harmonic mean tests
# ─────────────────────────────────────────────────────────────────────────────

def test_harmonic_mean_equal_weights_equal_scores():
    scores = {d: 0.8 for d in ("accuracy", "calibration", "safety",
                                "latency", "throughput", "cost")}
    weights = {d: 1/6 for d in scores}
    result = _weighted_harmonic_mean(scores, weights)
    assert abs(result - 0.8) < 1e-5


def test_harmonic_mean_low_score_penalizes():
    scores = {"accuracy": 0.9, "calibration": 0.9, "safety": 0.9,
              "latency": 0.01, "throughput": 0.9, "cost": 0.9}
    weights = {d: 1/6 for d in scores}
    result = _weighted_harmonic_mean(scores, weights)
    # Should be much less than 0.9 due to latency penalty
    assert result < 0.5


def test_harmonic_mean_monotonicity():
    base = {d: 0.7 for d in ("accuracy", "calibration", "safety",
                               "latency", "throughput", "cost")}
    improved = dict(base, accuracy=0.9)
    weights = {d: 1/6 for d in base}
    assert _weighted_harmonic_mean(improved, weights) > \
           _weighted_harmonic_mean(base, weights)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI tests
# ─────────────────────────────────────────────────────────────────────────────

def test_bootstrap_ci_bounds_in_range():
    values = [0.7] * 50 + [0.9] * 50  # mean 0.8
    lower, upper = _bootstrap_ci(values, n_resamples=1000, ci_level=0.95)
    assert lower < 0.8 < upper
    assert 0.0 <= lower <= upper <= 1.0


def test_bootstrap_ci_empty_list():
    lower, upper = _bootstrap_ci([], n_resamples=100, ci_level=0.95)
    assert lower == 0.0
    assert upper == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_eri integration tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_eri_returns_score_in_range():
    results = [_make_result()]
    score = compute_eri(results)
    assert 0.0 <= score.eri <= 1.0


def test_compute_eri_high_latency_reduces_score():
    fast = compute_eri([_make_result(p95_latency_ms=100)])
    slow = compute_eri([_make_result(p95_latency_ms=5000)])
    assert fast.eri > slow.eri


def test_compute_eri_high_cost_reduces_score():
    cheap = compute_eri([_make_result(cost_per_1k=0.01)])
    expensive = compute_eri([_make_result(cost_per_1k=5.0)])
    assert cheap.eri > expensive.eri


def test_compute_eri_dimension_scores_present():
    result = compute_eri([_make_result()])
    assert set(result.dimension_scores.keys()) == {
        "accuracy", "calibration", "safety",
        "latency", "throughput", "cost"
    }


def test_compute_eri_ci_ordered():
    result = compute_eri([_make_result()])
    assert result.ci_lower <= result.eri <= result.ci_upper


def test_compute_eri_dominance():
    """Model A dominates B on all dims → ERI(A) > ERI(B)."""
    a = compute_eri([_make_result(accuracy=0.9, latency_ms=100,
                                   p95_latency_ms=200, cost_per_1k=0.05)])
    b = compute_eri([_make_result(accuracy=0.7, latency_ms=800,
                                   p95_latency_ms=1500, cost_per_1k=0.50)])
    assert a.eri > b.eri
