"""
Tests for ContaminationDetector.
"""
import pytest
from evalforge.contamination import (
    BloomFilter, ContaminationConfig, ContaminationDetector,
    extract_ngrams,
)


# ─────────────────────────────────────────────────────────────────────────────
# BloomFilter tests
# ─────────────────────────────────────────────────────────────────────────────

def test_bloom_add_and_contains():
    bf = BloomFilter(expected_items=1000, false_positive_rate=1e-4)
    bf.add("hello world")
    assert "hello world" in bf


def test_bloom_not_contains():
    bf = BloomFilter(expected_items=1000, false_positive_rate=1e-4)
    bf.add("hello world")
    # "foo bar" is almost certainly not in the filter
    assert "foo bar baz qux quux" not in bf


def test_bloom_unicode():
    bf = BloomFilter(expected_items=100, false_positive_rate=1e-4)
    bf.add("こんにちは世界")
    assert "こんにちは世界" in bf


# ─────────────────────────────────────────────────────────────────────────────
# N-gram extraction tests
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_ngrams_basic():
    ngrams = extract_ngrams("the quick brown fox", n=2)
    assert "the quick" in ngrams
    assert "quick brown" in ngrams
    assert "brown fox" in ngrams
    assert len(ngrams) == 3


def test_extract_ngrams_short_text():
    ngrams = extract_ngrams("hello", n=3)
    assert ngrams == ["hello"]


def test_extract_ngrams_empty():
    ngrams = extract_ngrams("", n=3)
    assert ngrams == []


def test_extract_ngrams_lowercases():
    ngrams = extract_ngrams("The QUICK Brown", n=2)
    assert "the quick" in ngrams


def test_extract_ngrams_strips_punctuation():
    ngrams = extract_ngrams("hello, world!", n=2)
    assert "hello world" in ngrams


# ─────────────────────────────────────────────────────────────────────────────
# ContaminationDetector tests
# ─────────────────────────────────────────────────────────────────────────────

def test_detector_requires_index():
    detector = ContaminationDetector()
    with pytest.raises(RuntimeError, match="index_corpus"):
        detector.check_task("task", ["answer"])


def test_detector_no_contamination():
    cfg = ContaminationConfig(n=3)
    detector = ContaminationDetector(cfg)
    detector.index_corpus(["totally unrelated training text here"])
    report = detector.check_task("test/task", ["the correct answer is yes"])
    # Low overlap expected
    assert report.mean_score < 0.5
    assert not report.task_flagged


def test_detector_high_contamination():
    cfg = ContaminationConfig(
        n=3,
        flag_threshold_score=0.3,
        flag_threshold_frac=0.1,
    )
    detector = ContaminationDetector(cfg)
    answer = "the capital of france is paris"
    # Index the exact answer
    detector.index_corpus([answer])
    report = detector.check_task("test/task", [answer] * 10)
    assert report.mean_score > 0.5
    assert report.task_flagged


def test_detector_adjusted_score_unflagged():
    cfg = ContaminationConfig(n=3)
    detector = ContaminationDetector(cfg)
    detector.index_corpus(["something unrelated"])
    report = detector.check_task("task", ["totally different text here now"])
    adjusted = detector.adjusted_score(0.85, report)
    assert adjusted == pytest.approx(0.85)


def test_detector_adjusted_score_flagged():
    cfg = ContaminationConfig(
        n=3,
        flag_threshold_score=0.3,
        flag_threshold_frac=0.1,
    )
    detector = ContaminationDetector(cfg)
    answer = "the quick brown fox jumps"
    detector.index_corpus([answer])
    report = detector.check_task("task", [answer] * 10)
    adjusted = detector.adjusted_score(0.90, report)
    # Adjusted score must be < raw score when flagged
    assert adjusted < 0.90


def test_detector_empty_answers():
    detector = ContaminationDetector()
    detector.index_corpus(["some text"])
    report = detector.check_task("task", [])
    assert report.mean_score == 0.0
    assert not report.task_flagged
