"""
Tests for the C5 feedback re-ranking component.

These are UNIT tests: they exercise apply_feedback_rerank() directly with fake
feedback stores, plus deterministic decay / aggregation / boundedness checks.
They must NOT touch the network, the server, or any protected core module.
"""

import time

import pytest

from src.feedback.rerank import (
    apply_feedback_rerank,
    FEEDBACK_WEIGHT,
    HALF_LIFE_DAYS,
    _decay,
)
from src.search.result import SearchResult


def make_result(rid, score, file_name="f.txt"):
    return SearchResult(
        id=rid,
        file_name=file_name,
        path="/x/" + file_name,
        mime_type="text/plain",
        size=10,
        modified_at="2024-01-01T00:00:00",
        rank=0.0,
        snippet="",
        episodes=[],
        memories=[],
        evidence=[],
        result_type="artifact",
        score=score,
        why="",
    )


def now_ms():
    return int(time.time() * 1000)


class FakeFeedbackStore:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_feedback_for_documents(self, document_ids):
        out = {}
        for d in document_ids:
            if d in self.mapping:
                out[d] = self.mapping[d]
        return out


class RaisingFeedbackStore:
    def get_feedback_for_documents(self, document_ids):
        raise RuntimeError("feedback store exploded")


# A. No feedback -> order and scores unchanged.
def test_no_feedback_leaves_results_unchanged():
    results = [make_result(1, 0.80), make_result(2, 0.90)]
    out = apply_feedback_rerank(results, FakeFeedbackStore({}), "find qdrant pdf")
    assert [r.id for r in out] == [1, 2]
    assert out[0].score == 0.80
    assert out[1].score == 0.90


# B. feedback_store=None -> unchanged.
def test_none_store_leaves_results_unchanged():
    results = [make_result(1, 0.80), make_result(2, 0.90)]
    out = apply_feedback_rerank(results, None, "find qdrant pdf")
    assert [r.id for r in out] == [1, 2]


# C. Pin (relevant query) lifts a doc above a slightly higher-scored peer.
def test_pin_relevant_boosts():
    results = [make_result(1, 0.80), make_result(2, 0.82)]
    store = FakeFeedbackStore({
        "1": [{"query": "find qdrant pdf", "action": "pin", "timestamp": now_ms()}],
    })
    out = apply_feedback_rerank(results, store, "find qdrant pdf")
    assert [r.id for r in out] == [1, 2]
    assert out[0].score > out[1].score


# D. Click (relevant query) gives a smaller positive boost.
def test_click_relevant_boosts():
    results = [make_result(1, 0.80), make_result(2, 0.82)]
    store = FakeFeedbackStore({
        "1": [{"query": "find qdrant pdf", "action": "click", "timestamp": now_ms()}],
    })
    out = apply_feedback_rerank(results, store, "find qdrant pdf")
    assert [r.id for r in out] == [1, 2]
    assert out[0].score > out[1].score


# E. Ignore (relevant query) pushes a doc below a slightly lower-scored peer.
def test_ignore_relevant_penalizes():
    results = [make_result(1, 0.83), make_result(2, 0.80)]
    store = FakeFeedbackStore({
        "1": [{"query": "find qdrant pdf", "action": "ignore", "timestamp": now_ms()}],
    })
    out = apply_feedback_rerank(results, store, "find qdrant pdf")
    assert [r.id for r in out] == [2, 1]


# F. Query-relevance gate: unrelated stored query contributes zero.
def test_unrelated_query_no_boost():
    results = [make_result(1, 0.80), make_result(2, 0.82)]
    store = FakeFeedbackStore({
        "1": [{"query": "zzz unrelated qwerty words", "action": "pin", "timestamp": now_ms()}],
    })
    out = apply_feedback_rerank(results, store, "find qdrant pdf")
    assert [r.id for r in out] == [1, 2]
    assert out[0].score == 0.80


# G. Time decay: a recent pin outweighs an old pin.
def test_time_decay_recent_stronger():
    base = 0.50
    recent_ts = int((time.time() - 1 * 86400) * 1000)
    old_ts = int((time.time() - 200 * 86400) * 1000)
    recent_out = apply_feedback_rerank(
        [make_result(1, base)],
        FakeFeedbackStore({"1": [{"query": "q", "action": "pin", "timestamp": recent_ts}]}),
        "q",
    )
    old_out = apply_feedback_rerank(
        [make_result(1, base)],
        FakeFeedbackStore({"1": [{"query": "q", "action": "pin", "timestamp": old_ts}]}),
        "q",
    )
    assert recent_out[0].score > old_out[0].score


# H. Multiple signals aggregate deterministically and clamp to [-1, 1].
def test_multiple_signals_clamped():
    rows = [{"query": "q", "action": "pin", "timestamp": now_ms()} for _ in range(5)]
    out = apply_feedback_rerank(
        [make_result(1, 0.50)],
        FakeFeedbackStore({"1": rows}),
        "q",
    )
    # 5 pins would sum to >1; signal is clamped to 1.0 -> max contribution 0.05.
    assert out[0].score == pytest.approx(0.50 + FEEDBACK_WEIGHT, abs=1e-6)
    # Deterministic.
    out2 = apply_feedback_rerank(
        [make_result(1, 0.50)],
        FakeFeedbackStore({"1": rows}),
        "q",
    )
    assert out2[0].score == out[0].score


# I. Boundedness: feedback contribution never exceeds FEEDBACK_WEIGHT.
def test_feedback_contribution_bounded():
    rows = [{"query": "q", "action": "pin", "timestamp": now_ms()} for _ in range(10)]
    out = apply_feedback_rerank(
        [make_result(1, 0.50)],
        FakeFeedbackStore({"1": rows}),
        "q",
    )
    contribution = out[0].score - 0.50
    assert contribution <= FEEDBACK_WEIGHT + 1e-9
    # Feedback cannot dominate: adjusted score is base + at most FEEDBACK_WEIGHT.
    assert out[0].score <= 0.50 + FEEDBACK_WEIGHT + 1e-9


# J. Failure safety: a raising store returns the original results.
def test_failure_returns_original():
    results = [make_result(1, 0.80), make_result(2, 0.90)]
    out = apply_feedback_rerank(results, RaisingFeedbackStore(), "q")
    assert [r.id for r in out] == [1, 2]
    assert out[0].score == 0.80
    assert out[1].score == 0.90


# K. No mutation of the original list or its SearchResult objects.
def test_no_mutation_of_input():
    results = [make_result(1, 0.80), make_result(2, 0.90)]
    orig_order = [r.id for r in results]
    orig_scores = [r.score for r in results]
    store = FakeFeedbackStore({
        "1": [{"query": "q", "action": "pin", "timestamp": now_ms()}],
    })
    out = apply_feedback_rerank(results, store, "q")
    # Input list order and object scores are untouched.
    assert [r.id for r in results] == orig_order
    assert [r.score for r in results] == orig_scores
    for r, s in zip(results, orig_scores):
        assert r.score == s
    # Output is a different ordered list with adjusted (copied) objects.
    assert [r.id for r in out] != orig_order or any(
        o.score != s for o, s in zip(out, orig_scores)
    )


# L. Decay helper sanity (pure function, within (0, 1] for non-negative ages).
def test_decay_helper():
    now = int(time.time() * 1000)
    assert _decay(now) == pytest.approx(1.0, abs=1e-6)  # age ~0 -> full weight
    recent = _decay(int((time.time() - 1 * 86400) * 1000))
    old = _decay(int((time.time() - 100 * 86400) * 1000))
    assert 0.0 < old < recent <= 1.0
    assert recent > old
    # Half-life is respected approximately.
    half = _decay(int((time.time() - HALF_LIFE_DAYS * 86400) * 1000))
    assert half == pytest.approx(0.5, abs=1e-6)
