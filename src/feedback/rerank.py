"""
Feedback-aware re-ranking (C5).

This is a small, deterministic, OPTIONAL personalization layer that runs
*after* SearchEngine has already produced its hybrid results. It does NOT
modify SearchEngine, the VectorStore, or the core ranking module; it only
adjusts the existing ``score`` on copies of the returned ``SearchResult``
objects and re-sorts them.

Design constraints:
- Deterministic, cheap, local, testable. No embeddings, no network.
- The existing retrieval + ranking pipeline is untouched.
- Feedback can NEVER dominate relevance: the maximum contribution is bounded
  by ``FEEDBACK_WEIGHT`` (~0.05, per SEARCH.md).
- A feedback signal is only applied when the stored feedback query has
  meaningful lexical overlap with the current query (query-relevance gate).
- Recent feedback is weighted more than old feedback (exponential decay).
- Any failure (missing store, lookup error, malformed data) falls back to
  the ORIGINAL results unchanged. Feedback must never turn /search into 500.
"""

from __future__ import annotations

import dataclasses
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.search.result import SearchResult
from src.search.query_parser import STOPWORDS


# Maximum feedback contribution to a result's score (SEARCH.md suggests w_access=0.05).
FEEDBACK_WEIGHT = 0.05

# Half-life for the time-decay of feedback influence, in days.
HALF_LIFE_DAYS = 30.0

# Relative action strengths. pin > click > ignore. ignore is negative.
ACTION_STRENGTH = {
    "pin": 1.0,
    "click": 0.5,
    "ignore": -1.0,
}


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    if not text:
        return set()
    toks = _TOKEN_RE.findall(text.lower())
    return {t for t in toks if t not in STOPWORDS}


def _overlap_coverage(current: set, stored: set) -> float:
    """Fraction of the current-query tokens covered by the stored query.

    Returns 0.0 when there is no meaningful overlap (the query-relevance gate).
    """
    if not current:
        return 0.0
    inter = current & stored
    if not inter:
        return 0.0
    return len(inter) / len(current)


def _decay(timestamp_ms: Optional[int]) -> float:
    """Deterministic exponential-style decay based on age in days."""
    if not timestamp_ms:
        return 0.0
    now = time.time()
    age_days = (now - float(timestamp_ms) / 1000.0) / 86400.0
    if age_days < 0:
        age_days = 0.0
    return 0.5 ** (age_days / HALF_LIFE_DAYS)


def _aggregate(rows: List[Dict[str, Any]], query: str) -> float:
    """Aggregate a document's feedback events into a clamped [-1, 1] boost."""
    current = _tokens(query)
    total = 0.0
    for row in rows:
        q = row.get("query") or row.get("query_text") or ""
        action = (row.get("action") or "").strip().lower()
        ts = row.get("timestamp") or 0
        strength = ACTION_STRENGTH.get(action, 0.0)
        if strength == 0.0:
            continue
        overlap = _overlap_coverage(current, _tokens(q))
        if overlap <= 0.0:
            continue
        total += strength * overlap * _decay(ts)
    # Clamp so one document's signal can never exceed [-1, 1].
    return max(-1.0, min(1.0, total))


def _copy_with_score(result: SearchResult, new_score: float) -> SearchResult:
    return dataclasses.replace(result, score=new_score)


def apply_feedback_rerank(
    results: List[SearchResult],
    feedback_store: Any,
    query: str,
) -> List[SearchResult]:
    """Return a NEW ordered list with feedback-informed score adjustments.

    Falls back to the original (copied) ordering on any feedback problem.
    """
    if feedback_store is None or not results:
        return list(results)

    try:
        doc_ids = [str(r.id) for r in results]
        try:
            feedback_map = feedback_store.get_feedback_for_documents(doc_ids)
        except Exception:
            return list(results)

        boosts: Dict[str, float] = {}
        for doc_id, rows in feedback_map.items():
            try:
                boost = _aggregate(rows, query)
            except Exception:
                boost = 0.0
            if boost != 0.0:
                boosts[doc_id] = boost

        if not boosts:
            return list(results)

        decorated: List[Tuple[SearchResult, float]] = []
        for r in results:
            boost = boosts.get(str(r.id), 0.0)
            new_score = r.score + FEEDBACK_WEIGHT * boost
            decorated.append((_copy_with_score(r, new_score), new_score))

        # Stable sort by adjusted score (descending); ties keep original order.
        decorated.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in decorated]
    except Exception:
        # Never let feedback break search.
        return list(results)
