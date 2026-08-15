"""
Tests for C7 structured search filters in SearchEngine.

Covers MIME (exact + major-type + multiple), date_from/date_to (inclusive),
combinations, source-agnostic filtering (semantic/episode-shaped candidates),
malformed-timestamp safety, ignored/unsupported keys, and the guarantee that
the original candidate collection is never mutated.

The helper is source-agnostic, so semantic/episode candidate shapes are
exercised directly. An end-to-end `engine.search(...)` test confirms the
filter actually runs inside the existing retrieval -> filter -> rank flow.
"""

import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.search.engine import SearchEngine
from src.search.result import SearchResult


def iso_ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


# Representative candidate rows (same shape as rows returned by the artifact
# store for the FTS path, and later decorated by enrichment/ranking).
CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": 1,
        "file_name": "a.pdf",
        "path": "/a.pdf",
        "mime_type": "application/pdf",
        "size": 10,
        "modified_at": "2024-03-10T12:00:00+00:00",
        "rank": 0.0,
        "snippet": "",
    },
    {
        "id": 2,
        "file_name": "b.png",
        "path": "/b.png",
        "mime_type": "image/png",
        "size": 10,
        "modified_at": "2024-06-15T12:00:00+00:00",
        "rank": 0.0,
        "snippet": "",
    },
    {
        "id": 3,
        "file_name": "c.txt",
        "path": "/c.txt",
        "mime_type": "text/plain",
        "size": 10,
        "modified_at": "2024-09-20T12:00:00+00:00",
        "rank": 0.0,
        "snippet": "",
    },
    {
        "id": 4,
        "file_name": "d.pdf",
        "path": "/d.pdf",
        "mime_type": "application/pdf",
        "size": 10,
        "modified_at": "2024-12-01T12:00:00+00:00",
        "rank": 0.0,
        "snippet": "",
    },
]


def ids_of(result: List[Dict[str, Any]]) -> List[int]:
    return [c["id"] for c in result]


def apply(filters):
    return SearchEngine._apply_structured_filters(
        [dict(c) for c in CANDIDATES], filters
    )


# 1. no filters -> unchanged
def test_no_filters_returns_all():
    assert ids_of(apply(None)) == [1, 2, 3, 4]


# 2. empty filters -> unchanged
def test_empty_filters_returns_all():
    assert ids_of(apply({})) == [1, 2, 3, 4]


# 3. exact MIME match
def test_exact_mime_match():
    assert ids_of(apply({"mime_types": ["application/pdf"]})) == [1, 4]


# 4. major MIME match
def test_major_mime_match():
    assert ids_of(apply({"mime_types": ["image"]})) == [2]
    assert ids_of(apply({"mime_types": ["application"]})) == [1, 4]


# 5. multiple MIME types
def test_multiple_mime_types():
    assert ids_of(apply({"mime_types": ["application/pdf", "image/png"]})) == [1, 2, 4]


# 6. date_from (inclusive)
def test_date_from():
    assert ids_of(apply({"date_from": iso_ms(2024, 6, 1)})) == [2, 3, 4]


# 7. date_to (inclusive)
def test_date_to():
    assert ids_of(apply({"date_to": iso_ms(2024, 6, 30)})) == [1, 2]


# 8. both date bounds (inclusive)
def test_date_both_bounds():
    assert ids_of(
        apply({"date_from": iso_ms(2024, 6, 1), "date_to": iso_ms(2024, 9, 30)})
    ) == [2, 3]


# 9. MIME + date combination
def test_mime_and_date_combination():
    assert ids_of(
        apply({"mime_types": ["application/pdf"], "date_to": iso_ms(2024, 6, 30)})
    ) == [1]


# 10. semantic-shaped candidates are filtered
def test_semantic_candidates_filtered():
    semantic = [
        {
            "id": 10,
            "file_name": "shot.png",
            "path": "/shot.png",
            "mime_type": "image/png",
            "size": 10,
            "modified_at": "2024-07-01T00:00:00+00:00",
            "rank": 0.0,
            "snippet": "ocr text",
            "semantic_score": 0.9,
        },
        {
            "id": 11,
            "file_name": "note.pdf",
            "path": "/note.pdf",
            "mime_type": "application/pdf",
            "size": 10,
            "modified_at": "2024-07-02T00:00:00+00:00",
            "rank": 0.0,
            "snippet": "doc",
            "semantic_score": 0.8,
        },
    ]
    result = SearchEngine._apply_structured_filters(
        semantic, {"mime_types": ["image/png"]}
    )
    assert ids_of(result) == [10]


# 11. episode-shaped candidates are filtered
def test_episode_candidates_filtered():
    episode = [
        {
            "id": 20,
            "file_name": "e.pdf",
            "path": "/e.pdf",
            "mime_type": "application/pdf",
            "size": 10,
            "modified_at": "2024-08-01T00:00:00+00:00",
            "rank": 0.0,
            "snippet": "",
            "episodes": [{"id": 1, "title": " Episode"}],
        },
        {
            "id": 21,
            "file_name": "f.txt",
            "path": "/f.txt",
            "mime_type": "text/plain",
            "size": 10,
            "modified_at": "2024-08-02T00:00:00+00:00",
            "rank": 0.0,
            "snippet": "",
            "episodes": [{"id": 2, "title": " Episode"}],
        },
    ]
    result = SearchEngine._apply_structured_filters(
        episode, {"date_from": iso_ms(2024, 8, 2)}
    )
    assert ids_of(result) == [21]


# 12. malformed timestamp handled safely (excluded when a date bound is active)
def test_malformed_timestamp_excluded_when_date_filter_active():
    mixed = [
        {
            "id": 30,
            "file_name": "good.pdf",
            "path": "/good.pdf",
            "mime_type": "application/pdf",
            "size": 10,
            "modified_at": "2024-05-01T00:00:00+00:00",
            "rank": 0.0,
            "snippet": "",
        },
        {
            "id": 31,
            "file_name": "bad.pdf",
            "path": "/bad.pdf",
            "mime_type": "application/pdf",
            "size": 10,
            "modified_at": "not-a-real-timestamp",
            "rank": 0.0,
            "snippet": "",
        },
    ]
    result = SearchEngine._apply_structured_filters(
        mixed, {"date_from": iso_ms(2024, 1, 1)}
    )
    # The malformed row is excluded; the valid row passes the lower bound.
    assert ids_of(result) == [30]


# 13. app_origin / project ignored (API compatibility)
def test_unsupported_keys_ignored():
    # Only unsupported keys -> no filtering.
    assert ids_of(apply({"app_origin": "x", "project": "y"})) == [1, 2, 3, 4]
    # Unsupported keys combined with a supported key -> supported key still applies.
    assert ids_of(apply({"app_origin": "x", "mime_types": ["image/png"]})) == [2]


# 14. original input is not mutated
def test_original_input_not_mutated():
    original = [dict(c) for c in CANDIDATES]
    snapshot_keys = [{k: v for k, v in c.items()} for c in original]
    _ = SearchEngine._apply_structured_filters(original, {"mime_types": ["application/pdf"]})
    # The list and each dict are unchanged (no new keys, same values).
    assert len(original) == len(snapshot_keys)
    for before, after in zip(snapshot_keys, original):
        assert before == after


class _FakeStore:
    """Minimal artifact store returning canned rows for the FTS path only.

    Episodes/memory/vector/embedding are left None so only the FTS candidate
    path runs; this exercises the filter inside engine.search() end-to-end.
    """

    def __init__(self, rows):
        self._rows = rows

    def search_artifacts(self, query, limit=20, mime_type_filter=None):
        return [dict(r) for r in self._rows]

    def get_artifact(self, art_id):
        for r in self._rows:
            if r["id"] == art_id:
                return r
        return None


def _search(filters):
    store = _FakeStore(CANDIDATES)
    engine = SearchEngine(store)
    results = engine.search("anything", limit=10, filters=filters)
    return [r["mime_type"] for r in results], [r["id"] for r in results]


def test_search_applies_mime_filter_end_to_end():
    mimes, ids = _search({"mime_types": ["application/pdf"]})
    assert mimes == ["application/pdf", "application/pdf"]
    assert set(ids) == {1, 4}


def test_search_applies_date_filter_end_to_end():
    mimes, ids = _search({"date_from": iso_ms(2024, 6, 1)})
    # Only candidates >= 2024-06-01 survive.
    assert 1 not in ids
    assert set(ids) == {2, 3, 4}


def test_search_no_filter_returns_all_mimes():
    mimes, ids = _search(None)
    assert set(ids) == {1, 2, 3, 4}
