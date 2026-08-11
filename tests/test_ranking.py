from src.search.query_parser import ParsedQuery
from src.search.ranking import (
    compute_filename_score,
    compute_temporal_score,
    normalize_fts_ranks,
    rank_candidates,
)
from src.search.temporal import TimeRange


def test_normalize_fts_ranks():
    # SQLite FTS5 rank: lower/more negative is better
    raw = [-10.5, -5.0, -1.0]
    norm = normalize_fts_ranks(raw)
    assert len(norm) == 3
    assert norm[0] == 1.0  # best rank gets 1.0
    assert norm[2] == 0.0  # worst rank gets 0.0


def test_compute_filename_score():
    assert compute_filename_score("qdrant_notes.pdf", ["qdrant"]) == 1.0
    assert compute_filename_score("retrieval.py", ["qdrant"]) == 0.0


def test_compute_temporal_score():
    tr = TimeRange(
        start_ts="2026-05-01T00:00:00+00:00",
        end_ts="2026-05-31T23:59:59+00:00",
        precision="month",
        resolved=True,
    )
    # Inside range
    assert compute_temporal_score("2026-05-14T12:00:00+00:00", tr) == 1.0
    # Near boundary (+/- 7 days)
    assert compute_temporal_score("2026-06-03T12:00:00+00:00", tr) == 0.5
    # Far out
    assert compute_temporal_score("2026-08-01T12:00:00+00:00", tr) == 0.0


def test_rank_candidates_generates_explainable_why_string():
    candidates = [
        {
            "id": 1,
            "file_name": "qdrant_notes.pdf",
            "modified_at": "2026-05-14T14:00:00+00:00",
            "rank": -10.0,
            "snippet": "qdrant vector database",
            "mime_type": "application/pdf",
            "fts_matched": True,
        }
    ]
    pq = ParsedQuery(original="qdrant pdf", terms=["qdrant"], file_type="pdf")
    tr = TimeRange(start_ts="2026-05-01T00:00:00+00:00", end_ts="2026-05-31T23:59:59+00:00", resolved=True, original_expression="in May")

    scored = rank_candidates(candidates, pq, tr)
    assert len(scored) == 1
    cand, score, why = scored[0]
    assert score > 0.5
    assert "Matched 'qdrant' in filename." in why
    assert "Modified within requested range" in why
