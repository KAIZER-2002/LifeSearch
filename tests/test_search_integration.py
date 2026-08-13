import re
import pytest

"""
Integration tests for search over events/episodes/artifacts.
These skeleton tests assume a search API with a function like:
  search(query, k=10) -> list of result dicts
Each result dict should include: type (artifact|episode|event), id, score, evidence (list)
"""

try:
    from src.search.engine import search
except Exception:
    search = None

try:
    from tests.recall_cases import cases as recall_cases
except Exception:
    recall_cases = None


@pytest.mark.skipif(search is None or recall_cases is None, reason="Search engine or recall cases not available")
def test_recall_cases_top3():
    """Run curated recall cases and ensure expected results appear in top-3 for core cases."""
    priority_ids = ["rc-01", "rc-02", "rc-05", "rc-13"]

    def normalize_result(result: dict) -> str:
        return (result.get("id", "") + result.get("file_name", "") + result.get("path", "")).lower()

    def is_match(result: dict, expected: str) -> bool:
        normalized_expected = expected.replace("\\.", ".")
        pattern = re.escape(normalized_expected)
        pattern = (
            pattern
            .replace(r"\*", ".*")
            .replace(r"\|", "|")
            .replace(r"\(", "(")
            .replace(r"\)", ")")
        )
        return re.search(pattern, normalize_result(result), flags=re.IGNORECASE) is not None

    failures = []
    for case in recall_cases:
        if case.get("id") not in priority_ids:
            continue
        query = case["query"]
        expected = case["expected_match"]

        results = search(query, k=3)
        matched = False
        for r in results:
            if r.get("type") == "artifact" and is_match(r, expected):
                matched = True
                break
        if not matched:
            failures.append((case.get("id"), query, results))

    assert not failures, f"Some priority recall cases failed: {failures}"


@pytest.mark.skipif(search is None, reason="Search engine not implemented")
def test_evidence_trace_in_results():
    """Ensure search results include evidence traces linking to events/artifacts."""
    res = search("Find the Qdrant PDF I downloaded around May.", k=1)
    assert res, "Search should return at least one result"
    top = res[0]
    assert "evidence" in top and isinstance(top["evidence"], list)
    # Evidence items should reference event ids or artifact ids
    assert any(isinstance(ev, dict) and ev.get("type") in ("event", "artifact") for ev in top["evidence"]) 
