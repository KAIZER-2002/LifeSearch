"""
Mapping layer: SearchResult -> API ResultCard

Pure transformation functions with no side effects.
HTTP concerns (status codes, headers, serialization) stay out.
"""

from typing import Any, Dict, List, Optional

from src.search.result import SearchResult


def map_search_result(result: SearchResult) -> Dict[str, Any]:
    """Map a single SearchResult to the API ResultCard shape."""
    # Build highlights from snippet (simple approach - split on common delimiters)
    highlights = _extract_highlights(result.snippet) if result.snippet else []

    return {
        "document_id": str(result.id),
        "file_name": result.file_name,
        "path": result.path,
        "mime_type": result.mime_type,
        "snippet": result.snippet,
        "highlights": highlights,
        "score": result.score,
        "why": result.why,
        # Extensions: expose episodes, memories, evidence when available
        "episodes": result.episodes,
        "memories": result.memories,
        "evidence": result.evidence,
    }


def map_search_results(results: List[SearchResult]) -> List[Dict[str, Any]]:
    """Map a list of SearchResults to API ResultCards."""
    return [map_search_result(r) for r in results]


def _extract_highlights(snippet: str) -> List[str]:
    """Extract highlight terms from snippet (simple heuristic)."""
    # Simple extraction: words that appear emphasized or are key terms
    # For now, return empty list - highlights can be enhanced later
    return []


def build_search_response(
    results: List[SearchResult],
    took_ms: int,
    query_embedding_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the full SearchResponse object."""
    return {
        "took_ms": took_ms,
        "query_embedding_id": query_embedding_id,
        "results": map_search_results(results),
    }


def build_status_response(
    indexed_documents: int,
    indexed_chunks: int,
    model_available: bool,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the StatusResponse object."""
    return {
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "model_available": model_available,
        "model_id": model_id,
    }


def build_error_response(code: int, message: str, detail: Optional[str] = None) -> Dict[str, Any]:
    """Build a standardized error response."""
    error = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if detail:
        error["error"]["detail"] = detail
    return error