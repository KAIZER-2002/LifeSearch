import re
from typing import Any, Dict, Iterable, List, Tuple


_DEFAULT_DOCUMENTS = [
    {
        "id": "doc-uuid-1",
        "type": "artifact",
        "file_name": "qdrant_notes.pdf",
        "path": "C:/Users/Test/Downloads/qdrant_notes.pdf",
        "mime_type": "application/pdf",
        "snippet": "Qdrant is a vector database optimized for semantic search.",
        "text": "Qdrant is a vector database optimized for semantic search. The Qdrant PDF includes instructions for building a retrieval system.",
        "last_opened_ms": 1715635200000,
        "thumbnail_path": "C:/Users/Test/.lifesearch/cache/thumbs/doc-uuid-1.png",
        "evidence": [
            {"type": "event", "id": "evt-qdrant-download", "source": "browser-chrome"},
            {"type": "artifact", "id": "doc-uuid-1"},
        ],
    },
    {
        "id": "art-screenshot-mongo",
        "type": "artifact",
        "file_name": "mongo_error.png",
        "path": "C:/Users/Test/Screenshots/mongo_error.png",
        "mime_type": "image/png",
        "snippet": "Screenshot of a MongoDB error dialog.",
        "text": "MongoDB error during replication. The red error window shows server 500 and connection failure.",
        "last_opened_ms": 1715638800000,
        "thumbnail_path": "C:/Users/Test/.lifesearch/cache/thumbs/art-screenshot-mongo.png",
        "evidence": [
            {"type": "event", "id": "evt-screenshot-mongo", "source": "screenshot-watcher"},
            {"type": "artifact", "id": "art-screenshot-mongo"},
        ],
    },
    {
        "id": "art-retrieval-py",
        "type": "artifact",
        "file_name": "retrieval.py",
        "path": "C:/Users/Test/Projects/retrieval.py",
        "mime_type": "text/x-python",
        "snippet": "Python code for the retrieval pipeline and query orchestration.",
        "text": "This retrieval.py file contains the core search pipeline and Qdrant integration examples.",
        "last_opened_ms": 1715642400000,
        "thumbnail_path": "C:/Users/Test/.lifesearch/cache/thumbs/art-retrieval-py.png",
        "evidence": [
            {"type": "event", "id": "evt-edit-retrieval", "source": "vscode"},
            {"type": "artifact", "id": "art-retrieval-py"},
        ],
    },
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).lower()


def _tokenize(text: str) -> List[str]:
    return [token for token in re.split(r"\W+", text.lower()) if token]


def _score_document(query_tokens: Iterable[str], document: Dict[str, Any]) -> float:
    text = _normalize_text(document.get("text", ""))
    name = _normalize_text(document.get("file_name", ""))
    path = _normalize_text(document.get("path", ""))

    score = 0.0
    for token in query_tokens:
        if token in text:
            score += 2.0
        if token in name:
            score += 1.5
        if token in path:
            score += 1.0
    if not score:
        return 0.0
    return score / (1.0 + len(query_tokens) * 0.5)


def _document_highlights(query_tokens: Iterable[str], document: Dict[str, Any]) -> List[str]:
    text = _normalize_text(document.get("text", ""))
    highlights = []
    for token in query_tokens:
        if token in text and token not in highlights:
            highlights.append(token)
    return highlights


def search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: List[Tuple[Dict[str, Any], float]] = []
    for document in _DEFAULT_DOCUMENTS:
        score = _score_document(query_tokens, document)
        if score > 0:
            scored.append((document, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    results: List[Dict[str, Any]] = []
    for document, score in scored[:k]:
        result = {
            "type": document["type"],
            "id": document["id"],
            "file_name": document["file_name"],
            "path": document["path"],
            "mime_type": document["mime_type"],
            "snippet": document["snippet"],
            "highlights": _document_highlights(query_tokens, document),
            "score": round(score, 3),
            "why": _build_why_text(query_tokens, document),
            "last_opened_ms": document.get("last_opened_ms"),
            "thumbnail_path": document.get("thumbnail_path"),
            "evidence": document.get("evidence", []),
        }
        results.append(result)
    return results


def _build_why_text(query_tokens: Iterable[str], document: Dict[str, Any]) -> str:
    matches = [token for token in query_tokens if token in _normalize_text(document.get("text", ""))]
    if matches:
        return f"Matched {', '.join(matches)} in extracted content and filename."
    if any(token in _normalize_text(document.get("file_name", "")) for token in query_tokens):
        return "Matched the filename for the requested artifact."
    return "Matched the request using lexical search."
