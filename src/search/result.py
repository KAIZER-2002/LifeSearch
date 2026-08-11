from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SearchResult:
    """
    A single result from SearchEngine.search().

    Attributes:
        id: Artifact row ID from the artifacts table.
        file_name: Filename of the artifact.
        path: Absolute path of the artifact.
        mime_type: MIME type of the artifact.
        size: File size in bytes.
        modified_at: ISO 8601 timestamp of last modification.
        rank: BM25 rank from FTS5 (lower/more negative = more relevant).
        snippet: Highlighted FTS5 snippet.
        episodes: Serialized summaries of associated episodes.
        memories: Serialized summaries of associated memories.
        evidence: Typed evidence dicts for downstream consumers.
            Each item has a "type" key: "episode" or "memory".
        result_type: Type of result item ("artifact" | "episode" | "memory"). Default "artifact".
        score: Composite hybrid ranking score in [0.0, 1.0].
        why: Human-readable explanation string for why this result matched.
    """

    id: int
    file_name: str
    path: str
    mime_type: str
    size: int
    modified_at: str
    rank: float
    snippet: str
    episodes: List[Dict[str, Any]] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    result_type: str = "artifact"
    score: float = 0.0
    why: str = ""

    # ------------------------------------------------------------------
    # Backward-compatible dict-style access
    # Allows result["file_name"] in addition to result.file_name so that
    # existing tests written against the old Dict interface
    # continue to work without modification.
    # ------------------------------------------------------------------

    def _as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "path": self.path,
            "mime_type": self.mime_type,
            "size": self.size,
            "modified_at": self.modified_at,
            "rank": self.rank,
            "snippet": self.snippet,
            "episodes": self.episodes,
            "memories": self.memories,
            "evidence": self.evidence,
            "result_type": self.result_type,
            "score": self.score,
            "why": self.why,
        }

    def __getitem__(self, key: str) -> Any:
        return self._as_dict()[key]

    def __contains__(self, key: object) -> bool:
        return key in self._as_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self._as_dict().get(key, default)
