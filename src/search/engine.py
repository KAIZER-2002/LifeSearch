from typing import Dict, List

from .query import normalize_query
from src.artifacts.store import ArtifactStore


class SearchEngine:
    def __init__(self, artifact_store: ArtifactStore):
        self.artifact_store = artifact_store

    def search(self, query: str, limit: int = 20) -> List[Dict[str, str]]:
        normalized = normalize_query(query)
        if not normalized:
            return []
        return self.artifact_store.search_artifacts(normalized, limit)
