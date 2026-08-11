from __future__ import annotations

from typing import Any, Dict, List, Optional

from .query import normalize_query
from .result import SearchResult
from src.artifacts.store import ArtifactStore


class SearchEngine:
    """
    Search engine that queries artifact FTS5 and optionally enriches
    each hit with associated episode and memory context.

    episode_store and memory_store are optional injected dependencies.
    When omitted, results are returned with empty episodes/memories/evidence.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        episode_store: Optional[Any] = None,
        memory_store: Optional[Any] = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.episode_store = episode_store
        self.memory_store = memory_store

    def search(self, query: str, limit: int = 20) -> List[SearchResult]:
        normalized = normalize_query(query)
        if not normalized:
            return []

        raw_hits: List[Dict[str, Any]] = self.artifact_store.search_artifacts(normalized, limit)

        results: List[SearchResult] = []
        for hit in raw_hits:
            artifact_id = int(hit["id"])
            episodes = self._get_episode_summaries(artifact_id)
            memories = self._get_memory_summaries(episodes)
            evidence = self._build_evidence(episodes, memories)

            results.append(
                SearchResult(
                    id=artifact_id,
                    file_name=str(hit.get("file_name") or ""),
                    path=str(hit.get("path") or ""),
                    mime_type=str(hit.get("mime_type") or ""),
                    size=int(hit.get("size") or 0),
                    modified_at=str(hit.get("modified_at") or ""),
                    rank=float(hit.get("rank") or 0.0),
                    snippet=str(hit.get("snippet") or ""),
                    episodes=episodes,
                    memories=memories,
                    evidence=evidence,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_episode_summaries(self, artifact_id: int) -> List[Dict[str, Any]]:
        if self.episode_store is None:
            return []
        try:
            episodes = self.episode_store.get_episodes_for_artifact(artifact_id)
        except Exception:
            return []
        # Sort by start_ts descending, cap at 3 most recent
        sorted_episodes = sorted(episodes, key=lambda e: e.start_ts, reverse=True)[:3]
        return [
            {
                "id": ep.id,
                "start_ts": ep.start_ts,
                "end_ts": ep.end_ts,
                "title": ep.title,
                "grouping_confidence": ep.grouping_confidence,
                "event_count": len(ep.event_ids),
            }
            for ep in sorted_episodes
        ]

    def _get_memory_summaries(
        self, episode_summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if self.memory_store is None:
            return []
        memories: List[Dict[str, Any]] = []
        seen_memory_ids: set[str] = set()
        for ep_summary in episode_summaries:
            episode_id = ep_summary["id"]
            try:
                episode_memories = self.memory_store.get_memories_for_episode(episode_id)
            except Exception:
                continue
            for mem in episode_memories:
                if mem.id not in seen_memory_ids:
                    seen_memory_ids.add(mem.id)
                    memories.append(
                        {
                            "id": mem.id,
                            "title": mem.title,
                            "topics": list(mem.topics),
                            "confidence": mem.confidence,
                            "start_ts": mem.start_ts,
                            "end_ts": mem.end_ts,
                        }
                    )
        return memories

    def _build_evidence(
        self,
        episode_summaries: List[Dict[str, Any]],
        memory_summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for ep in episode_summaries:
            evidence.append(
                {
                    "type": "episode",
                    "id": ep["id"],
                    "title": ep["title"],
                    "start_ts": ep["start_ts"],
                    "end_ts": ep["end_ts"],
                }
            )
        for mem in memory_summaries:
            evidence.append(
                {
                    "type": "memory",
                    "id": mem["id"],
                    "title": mem["title"],
                    "topics": mem["topics"],
                }
            )
        return evidence
