from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from .query import normalize_query
from .query_parser import ParsedQuery, QueryParser
from .ranking import rank_candidates, parse_iso_utc
from .result import SearchResult
from .temporal import TemporalParser, TimeRange
from src.artifacts.store import ArtifactStore


class SearchEngine:
    """
    Unified Contextual & Semantic Search Engine (Slice 7).

    Combines:
      - Deterministic QueryParser & TemporalParser
      - Candidate retrieval (FTS5 lexical, local ONNX vector semantic, temporal episode window, activity path)
      - Episode & Memory store enrichment
      - Hybrid explainable ranking with human-readable why explanations
      - Full backward-compatibility with SearchResult dict-style access
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        episode_store: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
        min_semantic_similarity: float = 0.35,
    ) -> None:
        self.artifact_store = artifact_store
        self.episode_store = episode_store
        self.memory_store = memory_store
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.min_semantic_similarity = min_semantic_similarity
        self.query_parser = QueryParser()
        self.temporal_parser = TemporalParser()

    def search(
        self,
        query: str,
        limit: int = 20,
        reference_date: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        parsed = self.query_parser.parse(query)
        time_range: Optional[TimeRange] = None
        if parsed.has_temporal:
            time_range = self.temporal_parser.parse(parsed.time_expression, reference_date=reference_date)

        candidates = self._retrieve_candidates(parsed, time_range, limit)
        candidates = self._apply_structured_filters(candidates, filters)
        if not candidates:
            return []

        # Enrich candidates with episode & memory context
        for candidate in candidates:
            art_id = int(candidate["id"])
            episodes = self._get_episode_summaries(art_id)
            memories = self._get_memory_summaries(episodes)
            evidence = self._build_evidence(episodes, memories)
            candidate["episodes"] = episodes
            candidate["memories"] = memories
            candidate["evidence"] = evidence

        # Rank candidates
        ranked = rank_candidates(candidates, parsed, time_range)

        results: List[SearchResult] = []
        for cand, score, why_str in ranked[:limit]:
            if score <= 0.0:
                continue
            results.append(
                SearchResult(
                    id=int(cand["id"]),
                    file_name=str(cand.get("file_name") or ""),
                    path=str(cand.get("path") or ""),
                    mime_type=str(cand.get("mime_type") or ""),
                    size=int(cand.get("size") or 0),
                    modified_at=str(cand.get("modified_at") or ""),
                    rank=float(cand.get("rank") or 0.0),
                    snippet=str(cand.get("snippet") or ""),
                    episodes=cand.get("episodes") or [],
                    memories=cand.get("memories") or [],
                    evidence=cand.get("evidence") or [],
                    result_type="artifact",
                    score=float(score),
                    why=why_str,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Candidate Retrieval Strategy (Lexical + Semantic + Temporal)
    # ------------------------------------------------------------------

    def _retrieve_candidates(
        self,
        parsed: ParsedQuery,
        time_range: Optional[TimeRange],
        limit: int,
    ) -> List[Dict[str, Any]]:
        mime_filter: Optional[str] = None
        if parsed.file_type == "image":
            mime_filter = "image"
        elif parsed.file_type in ("pdf", "docx", "txt", "md"):
            mime_filter = parsed.file_type

        candidates_map: Dict[int, Dict[str, Any]] = {}

        # Path 1: Activity Intent (episode time-window retrieval)
        if parsed.intent == "activity" and time_range and time_range.resolved and self.episode_store:
            episodes = self.episode_store.get_episodes_in_time_range(time_range.start_ts, time_range.end_ts)
            for ep in episodes:
                for art_id in ep.artifact_ids:
                    if art_id not in candidates_map:
                        row = self.artifact_store.get_artifact(art_id)
                        if row and not row["missing"]:
                            cand = dict(row)
                            cand["rank"] = 0.0
                            cand["snippet"] = ""
                            cand["fts_matched"] = False
                            candidates_map[art_id] = cand

        # Path 2: FTS Lexical Search
        fts_term_source = parsed.filename_hint or " ".join(parsed.terms)
        if fts_term_source:
            sanitized_query = re.sub(r"[^\w\s]", " ", fts_term_source).strip()
            if sanitized_query:
                fts_results = self.artifact_store.search_artifacts(sanitized_query, limit=limit, mime_type_filter=mime_filter)
                for cand in fts_results:
                    art_id = int(cand["id"])
                    cand["fts_matched"] = True
                    if art_id not in candidates_map:
                        candidates_map[art_id] = cand
                    else:
                        candidates_map[art_id]["rank"] = cand.get("rank", 0.0)
                        candidates_map[art_id]["snippet"] = cand.get("snippet", "")
                        candidates_map[art_id]["fts_matched"] = True

        # Path 3: Local Semantic Search (if VectorStore & EmbeddingEngine present)
        if (
            self.vector_store is not None
            and self.embedding_engine is not None
            and getattr(self.embedding_engine, "dimension", 0) > 0
            and parsed.original
        ):
            try:
                query_vec = self.embedding_engine.embed_text(parsed.original)
                if query_vec:
                    chunk_matches = self.vector_store.search_semantic_chunks(
                        query_embedding=query_vec,
                        model_id=self.embedding_engine.model_id,
                        dimension=self.embedding_engine.dimension,
                        top_k=limit,
                        min_similarity=self.min_semantic_similarity,
                    )
                    for m in chunk_matches:
                        art_id = m.artifact_id
                        if art_id not in candidates_map:
                            row = self.artifact_store.get_artifact(art_id)
                            if row and not row["missing"]:
                                cand = dict(row)
                                cand["rank"] = 0.0
                                cand["snippet"] = m.text
                                cand["fts_matched"] = False
                                cand["semantic_score"] = m.similarity
                                candidates_map[art_id] = cand
                        else:
                            prev_sim = float(candidates_map[art_id].get("semantic_score") or 0.0)
                            candidates_map[art_id]["semantic_score"] = max(prev_sim, m.similarity)
                            if not candidates_map[art_id].get("snippet"):
                                candidates_map[art_id]["snippet"] = m.text
            except Exception as sem_exc:
                # Graceful failure isolation
                pass

        # Path 4: Temporal Window artifacts fallback
        if parsed.has_temporal and time_range and time_range.resolved and self.episode_store and len(candidates_map) < limit:
            episodes = self.episode_store.get_episodes_in_time_range(time_range.start_ts, time_range.end_ts)
            for ep in episodes:
                for art_id in ep.artifact_ids:
                    if art_id not in candidates_map:
                        row = self.artifact_store.get_artifact(art_id)
                        if row and not row["missing"]:
                            cand = dict(row)
                            cand["rank"] = 0.0
                            cand["snippet"] = ""
                            cand["fts_matched"] = False
                            candidates_map[art_id] = cand

        return list(candidates_map.values())

    # ------------------------------------------------------------------
    # Structured API filters (C7)
    # ------------------------------------------------------------------
    # Applied uniformly AFTER candidate retrieval and BEFORE enrichment /
    # ranking. This preserves the existing hybrid ranking: filters only
    # include/exclude candidates; they never alter scores. The input list
    # is never mutated; a new list is returned.

    @staticmethod
    def _apply_structured_filters(
        candidates: List[Dict[str, Any]],
        filters: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not filters:
            return candidates

        mime_types = filters.get("mime_types")
        date_from = SearchEngine._coerce_epoch_ms(filters.get("date_from"))
        date_to = SearchEngine._coerce_epoch_ms(filters.get("date_to"))

        # No supported, usable keys -> behave exactly as before.
        if not mime_types and date_from is None and date_to is None:
            return candidates
        if mime_types is not None and not isinstance(mime_types, list):
            mime_types = None

        filtered: List[Dict[str, Any]] = []
        for cand in candidates:
            if not SearchEngine._candidate_matches_mime(cand, mime_types):
                continue
            if not SearchEngine._candidate_matches_date(cand, date_from, date_to):
                continue
            filtered.append(cand)
        return filtered

    @staticmethod
    def _coerce_epoch_ms(value: Any) -> Optional[int]:
        """Accept int epoch-ms; reject bool (an int subclass) and non-ints."""
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        return None

    @staticmethod
    def _candidate_matches_mime(
        cand: Dict[str, Any], mime_types: Optional[List[str]]
    ) -> bool:
        if not mime_types:
            return True
        cand_mime = str(cand.get("mime_type") or "").lower()
        for entry in mime_types:
            f = str(entry or "").lower().strip()
            if not f:
                continue
            if cand_mime == f:
                return True
            # Major-type matching: "image" matches "image/png".
            if cand_mime.startswith(f + "/"):
                return True
        return False

    @staticmethod
    def _candidate_matches_date(
        cand: Dict[str, Any],
        date_from: Optional[int],
        date_to: Optional[int],
    ) -> bool:
        if date_from is None and date_to is None:
            return True
        ts = SearchEngine._modified_at_to_epoch_ms(cand.get("modified_at"))
        # Malformed/missing timestamps fail safely when a bound is active.
        if ts is None:
            return False
        if date_from is not None and ts < date_from:
            return False
        if date_to is not None and ts > date_to:
            return False
        return True

    @staticmethod
    def _modified_at_to_epoch_ms(value: Any) -> Optional[int]:
        if value is None:
            return None
        dt = parse_iso_utc(str(value))
        if dt is None:
            return None
        return int(dt.timestamp() * 1000)

    # ------------------------------------------------------------------
    # Context Enrichment Helpers (Preserved)
    # ------------------------------------------------------------------

    def _get_episode_summaries(self, artifact_id: int) -> List[Dict[str, Any]]:
        if self.episode_store is None:
            return []
        try:
            episodes = self.episode_store.get_episodes_for_artifact(artifact_id)
        except Exception:
            return []
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
