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


# ---------------------------------------------------------------------------
# Evidence confidence classification (deterministic, LLM-free).
#
# Terminology follows docs/SEARCH.md ("confidence score with classification:
# FACT (direct evidence), INFERENCE (derived via heuristics/models), GUESS
# (low confidence)") and docs/ARCHITECTURE_REVIEW.md (evidence provenance is
# FACT = direct observable, INFERENCE = derived via rules/models, GUESS =
# low-confidence hypothesis; keep provenance distinct from retrieval score).
# ---------------------------------------------------------------------------
CONFIDENCE_FACT = "FACT"
CONFIDENCE_INFERENCE = "INFERENCE"
CONFIDENCE_GUESS = "GUESS"


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
        event_store: Optional[Any] = None,
        vector_store: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
        min_semantic_similarity: float = 0.35,
    ) -> None:
        self.artifact_store = artifact_store
        self.episode_store = episode_store
        self.memory_store = memory_store
        self.event_store = event_store
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
            # Evidence construction must never turn a search into a 500: any
            # failure degrades to empty evidence for this candidate.
            try:
                evidence = self._build_evidence(art_id, candidate, episodes, memories)
            except Exception:
                evidence = []
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

    # ------------------------------------------------------------------
    # Evidence construction (C8-2): real, classified source evidence
    # ------------------------------------------------------------------
    # Builds evidence from the runtime Events -> Episodes -> Memories data
    # that the artifact participates in. Evidence is EMPTY when the artifact
    # has no runtime-derived source data (events/episodes/memories), which
    # preserves the prior "no context -> empty evidence" contract.
    #
    # Each evidence item carries:
    #   type            - what the evidence points to: "event" | "episode" | "memory"
    #   id              - the source id (event/episode/memory id)
    #   confidence_type - "FACT" | "INFERENCE" | "GUESS" (provenance class)
    #   confidence      - factual confidence of this evidence item (0..1)
    #   source          - provenance label (event source name or store name)
    #   source_kind     - "filesystem" | "simulated" | "derived"
    #   title           - human-readable label
    #   snippet         - source text when available, else ""
    #   path            - source path / equivalent source info when available
    #   timestamp       - source timestamp when available
    #   artifact_id     - the document this evidence is attached to
    #   (plus type-specific fields: event_type, event_count, topics, ...)
    #
    # The matched artifact itself is the direct FACT record and is surfaced
    # via the result's own fields (id/path/file_name) and via the event
    # evidence that references it (artifact_id + path + snippet).

    def _build_evidence(
        self,
        artifact_id: int,
        candidate: Dict[str, Any],
        episode_summaries: List[Dict[str, Any]],
        memory_summaries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []

        # 1) Direct event evidence (FACT). Events are immutable observations.
        events = self._fetch_events_for_artifact(artifact_id)
        for ev in sorted(events, key=lambda e: e.timestamp):
            evidence.append(self._event_evidence(ev, artifact_id, candidate))

        # 2) Episode evidence: INFERENCE when underpinning events resolve,
        #    otherwise GUESS (provenance insufficient - never fabricated).
        episodes = self._fetch_full_episodes_for_artifact(artifact_id)
        for ep in episodes:
            evidence.append(self._episode_evidence(ep, artifact_id, candidate))

        # 3) Memory evidence: derived from episodes; same classification rule.
        seen_memory_ids: set[str] = set()
        for ep in episodes:
            for mem in self._fetch_memories_for_episode(ep.id):
                if mem.id in seen_memory_ids:
                    continue
                seen_memory_ids.add(mem.id)
                evidence.append(self._memory_evidence(mem, artifact_id, candidate))

        return evidence

    @staticmethod
    def classify_confidence(*, direct: bool, has_direct_signals: bool) -> str:
        """Deterministic, LLM-free confidence classification.

        Rules (per docs/SEARCH.md and docs/ARCHITECTURE_REVIEW.md):
          - FACT: the evidence is a direct observable record (an Event, or the
            artifact's own stored record). No inference is involved.
          - INFERENCE: the evidence is derived from multiple direct signals
            (an Episode or Memory assembled from underlying Events) AND those
            underpinning signals were confirmed via the event store.
          - GUESS: the evidence is derived but the underpinning direct signals
            could not be confirmed (event store absent or events missing). We
            do NOT fabricate certainty, so the weakest supported class is used.

        This is intentionally pure and side-effect free so classification is
        reproducible and testable.
        """
        if direct:
            return CONFIDENCE_FACT
        return CONFIDENCE_INFERENCE if has_direct_signals else CONFIDENCE_GUESS

    def _fetch_events_for_artifact(self, artifact_id: int) -> List[Any]:
        if self.event_store is None:
            return []
        try:
            return list(self.event_store.get_events_for_artifact(artifact_id))
        except Exception:
            return []

    def _fetch_full_episodes_for_artifact(self, artifact_id: int) -> List[Any]:
        if self.episode_store is None:
            return []
        try:
            all_episodes = self.episode_store.get_episodes_for_artifact(artifact_id)
            return sorted(all_episodes, key=lambda e: e.start_ts, reverse=True)[:3]
        except Exception:
            return []

    def _fetch_memories_for_episode(self, episode_id: str) -> List[Any]:
        if self.memory_store is None:
            return []
        try:
            return list(self.memory_store.get_memories_for_episode(episode_id))
        except Exception:
            return []

    def _episode_has_direct_signals(self, episode: Any) -> bool:
        if self.event_store is None:
            return False
        for eid in getattr(episode, "event_ids", []) or []:
            try:
                if self.event_store.get_event(eid) is not None:
                    return True
            except Exception:
                continue
        return False

    def _memory_has_direct_signals(self, memory: Any) -> bool:
        if self.event_store is None:
            return False
        for eid in getattr(memory, "event_ids", []) or []:
            try:
                if self.event_store.get_event(eid) is not None:
                    return True
            except Exception:
                continue
        return False

    def _event_evidence(
        self, ev: Any, artifact_id: int, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = getattr(ev, "payload", None) or {}
        path = payload.get("path") or candidate.get("path") or ""
        # Events carry no body text; the artifact's extracted snippet is the
        # available source text (populated by FTS/semantic match). Empty when
        # no source text exists (e.g. images).
        snippet = candidate.get("snippet") or ""
        return {
            "type": "event",
            "id": ev.id,
            "confidence_type": self.classify_confidence(direct=True, has_direct_signals=False),
            "confidence": float(getattr(ev, "event_confidence", 1.0)),
            "source": getattr(ev, "source", ""),
            "source_kind": getattr(ev, "source_kind", ""),
            "event_type": getattr(ev, "type", ""),
            "title": f"{getattr(ev, 'type', 'event')} ({getattr(ev, 'source_kind', '')})",
            "snippet": snippet,
            "path": path,
            "timestamp": getattr(ev, "timestamp", ""),
            "artifact_id": artifact_id,
        }

    def _episode_evidence(
        self, ep: Any, artifact_id: int, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        has_signals = self._episode_has_direct_signals(ep)
        return {
            "type": "episode",
            "id": ep.id,
            "confidence_type": self.classify_confidence(
                direct=False, has_direct_signals=has_signals
            ),
            "confidence": float(getattr(ep, "grouping_confidence", 0.0)),
            "source": "episode",
            "source_kind": "derived",
            "title": getattr(ep, "title", ""),
            "snippet": "",
            "path": candidate.get("path") or "",
            "timestamp": getattr(ep, "start_ts", ""),
            "start_ts": getattr(ep, "start_ts", ""),
            "end_ts": getattr(ep, "end_ts", ""),
            "artifact_id": artifact_id,
            "event_count": len(getattr(ep, "event_ids", []) or []),
            "grouping_confidence": float(getattr(ep, "grouping_confidence", 0.0)),
        }

    def _memory_evidence(
        self, mem: Any, artifact_id: int, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        has_signals = self._memory_has_direct_signals(mem)
        return {
            "type": "memory",
            "id": mem.id,
            "confidence_type": self.classify_confidence(
                direct=False, has_direct_signals=has_signals
            ),
            "confidence": float(getattr(mem, "confidence", 0.0)),
            "source": "memory",
            "source_kind": "derived",
            "title": getattr(mem, "title", ""),
            # The deterministic template summary is the available source text.
            "snippet": getattr(mem, "title", ""),
            "path": candidate.get("path") or "",
            "timestamp": getattr(mem, "start_ts", ""),
            "artifact_id": artifact_id,
            "topics": list(getattr(mem, "topics", []) or []),
        }
