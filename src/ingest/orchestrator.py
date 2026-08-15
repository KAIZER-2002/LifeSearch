"""
Runtime ingestion orchestration (C8-1).

Purpose
-------
Make the existing, already-tested pipeline run automatically after
artifact indexing, for BOTH the CLI (`lifesearch index`) and the HTTP
(`POST /index`) paths, via a single reusable service:

    ArtifactScanner.index_folder
        -> FilesystemEventSource.generate_events
        -> EventStore.append_event
        -> EpisodeEngine.detect_and_persist
        -> MemoryBuilder.build (+ MemoryStore.save_memories)

Design constraints
------------------
- No component internals are modified; only their public APIs are used.
- Each call opens its OWN EventStore / EpisodeStore / MemoryStore
  connections (created and used on whichever thread calls this, so it
  is safe from the CLI main thread and from the server's indexing
  worker thread). The data still lands in the same on-disk SQLite file
  that SearchEngine reads, so search enrichment sees it.
- Idempotency is delegated to the existing components:
    * FilesystemEventSource only emits FILE_CREATED / FILE_DELETED when
      the passed EventStore has no prior matching event, so re-indexing
      an unchanged folder does not duplicate events.
    * Episode / Memory stores use INSERT OR REPLACE keyed by stable
      ids, so re-runs replace rather than duplicate.
- The existing index_folder behavior (progress callback, incremental
  indexing, changed/missing handling) is preserved unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.engine import EpisodeEngine
from src.episodes.store import EpisodeStore
from src.events.source import FilesystemEventSource
from src.events.store import EventStore
from src.memories.builder import MemoryBuilder
from src.memories.store import MemoryStore

logger = logging.getLogger(__name__)


def run_ingest_folder(
    scanner: ArtifactScanner,
    folder: str,
    db_path: str,
    reindex: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, int]:
    """Run the full ingest for a SINGLE folder and return cumulative counts.

    Order:
        1. ArtifactScanner.index_folder (unchanged existing behavior)
        2. FilesystemEventSource.generate_events -> EventStore.append_event
        3. EpisodeEngine.detect_and_persist (EventStore -> EpisodeStore)
        4. MemoryBuilder.build per episode -> MemoryStore.save_memories

    The event/episode/memory stores are opened and closed within this
    call so the function is safe to invoke from any thread.
    """
    # Open our own connections from db_path (self-contained, thread-safe,
    # and independent of scanner internals / injected test fakes).
    artifact_store = ArtifactStore(db_path)
    event_store = EventStore(db_path)
    episode_store = EpisodeStore(db_path)
    memory_store = MemoryStore(db_path)
    episode_engine = EpisodeEngine()
    memory_builder = MemoryBuilder(event_provider=event_store)

    totals: Dict[str, int] = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "events": 0,
        "episodes": 0,
        "memories": 0,
    }

    try:
        # 1. Artifact indexing (existing behavior, including progress reporting).
        result = scanner.index_folder(
            folder,
            reindex_on_model_change=reindex,
            force_reindex=reindex,
            progress_callback=progress_callback,
        )
        totals["processed"] = int(result.get("processed", 0) or 0)
        totals["skipped"] = int(result.get("skipped", 0) or 0)
        totals["errors"] = int(result.get("errors", 0) or 0)

        # 2. Generate and persist filesystem events (idempotent via EventStore).
        events = FilesystemEventSource(artifact_store, event_store).generate_events(folder)
        for ev in events:
            try:
                event_store.append_event(ev)
                totals["events"] += 1
            except ValueError:
                # Exact-id collision guard (astronomically unlikely with uuid4
                # ids); safe to skip without affecting idempotency.
                pass

        # 3. Detect and persist episodes (INSERT OR REPLACE by stable id).
        episodes = episode_engine.detect_and_persist(event_store, episode_store)
        totals["episodes"] = len(episodes)

        # 4. Build and persist memories for each episode.
        memories = []
        for ep in episodes:
            mem = memory_builder.build(ep)
            if mem is not None:
                memories.append(mem)
        if memories:
            memory_store.save_memories(memories)
        totals["memories"] = len(memories)
    finally:
        event_store.close()
        episode_store.close()
        memory_store.close()

    return totals


def run_ingest(
    scanner: ArtifactScanner,
    folders: List[str],
    db_path: str,
    reindex: bool = False,
    progress_callback: Optional[Any] = None,
) -> Dict[str, int]:
    """Reusable orchestration service over one or more folders.

    Both the CLI and the HTTP server call this (or its single-folder
    variant) so the Events -> Episodes -> Memories logic lives in
    exactly one place.
    """
    totals: Dict[str, int] = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "events": 0,
        "episodes": 0,
        "memories": 0,
    }
    for folder in folders:
        sub = run_ingest_folder(
            scanner,
            folder,
            db_path,
            reindex=reindex,
            progress_callback=progress_callback,
        )
        for key in totals:
            totals[key] += sub.get(key, 0)
    return totals
