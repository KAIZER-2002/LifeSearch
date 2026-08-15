"""
C8-2 focused tests: real source evidence with deterministic
FACT / INFERENCE / GUESS confidence classification.

These tests verify the evidence enrichment added in C8-2:
  - evidence items carry real source identifiers (event/episode/memory ids,
    artifact_id, path, source, timestamp, snippet)
  - evidence classification (FACT / INFERENCE / GUESS) is deterministic and
    LLM-free
  - searches with no runtime data keep returning empty evidence (backward
    compatible)
  - evidence construction fails safely (never turns /search into a 500)
  - the REAL runtime path (temp files -> run_ingest -> events -> episodes
    -> memories -> SearchEngine.search) yields non-empty, correctly
    classified evidence.

The mandatory integration test (test_runtime_ingest_produces_real_evidence)
does NOT hand-wire the final evidence state: it runs the exact same
run_ingest orchestrator used by the CLI and HTTP server and then searches,
closing the false-confidence gap the C8 audit identified.
"""

import os
import tempfile

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.engine import EpisodeEngine
from src.episodes.store import EpisodeStore
from src.events.model import Event
from src.events.store import EventStore
from src.memories.builder import MemoryBuilder
from src.memories.store import MemoryStore
from src.search.engine import CONFIDENCE_FACT, CONFIDENCE_GUESS, CONFIDENCE_INFERENCE, SearchEngine
from src.vector.embeddings import NullEmbeddingEngine
from src.vector.store import VectorStore

from src.ingest.orchestrator import run_ingest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _make_event(event_id: str, artifact_id, path: str, timestamp: str) -> Event:
    return Event(
        id=event_id,
        type="FILE_OPENED",
        timestamp=timestamp,
        source="test-source",
        source_kind="simulated",
        artifact_id=artifact_id,
        payload={"path": path},
        event_confidence=0.9,
    )


def _index_single(tmp_path, name: str, content: str):
    """Index a single file, return (db_path, artifact_id, path)."""
    db_path = str(tmp_path / "ls.db")
    txt_path = str(tmp_path / name)
    _write_text(txt_path, content)
    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(str(tmp_path))
    row = artifact_store.get_artifact_by_path(txt_path)
    artifact_store.close()
    assert row is not None
    return db_path, int(row["id"]), txt_path


def _build_episode_memory(db_path, event_store, episode_store, memory_store):
    """Build one episode and one memory from whatever events exist (real
    components, no manual evidence fabrication)."""
    engine_ep = EpisodeEngine()
    episodes = engine_ep.detect_and_persist(
        event_store, episode_store,
        start_ts="2026-01-01T09:00:00+00:00",
        end_ts="2026-01-01T11:00:00+00:00",
    )
    builder = MemoryBuilder(event_provider=event_store)
    memories = [m for ep in episodes if (m := builder.build(ep)) is not None]
    memory_store.save_memories(
        memories,
        range_start_ts="2026-01-01T09:00:00+00:00",
        range_end_ts="2026-01-01T11:00:00+00:00",
    )
    return episodes, memories


# ---------------------------------------------------------------------------
# 1. Direct event evidence is classified FACT
# ---------------------------------------------------------------------------

def test_direct_event_evidence_is_fact(tmp_path):
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    ev = _make_event("e-fact", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)

    engine = SearchEngine(
        ArtifactStore(db_path),
        event_store=event_store,
    )
    results = engine.search("retrieval")
    event_store.close()

    assert results, "expected a result"
    evidence = results[0].evidence
    event_items = [e for e in evidence if e["type"] == "event"]
    assert event_items, "event evidence must be present"
    for item in event_items:
        assert item["confidence_type"] == CONFIDENCE_FACT
        assert item["confidence"] == 0.9


# ---------------------------------------------------------------------------
# 2. Episode-derived evidence is classified correctly (INFERENCE)
# ---------------------------------------------------------------------------

def test_episode_evidence_is_inference_when_events_resolve(tmp_path):
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    ev = _make_event("e-inf", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)
    episode_store = EpisodeStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, MemoryStore(db_path))

    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
        event_store=event_store,
    )
    results = engine.search("retrieval")
    event_store.close()
    episode_store.close()

    assert results
    ep_items = [e for e in results[0].evidence if e["type"] == "episode"]
    assert ep_items, "episode evidence must be present"
    for item in ep_items:
        assert item["confidence_type"] == CONFIDENCE_INFERENCE


# ---------------------------------------------------------------------------
# 3. Memory-derived evidence is classified correctly (INFERENCE)
# ---------------------------------------------------------------------------

def test_memory_evidence_is_inference_when_events_resolve(tmp_path):
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    ev = _make_event("e-mem", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)
    episode_store = EpisodeStore(db_path)
    memory_store = MemoryStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, memory_store)

    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
        memory_store=memory_store,
        event_store=event_store,
    )
    results = engine.search("retrieval")
    event_store.close()
    episode_store.close()
    memory_store.close()

    assert results
    mem_items = [e for e in results[0].evidence if e["type"] == "memory"]
    assert mem_items, "memory evidence must be present"
    for item in mem_items:
        assert item["confidence_type"] == CONFIDENCE_INFERENCE
        assert isinstance(item["topics"], list)


# ---------------------------------------------------------------------------
# 4. Evidence contains the expected source identifiers
# ---------------------------------------------------------------------------

def test_evidence_contains_source_identifiers(tmp_path):
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    ev = _make_event("e-id", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)
    episode_store = EpisodeStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, MemoryStore(db_path))

    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
        event_store=event_store,
    )
    results = engine.search("retrieval")
    event_store.close()
    episode_store.close()

    assert results
    top = results[0]
    ev_by_type = {e["type"]: e for e in top.evidence}

    assert "event" in ev_by_type
    e = ev_by_type["event"]
    assert e["id"] == "e-id"
    assert e["event_type"] == "FILE_OPENED"
    assert e["artifact_id"] == artifact_id
    assert e["path"] == txt_path
    assert e["source"] == "test-source"
    assert e["source_kind"] == "simulated"
    assert e["timestamp"] == "2026-01-01T10:00:00+00:00"

    assert "episode" in ev_by_type
    ep = ev_by_type["episode"]
    assert ep["id"]
    assert ep["artifact_id"] == artifact_id
    assert ep["event_count"] >= 1


# ---------------------------------------------------------------------------
# 5. Evidence snippets are populated when source text exists
# ---------------------------------------------------------------------------

def test_evidence_snippet_populated_when_source_text_exists(tmp_path):
    db_path, artifact_id, txt_path = _index_single(
        tmp_path, "snippet.txt", "zygomorphic retrieval pipeline example"
    )

    event_store = EventStore(db_path)
    ev = _make_event("e-snip", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)

    engine = SearchEngine(
        ArtifactStore(db_path),
        event_store=event_store,
    )
    results = engine.search("zygomorphic")
    event_store.close()

    assert results
    event_items = [e for e in results[0].evidence if e["type"] == "event"]
    assert event_items
    # The artifact's extracted text produced an FTS snippet for this match.
    assert event_items[0]["snippet"], "snippet must be populated when source text exists"
    assert "zygomorphic" in event_items[0]["snippet"]


# ---------------------------------------------------------------------------
# 6. Missing / invalid source data fails safely (no 500, safe evidence)
# ---------------------------------------------------------------------------

class _BrokenEventStore:
    """Mimics an event store that raises on every access."""

    def get_events_for_artifact(self, artifact_id):
        raise RuntimeError("injected event store failure")

    def get_event(self, event_id):
        raise RuntimeError("injected event store failure")


def test_missing_source_data_fails_safely(tmp_path):
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    ev = _make_event("e-broken", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)
    episode_store = EpisodeStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, MemoryStore(db_path))

    # A broken event store must not crash search; derived evidence degrades
    # to GUESS (provenance could not be confirmed) instead of raising.
    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
        event_store=_BrokenEventStore(),
    )
    # Must not raise.
    results = engine.search("retrieval")
    event_store.close()
    episode_store.close()

    assert results, "search must still return results"
    ep_items = [e for e in results[0].evidence if e["type"] == "episode"]
    assert ep_items
    assert ep_items[0]["confidence_type"] == CONFIDENCE_GUESS


# ---------------------------------------------------------------------------
# 7. Existing search results without evidence remain valid (empty evidence)
# ---------------------------------------------------------------------------

def test_existing_search_without_evidence_remains_valid(tmp_path):
    db_path, _, _ = _index_single(tmp_path, "orphan.txt", "orphan content unrelated")

    # No episode/memory/event stores -> no runtime data -> empty evidence.
    engine = SearchEngine(ArtifactStore(db_path))
    results = engine.search("orphan")
    assert results
    assert results[0].evidence == []


# ---------------------------------------------------------------------------
# 8. Multiple evidence items are deterministic (stable ordering/shape)
# ---------------------------------------------------------------------------

def test_multiple_evidence_items_deterministic(tmp_path):
    db_path, artifact_id, txt_path = _index_single(
        tmp_path, "multi.txt", "multi signal retrieval pipeline content"
    )

    event_store = EventStore(db_path)
    event_store.append_event(_make_event("e-a", artifact_id, txt_path, "2026-01-01T09:30:00+00:00"))
    event_store.append_event(_make_event("e-b", artifact_id, txt_path, "2026-01-01T10:00:00+00:00"))
    event_store.append_event(_make_event("e-c", artifact_id, txt_path, "2026-01-01T10:30:00+00:00"))
    episode_store = EpisodeStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, MemoryStore(db_path))

    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
        event_store=event_store,
    )
    run_a = engine.search("multi")
    run_b = engine.search("multi")
    event_store.close()
    episode_store.close()

    assert run_a and run_b
    # Determinism: identical evidence across repeated searches.
    assert [e.get("id") for e in run_a[0].evidence] == [e.get("id") for e in run_b[0].evidence]
    # Events are ordered by timestamp (deterministic ascending sort).
    event_ids = [e["id"] for e in run_a[0].evidence if e["type"] == "event"]
    assert event_ids == ["e-a", "e-b", "e-c"]


# ---------------------------------------------------------------------------
# 9. REAL runtime path: temp files -> run_ingest -> search -> real evidence
#    (no hand-manufactured final evidence state)
# ---------------------------------------------------------------------------

def _make_scanner(db_path: str) -> ArtifactScanner:
    artifact_store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    scanner = ArtifactScanner(
        artifact_store,
        Extractor(),
        vector_store=vector_store,
        embedding_engine=NullEmbeddingEngine(),
    )
    return scanner


def test_runtime_ingest_produces_real_evidence(tmp_path):
    db = str(tmp_path / "runtime.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    _write_text(
        folder / "qdrant_notes.txt",
        "qdrant vector database project notes for retrieval evidence",
    )

    scanner = _make_scanner(db)
    # The exact orchestrator the CLI and HTTP server use. Evidence state is
    # produced by the runtime, not by this test.
    totals = run_ingest(scanner, [str(folder)], db)
    assert totals["events"] >= 1
    assert totals["episodes"] >= 1
    assert totals["memories"] >= 1

    engine = SearchEngine(
        ArtifactStore(db),
        episode_store=EpisodeStore(db),
        memory_store=MemoryStore(db),
        event_store=EventStore(db),
        vector_store=VectorStore(db),
        embedding_engine=NullEmbeddingEngine(),
    )
    results = engine.search("qdrant")
    assert results, "expected search results after runtime ingest"
    top = results[0]
    assert top.evidence, "runtime ingest must yield non-empty real evidence"

    by_type = {}
    for item in top.evidence:
        by_type.setdefault(item["type"], []).append(item)

    # Event evidence is a direct observable -> FACT.
    assert "event" in by_type
    assert all(e["confidence_type"] == CONFIDENCE_FACT for e in by_type["event"])
    assert all(e["artifact_id"] == top.id for e in by_type["event"])

    # Episode evidence is derived from events that now resolve -> INFERENCE.
    assert "episode" in by_type
    assert all(e["confidence_type"] == CONFIDENCE_INFERENCE for e in by_type["episode"])

    # Memory evidence is derived -> INFERENCE.
    assert "memory" in by_type
    assert all(e["confidence_type"] == CONFIDENCE_INFERENCE for e in by_type["memory"])


# ---------------------------------------------------------------------------
# 10. Classification is deterministic and follows the documented terminology
# ---------------------------------------------------------------------------

def test_classification_is_deterministic():
    # direct observable -> FACT
    assert SearchEngine.classify_confidence(direct=True, has_direct_signals=False) == CONFIDENCE_FACT
    assert SearchEngine.classify_confidence(direct=True, has_direct_signals=True) == CONFIDENCE_FACT

    # derived + confirmed signals -> INFERENCE
    assert SearchEngine.classify_confidence(direct=False, has_direct_signals=True) == CONFIDENCE_INFERENCE

    # derived + unconfirmed signals -> GUESS (never fabricated certainty)
    assert SearchEngine.classify_confidence(direct=False, has_direct_signals=False) == CONFIDENCE_GUESS

    # Stable across repeated calls (no randomness / LLM).
    assert SearchEngine.classify_confidence(direct=False, has_direct_signals=True) == CONFIDENCE_INFERENCE
    assert SearchEngine.classify_confidence(direct=False, has_direct_signals=False) == CONFIDENCE_GUESS


def test_episode_evidence_is_guess_without_event_store(tmp_path):
    """Without an event store we cannot confirm underpinning events, so a
    derived episode is classified GUESS (safe default), not INFERENCE."""
    db_path, artifact_id, txt_path = _index_single(tmp_path, "retrieval.txt", "retrieval pipeline content")

    event_store = EventStore(db_path)
    event_store.append_event(_make_event("e-g", artifact_id, txt_path, "2026-01-01T10:00:00+00:00"))
    episode_store = EpisodeStore(db_path)
    _build_episode_memory(db_path, event_store, episode_store, MemoryStore(db_path))

    # Note: NO event_store passed to the engine.
    engine = SearchEngine(
        ArtifactStore(db_path),
        episode_store=episode_store,
    )
    results = engine.search("retrieval")
    event_store.close()
    episode_store.close()

    assert results
    ep_items = [e for e in results[0].evidence if e["type"] == "episode"]
    assert ep_items
    assert ep_items[0]["confidence_type"] == CONFIDENCE_GUESS
