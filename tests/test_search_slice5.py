"""
tests/test_search_slice5.py

Slice 5 tests: unified search over artifacts, episodes, and memories.
Tests verify:
  1. SearchResult returned (not bare dict)
  2. Backward-compatible dict-style access
  3. Results without context stores → empty episodes/memories/evidence
  4. Results with EpisodeStore → episode context enriched
  5. Results with EpisodeStore + MemoryStore → memory context enriched
  6. Evidence typed dicts with "type" key
  7. Artifact not in any episode → empty context
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
from src.search.engine import SearchEngine
from src.search.result import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _make_event(event_id: str, artifact_id, path: str, timestamp: str) -> Event:
    return Event(
        id=event_id,
        type="FILE_OPENED",
        timestamp=timestamp,
        source="test",
        source_kind="simulated",
        artifact_id=artifact_id,
        payload={"path": path},
    )


# ---------------------------------------------------------------------------
# 1. SearchResult is returned (not bare dict)
# ---------------------------------------------------------------------------

def test_search_result_type():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        with ArtifactStore(db) as store:
            scanner = ArtifactScanner(store, Extractor())
            _write_text(os.path.join(tmp, "notes.txt"), "qdrant vector search")
            scanner.index_folder(tmp)
            engine = SearchEngine(store)
            results = engine.search("qdrant")
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)


# ---------------------------------------------------------------------------
# 2. Backward-compatible dict-style access
# ---------------------------------------------------------------------------

def test_search_result_dict_access_backward_compatible():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        with ArtifactStore(db) as store:
            scanner = ArtifactScanner(store, Extractor())
            _write_text(os.path.join(tmp, "notes.txt"), "qdrant content")
            scanner.index_folder(tmp)
            engine = SearchEngine(store)
            results = engine.search("qdrant")
        assert len(results) == 1
        result = results[0]
        # Attribute access
        assert result.file_name == "notes.txt"
        # Dict-style access
        assert result["file_name"] == result.file_name
        # __contains__
        assert "file_name" in result
        assert "nonexistent_key" not in result
        # .get()
        assert result.get("file_name") == "notes.txt"
        assert result.get("nonexistent_key", "default") == "default"


# ---------------------------------------------------------------------------
# 3. No stores → empty context
# ---------------------------------------------------------------------------

def test_search_with_no_stores_returns_results_without_context():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        with ArtifactStore(db) as store:
            scanner = ArtifactScanner(store, Extractor())
            _write_text(os.path.join(tmp, "guide.txt"), "retrieval pipeline notes")
            scanner.index_folder(tmp)
            engine = SearchEngine(store)  # no episode_store, no memory_store
            results = engine.search("retrieval")
        assert len(results) == 1
        result = results[0]
        assert result.episodes == []
        assert result.memories == []
        assert result.evidence == []


# ---------------------------------------------------------------------------
# 4. With EpisodeStore → episode context enriched
# ---------------------------------------------------------------------------

def test_search_enriches_with_episode(tmp_path):
    db_path = str(tmp_path / "ls.db")
    txt_path = str(tmp_path / "qdrant_notes.txt")
    _write_text(txt_path, "qdrant notes content")

    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(str(tmp_path))

    artifact_row = artifact_store.get_artifact_by_path(txt_path)
    assert artifact_row is not None
    artifact_id = int(artifact_row["id"])

    event_store = EventStore(db_path)
    ev = _make_event("e1", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)

    engine_ep = EpisodeEngine()
    episode_store = EpisodeStore(db_path)
    episodes = engine_ep.detect_and_persist(
        event_store, episode_store,
        start_ts="2026-01-01T09:00:00+00:00",
        end_ts="2026-01-01T11:00:00+00:00",
    )
    assert len(episodes) == 1

    search_engine = SearchEngine(artifact_store, episode_store)
    results = search_engine.search("qdrant")
    artifact_store.close()

    assert len(results) == 1
    result = results[0]
    assert len(result.episodes) == 1
    assert result.episodes[0]["id"] == episodes[0].id
    assert result.memories == []


# ---------------------------------------------------------------------------
# 5. With EpisodeStore + MemoryStore → memory context enriched
# ---------------------------------------------------------------------------

def test_search_enriches_with_memory(tmp_path):
    db_path = str(tmp_path / "ls.db")
    txt_path = str(tmp_path / "qdrant_notes.txt")
    _write_text(txt_path, "qdrant notes content")

    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(str(tmp_path))

    artifact_row = artifact_store.get_artifact_by_path(txt_path)
    artifact_id = int(artifact_row["id"])

    event_store = EventStore(db_path)
    ev = _make_event("e2", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)

    engine_ep = EpisodeEngine()
    episode_store = EpisodeStore(db_path)
    episodes = engine_ep.detect_and_persist(
        event_store, episode_store,
        start_ts="2026-01-01T09:00:00+00:00",
        end_ts="2026-01-01T11:00:00+00:00",
    )

    memory_store = MemoryStore(str(tmp_path / "memories.db"))
    builder = MemoryBuilder(event_provider=event_store)
    memories = [m for ep in episodes if (m := builder.build(ep)) is not None]
    memory_store.save_memories(
        memories,
        range_start_ts="2026-01-01T09:00:00+00:00",
        range_end_ts="2026-01-01T11:00:00+00:00",
    )

    search_engine = SearchEngine(artifact_store, episode_store, memory_store)
    results = search_engine.search("qdrant")
    artifact_store.close()

    assert len(results) == 1
    result = results[0]
    assert len(result.memories) == 1
    assert result.memories[0]["title"] != ""
    assert isinstance(result.memories[0]["topics"], list)


# ---------------------------------------------------------------------------
# 6. Evidence typed dicts with "type" key
# ---------------------------------------------------------------------------

def test_evidence_contains_typed_dicts(tmp_path):
    db_path = str(tmp_path / "ls.db")
    txt_path = str(tmp_path / "retrieval.txt")
    _write_text(txt_path, "retrieval pipeline content")

    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(str(tmp_path))

    artifact_row = artifact_store.get_artifact_by_path(txt_path)
    artifact_id = int(artifact_row["id"])

    event_store = EventStore(db_path)
    ev = _make_event("e3", artifact_id, txt_path, "2026-01-01T10:00:00+00:00")
    event_store.append_event(ev)

    engine_ep = EpisodeEngine()
    episode_store = EpisodeStore(db_path)
    engine_ep.detect_and_persist(
        event_store, episode_store,
        start_ts="2026-01-01T09:00:00+00:00",
        end_ts="2026-01-01T11:00:00+00:00",
    )

    search_engine = SearchEngine(artifact_store, episode_store)
    results = search_engine.search("retrieval")
    artifact_store.close()

    assert len(results) == 1
    evidence = results[0].evidence
    assert len(evidence) >= 1
    for item in evidence:
        assert "type" in item
        assert item["type"] in ("episode", "memory")


# ---------------------------------------------------------------------------
# 7. Empty query → empty results
# ---------------------------------------------------------------------------

def test_empty_query_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        with ArtifactStore(db) as store:
            scanner = ArtifactScanner(store, Extractor())
            _write_text(os.path.join(tmp, "notes.txt"), "some content")
            scanner.index_folder(tmp)
            engine = SearchEngine(store)
            results = engine.search("")
        assert results == []


# ---------------------------------------------------------------------------
# Bonus: artifact not in any episode → empty context
# ---------------------------------------------------------------------------

def test_artifact_not_in_any_episode_returns_empty_context(tmp_path):
    db_path = str(tmp_path / "ls.db")
    txt_path = str(tmp_path / "orphan.txt")
    _write_text(txt_path, "orphan content unrelated")

    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(str(tmp_path))

    # Episode store has no episodes
    episode_store = EpisodeStore(db_path)

    search_engine = SearchEngine(artifact_store, episode_store)
    results = search_engine.search("orphan")
    artifact_store.close()

    assert len(results) == 1
    result = results[0]
    assert result.episodes == []
    assert result.evidence == []
