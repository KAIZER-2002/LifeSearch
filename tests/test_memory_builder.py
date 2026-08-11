import pytest

from src.events.model import Event
from src.memories.builder import MemoryBuilder
from src.memories.model import Memory
from src.episodes.model import Episode


def make_event(event_id: str, event_type: str, timestamp: str, artifact_id=None, path=None, tags=None):
    payload = {}
    if path is not None:
        payload["path"] = path
    if tags is not None:
        payload["tags"] = tags
    return Event(
        id=event_id,
        type=event_type,
        timestamp=timestamp,
        source="test",
        source_kind="simulated",
        artifact_id=artifact_id,
        payload=payload,
    )


def make_episode(event_ids, artifact_ids=None, start_ts="2026-01-01T10:00:00+00:00", end_ts="2026-01-01T10:05:00+00:00", grouping_confidence=0.5):
    return Episode(
        id="episode1",
        start_ts=start_ts,
        end_ts=end_ts,
        event_ids=list(event_ids),
        artifact_ids=list(artifact_ids or []),
        grouping_confidence=grouping_confidence,
        title="",
        evidence=[],
    )


def test_empty_episode_returns_none():
    builder = MemoryBuilder(event_provider={})
    assert builder.build(make_episode([])) is None


def test_basic_episode_to_memory():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/qdrant.pdf"),
        "e2": make_event("e2", "FILE_MODIFIED", "2026-01-01T10:04:00+00:00", artifact_id=1, path="/project/qdrant.pdf"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1", "e2"], artifact_ids=[1], grouping_confidence=0.8)

    memory = builder.build(episode)
    assert isinstance(memory, Memory)
    assert memory.id == "memory_episode1"
    assert memory.episode_ids == ["episode1"]
    assert memory.event_ids == ["e1", "e2"]
    assert memory.artifact_ids == [1]
    assert memory.title == "Activity involving qdrant.pdf"
    assert memory.topics == ["qdrant"]
    assert 0.0 <= memory.confidence <= 1.0
    assert memory.evidence[0]["episode_id"] == "episode1"


def test_multiple_filenames_builds_combined_title():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/qdrant.pdf"),
        "e2": make_event("e2", "FILE_MODIFIED", "2026-01-01T10:04:00+00:00", artifact_id=2, path="/project/retrieval.py"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1", "e2"], artifact_ids=[1, 2], grouping_confidence=0.7)

    memory = builder.build(episode)
    assert memory.title == "Activity involving qdrant.pdf and retrieval.py"
    assert memory.topics == ["qdrant", "retrieval"]


def test_no_artifact_ids_still_creates_memory():
    events = {
        "e1": make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=None, path="/workspace/retrieval.py"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1"], artifact_ids=[], grouping_confidence=0.6)

    memory = builder.build(episode)
    assert memory is not None
    assert memory.artifact_ids == []
    assert memory.title == "Activity involving retrieval.py"
    assert memory.topics == ["retrieval"]


def test_generic_directory_tokens_are_excluded():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=None, path="C:/Users/Swapnil/Downloads/vacation.pdf"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1"], artifact_ids=[], grouping_confidence=0.5)

    memory = builder.build(episode)
    assert memory.topics == ["vacation"]
    assert memory.title == "Activity involving vacation.pdf"


def test_file_extensions_are_not_topics():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=None, path="/project/data.qdrant.md"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1"], artifact_ids=[], grouping_confidence=0.5)

    memory = builder.build(episode)
    assert memory.topics == ["qdrant", "data"]
    assert memory.title == "Activity involving data.qdrant.md"


def test_duplicate_topics_are_removed_preserving_order():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=None, path="/project/qdrant_notes.pdf"),
        "e2": make_event("e2", "FILE_MODIFIED", "2026-01-01T10:04:00+00:00", artifact_id=None, path="/project/qdrant_summary.pdf"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1", "e2"], artifact_ids=[], grouping_confidence=0.5)

    memory = builder.build(episode)
    assert memory.topics == ["qdrant", "notes", "summary"]


def test_fallback_title_when_no_topic_evidence():
    event = Event(
        id="e1",
        type="APPLICATION_ACTIVE",
        timestamp="2026-01-01T10:00:00+00:00",
        source="test",
        source_kind="simulated",
        artifact_id=None,
        payload={"app": "Chrome"},
    )
    builder = MemoryBuilder(event_provider={"e1": event})
    episode = make_episode(["e1"], artifact_ids=[], start_ts="2026-01-01T10:00:00+00:00", end_ts="2026-01-01T10:05:00+00:00", grouping_confidence=0.5)

    memory = builder.build(episode)
    assert memory.title == "Activity — 10:00–10:05"
    assert memory.topics == []


def test_deterministic_memory_id():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/qdrant.pdf"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1"], artifact_ids=[1], grouping_confidence=0.7)

    memory = builder.build(episode)
    assert memory.id == "memory_episode1"


def test_missing_event_references_do_not_crash():
    events = {
        "e1": make_event("e1", "FILE_OPENED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/qdrant.pdf"),
    }
    builder = MemoryBuilder(event_provider=events)
    episode = make_episode(["e1", "e2"], artifact_ids=[1], grouping_confidence=0.7)

    memory = builder.build(episode)
    assert memory is not None
    assert memory.event_ids == ["e1", "e2"]
