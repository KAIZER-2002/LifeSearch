import pytest

from src.episodes.engine import EpisodeEngine
from src.events.model import Event


def make_event(event_id: str, event_type: str, timestamp: str, artifact_id=None, path=None, app=None):
    payload = {}
    if path is not None:
        payload["path"] = path
    if app is not None:
        payload["app"] = app
    return Event(
        id=event_id,
        type=event_type,
        timestamp=timestamp,
        source="test",
        source_kind="simulated",
        artifact_id=artifact_id,
        payload=payload,
    )


def test_same_artifact_merges():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"),
        make_event("e2", "FILE_OPENED", "2026-01-01T10:05:00+00:00", artifact_id=1, path="/project/retrieval.py"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 1
    assert episodes[0].event_ids == ["e1", "e2"]
    assert episodes[0].artifact_ids == [1]
    assert episodes[0].title == "Activity involving retrieval.py"


def test_same_broad_directory_does_not_merge_alone():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_DOWNLOADED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="C:/Users/Swapnil/Downloads/a.pdf"),
        make_event("e2", "FILE_DOWNLOADED", "2026-01-01T10:03:00+00:00", artifact_id=2, path="C:/Users/Swapnil/Downloads/vacation.jpg"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 2
    assert episodes[0].event_ids == ["e1"]
    assert episodes[1].event_ids == ["e2"]


def test_same_specific_project_directory_can_merge():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_CREATED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/workspace/JyotishAI/retrieval.py"),
        make_event("e2", "FILE_OPENED", "2026-01-01T10:04:00+00:00", artifact_id=2, path="/workspace/JyotishAI/qdrant_notes.md"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 1
    assert episodes[0].artifact_ids == [1, 2]


def test_same_application_alone_does_not_merge():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "APPLICATION_ACTIVE", "2026-01-01T10:00:00+00:00", app="Chrome"),
        make_event("e2", "APPLICATION_ACTIVE", "2026-01-01T10:05:00+00:00", app="Chrome"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 2


def test_missing_artifact_reference_can_group_on_path():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=None, path="/workspace/retrieval.py"),
        make_event("e2", "FILE_OPENED", "2026-01-01T10:02:00+00:00", artifact_id=None, path="/workspace/retrieval.py"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 1
    assert episodes[0].artifact_ids == []


def test_out_of_order_input_is_ordered():
    engine = EpisodeEngine()
    events = [
        make_event("e2", "FILE_OPENED", "2026-01-01T10:05:00+00:00", artifact_id=1, path="/project/retrieval.py"),
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 1
    assert episodes[0].event_ids == ["e1", "e2"]


def test_duplicate_event_ids_are_ignored():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"),
        make_event("e1", "FILE_OPENED", "2026-01-01T10:05:00+00:00", artifact_id=1, path="/project/retrieval.py"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 1
    assert episodes[0].event_ids == ["e1"]


def test_max_gap_always_splits():
    engine = EpisodeEngine(max_episode_gap_seconds=60)
    events = [
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"),
        make_event("e2", "FILE_OPENED", "2026-01-01T10:02:30+00:00", artifact_id=1, path="/project/retrieval.py"),
    ]

    episodes = engine.group_events(events)
    assert len(episodes) == 2


def test_deterministic_episode_ids():
    engine = EpisodeEngine()
    events = [
        make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"),
        make_event("e2", "FILE_OPENED", "2026-01-01T10:05:00+00:00", artifact_id=1, path="/project/retrieval.py"),
    ]

    first = engine.group_events(events)
    second = engine.group_events(list(reversed(events)))
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].id == second[0].id
