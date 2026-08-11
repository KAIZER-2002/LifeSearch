from src.episodes.engine import EpisodeEngine
from src.memories.builder import MemoryBuilder
from src.memories.store import MemoryStore
from src.events.model import Event
from src.events.store import EventStore


def make_event(event_id: str, event_type: str, timestamp: str, artifact_id=None, path=None):
    payload = {}
    if path is not None:
        payload["path"] = path
    return Event(
        id=event_id,
        type=event_type,
        timestamp=timestamp,
        source="simulated",
        source_kind="simulated",
        artifact_id=artifact_id,
        payload=payload,
    )


def test_memory_builder_integration_and_rebuild(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    event_store = EventStore(db_path=db_path)
    memory_store = MemoryStore(db_path=str(tmp_path / "memories.db"))
    engine = EpisodeEngine(max_episode_gap_seconds=900)
    builder = MemoryBuilder(event_provider=event_store)

    event_store.append_event(make_event("e1", "FILE_OPENED", "2026-01-01T20:00:00+00:00", artifact_id=1, path="C:/project/qdrant.pdf"))
    event_store.append_event(make_event("e2", "FILE_MODIFIED", "2026-01-01T20:04:00+00:00", artifact_id=1, path="C:/project/qdrant.pdf"))
    event_store.append_event(make_event("e3", "FILE_MODIFIED", "2026-01-01T20:07:00+00:00", artifact_id=2, path="C:/project/retrieval.py"))
    event_store.append_event(make_event("e4", "SCREENSHOT_DISCOVERED", "2026-01-01T20:12:00+00:00", artifact_id=None, path="C:/project/screenshot.png"))
    event_store.append_event(make_event("e5", "FILE_OPENED", "2026-01-01T20:16:00+00:00", artifact_id=1, path="C:/project/qdrant.pdf"))
    event_store.append_event(make_event("e6", "SCREENSHOT_DISCOVERED", "2026-01-01T21:30:00+00:00", artifact_id=3, path="C:/Users/Swapnil/Downloads/vacation.jpg"))
    event_store.append_event(make_event("e7", "FILE_MODIFIED", "2026-01-01T21:35:00+00:00", artifact_id=3, path="C:/Users/Swapnil/Downloads/vacation.jpg"))

    episodes = engine.group_events(event_store.query_events(start_ts="2026-01-01T19:00:00+00:00", end_ts="2026-01-01T22:00:00+00:00"))
    assert len(episodes) == 2

    memories = [memory for episode in episodes if (memory := builder.build(episode)) is not None]
    assert len(memories) == 2
    memory_store.save_memories(memories, range_start_ts="2026-01-01T19:00:00+00:00", range_end_ts="2026-01-01T22:00:00+00:00")

    persisted = memory_store.get_memories()
    assert len(persisted) == 2
    first_memory = persisted[0]
    assert first_memory.title == "Activity involving qdrant.pdf and retrieval.py"
    assert first_memory.topics == ["qdrant", "retrieval"]

    memory_store.save_memories(memories, range_start_ts="2026-01-01T19:00:00+00:00", range_end_ts="2026-01-01T22:00:00+00:00")
    persisted_again = memory_store.get_memories()
    assert len(persisted_again) == 2
    assert [memory.id for memory in persisted] == [memory.id for memory in persisted_again]

    unrelated_event = make_event("e8", "FILE_OPENED", "2026-01-01T23:00:00+00:00", artifact_id=4, path="C:/project/other.pdf")
    event_store.append_event(unrelated_event)
    unrelated_episode = engine.group_events([unrelated_event])
    unrelated_memory = [memory for episode in unrelated_episode if (memory := builder.build(episode)) is not None]
    memory_store.save_memories(unrelated_memory, range_start_ts="2026-01-01T23:00:00+00:00", range_end_ts="2026-01-01T23:10:00+00:00")
    assert len(memory_store.get_memories()) == 3

    event_store.close()
    memory_store.close()
