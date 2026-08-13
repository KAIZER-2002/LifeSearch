import json

from src.memories.model import Memory
from src.memories.store import MemoryStore


def make_memory(memory_id: str, episode_id: str, event_ids, artifact_ids, start_ts, end_ts):
    return Memory(
        id=memory_id,
        episode_ids=[episode_id],
        event_ids=list(event_ids),
        artifact_ids=list(artifact_ids),
        start_ts=start_ts,
        end_ts=end_ts,
        title="Activity involving test",
        topics=["test"],
        confidence=0.75,
        evidence=[
            {
                "episode_id": episode_id,
                "event_ids": list(event_ids),
                "artifact_ids": list(artifact_ids),
                "title_basis": "artifact_names",
                "topic_sources": [{"topic": "test", "source": "test.txt", "source_type": "filename"}],
                "grouping_confidence": 0.75,
            }
        ],
    )


def test_memory_store_roundtrip(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memories.db"))
    m1 = make_memory("memory_ep1", "ep1", ["e1", "e2"], [1], "2026-01-01T10:00:00+00:00", "2026-01-01T10:05:00+00:00")
    m2 = make_memory("memory_ep2", "ep2", ["e3"], [2], "2026-01-01T10:10:00+00:00", "2026-01-01T10:10:00+00:00")

    store.save_memories([m1, m2], range_start_ts=m1.start_ts, range_end_ts=m2.end_ts)
    memories = store.get_memories()
    assert len(memories) == 2
    assert memories[0].id == "memory_ep1"
    assert memories[1].id == "memory_ep2"

    memory = store.get_memory("memory_ep1")
    assert memory is not None
    assert memory.id == "memory_ep1"

    artifact_memories = store.get_memories_for_artifact(1)
    assert len(artifact_memories) == 1
    assert artifact_memories[0].id == "memory_ep1"

    episode_memories = store.get_memories_for_episode("ep2")
    assert len(episode_memories) == 1
    assert episode_memories[0].id == "memory_ep2"

    range_memories = store.get_memories_in_time_range("2026-01-01T10:00:00+00:00", "2026-01-01T10:07:00+00:00")
    assert len(range_memories) == 1
    assert range_memories[0].id == "memory_ep1"

    store.save_memories([m1, m2], range_start_ts=m1.start_ts, range_end_ts=m2.end_ts)
    assert len(store.get_memories()) == 2
    store.close()


def test_memory_store_range_deletion(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "memories.db"))
    m1 = make_memory("memory_ep1", "ep1", ["e1"], [1], "2026-01-01T10:00:00+00:00", "2026-01-01T10:05:00+00:00")
    m2 = make_memory("memory_ep2", "ep2", ["e2"], [2], "2026-01-01T11:00:00+00:00", "2026-01-01T11:05:00+00:00")

    store.save_memories([m1, m2])
    store.delete_memories_in_range("2026-01-01T09:00:00+00:00", "2026-01-01T10:30:00+00:00")
    remaining = store.get_memories()
    assert len(remaining) == 1
    assert remaining[0].id == "memory_ep2"
    store.close()
