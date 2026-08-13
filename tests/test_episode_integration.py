from src.episodes.engine import EpisodeEngine
from src.episodes.store import EpisodeStore
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


def test_detect_and_persist_idempotence(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    event_store = EventStore(db_path=db_path)
    episode_store = EpisodeStore(db_path=db_path)
    engine = EpisodeEngine()

    event_store.append_event(make_event("e1", "FILE_MODIFIED", "2026-01-01T10:00:00+00:00", artifact_id=1, path="/project/retrieval.py"))
    event_store.append_event(make_event("e2", "FILE_OPENED", "2026-01-01T10:03:00+00:00", artifact_id=1, path="/project/retrieval.py"))
    event_store.append_event(make_event("e3", "FILE_DOWNLOADED", "2026-01-01T10:40:00+00:00", artifact_id=2, path="/project/notes.pdf"))

    episodes_first = engine.detect_and_persist(event_store, episode_store, start_ts="2026-01-01T09:00:00+00:00", end_ts="2026-01-01T11:00:00+00:00")
    assert len(episodes_first) == 2
    stored_first = episode_store.get_episodes()
    assert len(stored_first) == 2

    episodes_second = engine.detect_and_persist(event_store, episode_store, start_ts="2026-01-01T09:00:00+00:00", end_ts="2026-01-01T11:00:00+00:00")
    stored_second = episode_store.get_episodes()
    assert len(stored_second) == 2
    assert [ep.id for ep in stored_first] == [ep.id for ep in stored_second]

    event_store.close()
    episode_store.close()
