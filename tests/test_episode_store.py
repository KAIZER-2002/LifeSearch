import json

from src.episodes.model import Episode, EpisodeDecision, EvidenceSignal
from src.episodes.store import EpisodeStore


def make_episode(episode_id: str, event_ids, artifact_ids, start_ts, end_ts):
    decision = EpisodeDecision(
        previous_event_id="e1",
        current_event_id="e2",
        gap_seconds=30.0,
        signals=[EvidenceSignal(name="same_artifact_id", present=True, contribution=1.0)],
        score=1.0,
        threshold=0.75,
        decision="MERGED",
    )
    return Episode(
        id=episode_id,
        start_ts=start_ts,
        end_ts=end_ts,
        event_ids=list(event_ids),
        artifact_ids=list(artifact_ids),
        grouping_confidence=1.0,
        title="Activity involving test",
        evidence=[decision],
    )


def test_episode_store_roundtrip(tmp_path):
    store = EpisodeStore(db_path=str(tmp_path / "episodes.db"))
    ep1 = make_episode("ep1", ["e1", "e2"], [1], "2026-01-01T10:00:00+00:00", "2026-01-01T10:05:00+00:00")
    ep2 = make_episode("ep2", ["e3"], [2], "2026-01-01T10:10:00+00:00", "2026-01-01T10:10:00+00:00")

    store.save_episodes([ep1, ep2], range_start_ts=ep1.start_ts, range_end_ts=ep2.end_ts)
    episodes = store.get_episodes()
    assert len(episodes) == 2
    assert episodes[0].id == "ep1"
    assert episodes[1].id == "ep2"

    artifact_episodes = store.get_episodes_for_artifact(1)
    assert len(artifact_episodes) == 1
    assert artifact_episodes[0].id == "ep1"

    range_episodes = store.get_episodes_in_time_range("2026-01-01T10:00:00+00:00", "2026-01-01T10:07:00+00:00")
    assert len(range_episodes) == 1
    assert range_episodes[0].id == "ep1"

    # Save again should not create duplicates and should replace same range
    store.save_episodes([ep1, ep2], range_start_ts=ep1.start_ts, range_end_ts=ep2.end_ts)
    episodes_after = store.get_episodes()
    assert len(episodes_after) == 2
    store.close()
