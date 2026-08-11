import pytest

"""
Tests for Episode detection heuristics.
These tests assume an episode engine with API like:
  EpisodeEngine.add_event(event)
  EpisodeEngine.detect_episodes(from_ts=None, to_ts=None)

They are skeletons — adapt imports and APIs when implementing.
"""

try:
    from src.episode.engine import EpisodeEngine
except Exception:
    EpisodeEngine = None


def sample_session_events():
    # A simulated sequence: browse -> download -> edit -> screenshot
    base = 1712750400000
    return [
        {"id": "e1", "type": "BrowserVisited", "timestamp": base + 0, "metadata": {"url": "https://blog/qdrant"}, "tags": ["browse"], "linked_entities": ["ent-qdrant"]},
        {"id": "e2", "type": "FileDownloaded", "timestamp": base + 5*60*1000, "artifact_id": "a1", "metadata": {"path": "C:/.../qdrant_notes.pdf"}, "tags": ["download"]},
        {"id": "e3", "type": "AppOpened", "timestamp": base + 15*60*1000, "app": "VSCode"},
        {"id": "e4", "type": "FileModified", "timestamp": base + 20*60*1000, "artifact_id": "a2", "metadata": {"path": "C:/.../retrieval.py"}},
        {"id": "e5", "type": "ScreenshotCaptured", "timestamp": base + 30*60*1000, "artifact_id": "s1", "tags": ["error", "mongo"]},
    ]


@pytest.mark.skipif(EpisodeEngine is None, reason="EpisodeEngine not implemented")
def test_episode_grouping_simple(tmp_path):
    """A simple session should be grouped into a single episode by heuristics."""
    engine = EpisodeEngine(db_path=str(tmp_path / "episodes.db"))
    events = sample_session_events()
    for e in events:
        engine.add_event(e)

    episodes = engine.detect_episodes()
    assert len(episodes) >= 1, "At least one episode should be detected"

    # Find episode that contains the download event
    found = [ep for ep in episodes if any(ev["id"] == "e2" for ev in ep["events"])]
    assert found, "Download event should be included in an episode"
    ep = found[0]
    assert "ent-qdrant" in ep.get("dominant_entities", []) or "qdrant" in ep.get("inferred_title", "").lower()


@pytest.mark.skipif(EpisodeEngine is None, reason="EpisodeEngine not implemented")
def test_episode_confidence_and_bounds(tmp_path):
    engine = EpisodeEngine(db_path=str(tmp_path / "episodes.db"))
    events = sample_session_events()
    for e in events:
        engine.add_event(e)

    episodes = engine.detect_episodes()
    ep = episodes[0]
    assert ep["start_ts"] <= events[0]["timestamp"]
    assert ep["end_ts"] >= events[-1]["timestamp"]
    assert 0.0 <= ep.get("confidence", 1.0) <= 1.0
