import pytest

"""
Tests for the Event store. These are skeletons intended to guide development.
Replace the import paths and implementations with the actual modules in src/ when available.
"""

try:
    from src.storage.event_store import EventStore
except Exception:
    EventStore = None


def make_sample_event():
    return {
        "id": "evt-sample-1",
        "type": "FileDownloaded",
        "timestamp": 1712750400000,
        "source": "test-sim",
        "artifact_id": "art-sample-1",
        "app": "Chrome",
        "raw_payload": {"url": "https://example.com/qdrant.pdf"},
        "metadata": {"path": "C:/Users/Test/Downloads/qdrant.pdf", "mime_type": "application/pdf"},
        "confidence": 0.99,
        "tags": ["download", "pdf"],
        "linked_entities": ["ent-qdrant"]
    }


@pytest.mark.skipif(EventStore is None, reason="EventStore not implemented")
def test_append_and_query_event(tmp_path):
    """Append an event then query by type and time range."""
    db_path = tmp_path / "events.db"
    store = EventStore(str(db_path))

    evt = make_sample_event()
    store.append_event(evt)

    results = store.query_events(event_type="FileDownloaded")
    assert any(r["id"] == evt["id"] for r in results), "Appended event should be returned by query"


@pytest.mark.skipif(EventStore is None, reason="EventStore not implemented")
def test_event_persistence(tmp_path):
    """Events should persist across new store instances backed by the same DB file."""
    db_path = tmp_path / "events.db"
    store1 = EventStore(str(db_path))
    evt = make_sample_event()
    store1.append_event(evt)
    store1.close()

    store2 = EventStore(str(db_path))
    results = store2.query_events(event_type="FileDownloaded")
    assert any(r["id"] == evt["id"] for r in results)
