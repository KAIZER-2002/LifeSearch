import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.events.model import Event
from src.events.store import EventStore


def test_event_store_append_and_query():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with EventStore(db_path) as store:
            event = Event(
                id="event-1",
                type="FILE_CREATED",
                timestamp="2026-08-12T00:00:00+00:00",
                source="FilesystemEventSource",
                source_kind="filesystem",
                artifact_id=1,
                payload={"path": "/tmp/file.txt"},
                event_confidence=0.9,
            )
            store.append_event(event)
            loaded = store.get_event("event-1")
            assert loaded is not None
            assert loaded.id == "event-1"
            assert loaded.type == "FILE_CREATED"
            assert loaded.source_kind == "filesystem"
            assert loaded.artifact_id == 1
            assert loaded.payload == {"path": "/tmp/file.txt"}

            events_by_artifact = store.get_events_for_artifact(1)
            assert len(events_by_artifact) == 1
            assert events_by_artifact[0].id == "event-1"

            events_by_type = store.get_events_by_type("FILE_CREATED")
            assert len(events_by_type) == 1

            recent = store.get_recent_events(limit=1)
            assert recent[0].id == "event-1"


def test_event_store_duplicate_id_rejected():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with EventStore(db_path) as store:
            event = Event(
                id="duplicate-id",
                type="FILE_CREATED",
                timestamp="2026-08-12T00:00:00+00:00",
                source="FilesystemEventSource",
                source_kind="filesystem",
                artifact_id=None,
                payload={"path": "/tmp/file.txt"},
            )
            store.append_event(event)
            try:
                store.append_event(event)
            except ValueError as exc:
                assert "already exists" in str(exc)
            else:
                raise AssertionError("Expected duplicate event id to be rejected")


def test_event_model_normalizes_timestamp_to_utc():
            event = Event(
                id="e-timezone",
                type="FILE_CREATED",
                timestamp="2026-08-12T02:00:00+02:00",
                source="FilesystemEventSource",
                source_kind="filesystem",
                artifact_id=None,
                payload={"path": "/tmp/file.txt"},
            )
            assert event.timestamp == "2026-08-12T00:00:00+00:00"


def test_event_model_rejects_invalid_confidence():
            try:
                Event(
                    id="e-bad-confidence",
                    type="FILE_CREATED",
                    timestamp="2026-08-12T00:00:00+00:00",
                    source="FilesystemEventSource",
                    source_kind="filesystem",
                    artifact_id=None,
                    payload={"path": "/tmp/file.txt"},
                    event_confidence=1.5,
                )
            except ValueError as exc:
                assert "event_confidence" in str(exc)
            else:
                raise AssertionError("Expected invalid confidence to raise ValueError")


def test_query_events_in_time_range():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with EventStore(db_path) as store:
            event1 = Event(
                id="e1",
                type="FILE_CREATED",
                timestamp="2026-08-12T00:00:00+00:00",
                source="FilesystemEventSource",
                source_kind="filesystem",
                artifact_id=1,
                payload={},
            )
            event2 = Event(
                id="e2",
                type="FILE_MODIFIED",
                timestamp="2026-08-12T01:00:00+00:00",
                source="FilesystemEventSource",
                source_kind="filesystem",
                artifact_id=1,
                payload={},
            )
            store.append_event(event1)
            store.append_event(event2)

            results = store.get_events_in_time_range(
                "2026-08-12T00:30:00+00:00", "2026-08-12T02:00:00+00:00"
            )
            assert len(results) == 1
            assert results[0].id == "e2"


def test_get_events_inclusive_time_range():
            with tempfile.TemporaryDirectory() as temp_dir:
                db_path = os.path.join(temp_dir, "lifesearch.db")
                with EventStore(db_path) as store:
                    event1 = Event(
                        id="e3",
                        type="FILE_CREATED",
                        timestamp="2026-08-12T00:00:00+00:00",
                        source="FilesystemEventSource",
                        source_kind="filesystem",
                        artifact_id=2,
                        payload={},
                    )
                    event2 = Event(
                        id="e4",
                        type="FILE_MODIFIED",
                        timestamp="2026-08-12T00:00:00+00:00",
                        source="FilesystemEventSource",
                        source_kind="filesystem",
                        artifact_id=2,
                        payload={},
                    )
                    store.append_event(event1)
                    store.append_event(event2)

                    results = store.get_events_in_time_range(
                        "2026-08-12T00:00:00+00:00", "2026-08-12T00:00:00+00:00"
                    )
                    assert len(results) == 2


def test_malformed_payload_raises_type_error():
    try:
        Event(
            id="e-malformed",
            type="FILE_CREATED",
            timestamp="2026-08-12T00:00:00+00:00",
            source="FilesystemEventSource",
            source_kind="filesystem",
            artifact_id=None,
            payload={"not_json": {"bad_set"}},
        )
    except TypeError as exc:
        assert "payload must be JSON serializable" in str(exc)
    else:
        raise AssertionError("Expected non-json payload to raise TypeError")
