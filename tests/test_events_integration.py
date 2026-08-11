import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.events.source import FilesystemEventSource, SimulatedEventSource
from src.events.store import EventStore


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def test_event_lifecycle_integration():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        file_path = os.path.join(temp_dir, "note.txt")
        create_text_file(file_path, "initial content")

        with ArtifactStore(db_path) as artifact_store:
            extractor = Extractor()
            scanner = ArtifactScanner(artifact_store, extractor)
            scanner.index_folder(temp_dir)
            artifact = artifact_store.get_artifact_by_path(file_path)
            assert artifact is not None
            artifact_id = int(artifact["id"])

        with EventStore(db_path) as event_store:
            with ArtifactStore(db_path) as artifact_store:
                fs_source = FilesystemEventSource(artifact_store, event_store)
                creation_events = fs_source.generate_events(temp_dir)
                assert len(creation_events) == 1
                assert creation_events[0].type == "FILE_CREATED"
                assert creation_events[0].artifact_id == artifact_id
                event_store.append_event(creation_events[0])

            create_text_file(file_path, "updated content")
            os.utime(file_path, None)
            with ArtifactStore(db_path) as artifact_store:
                fs_source = FilesystemEventSource(artifact_store, event_store)
                modified_events = fs_source.generate_events(temp_dir)
                assert any(event.type == "FILE_MODIFIED" for event in modified_events)
                for event in modified_events:
                    event_store.append_event(event)

            simulated = SimulatedEventSource(
                [
                    {
                        "id": "sim-opened",
                        "type": "FILE_OPENED",
                        "timestamp": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
                        "artifact_id": artifact_id,
                        "payload": {"path": file_path},
                        "event_confidence": 0.75,
                    }
                ]
            )
            simulated_events = simulated.generate_events()
            assert len(simulated_events) == 1
            assert simulated_events[0].source_kind == "simulated"
            event_store.append_event(simulated_events[0])

            start_ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
            end_ts = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
            events_in_range = event_store.get_events_in_time_range(start_ts, end_ts)
            assert any(event.type == "FILE_CREATED" for event in events_in_range)
            assert any(event.type == "FILE_OPENED" for event in events_in_range)

            artifact_events = event_store.get_events_for_artifact(artifact_id)
            assert any(event.type == "FILE_CREATED" for event in artifact_events)
            assert any(event.type == "FILE_OPENED" for event in artifact_events)

            assert any(event.source == "SimulatedEventSource" for event in artifact_events)

            os.remove(file_path)
            with ArtifactStore(db_path) as artifact_store:
                fs_source = FilesystemEventSource(artifact_store, event_store)
                deletion_events = fs_source.generate_events(temp_dir)
                assert any(event.type == "FILE_DELETED" for event in deletion_events)
                for event in deletion_events:
                    event_store.append_event(event)

            artifact_events = event_store.get_events_for_artifact(artifact_id)
            assert any(event.type == "FILE_DELETED" for event in artifact_events)
