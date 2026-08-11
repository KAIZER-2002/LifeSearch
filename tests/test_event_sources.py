import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.events.source import FilesystemEventSource, SimulatedEventSource
from src.events.store import EventStore


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def test_filesystem_event_source_creates_file_events():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as artifact_store:
            extractor = Extractor()
            scanner = ArtifactScanner(artifact_store, extractor)
            file_path = os.path.join(temp_dir, "note.txt")
            create_text_file(file_path, "first version")
            scanner.index_folder(temp_dir)

        with EventStore(db_path) as event_store:
            with ArtifactStore(db_path) as artifact_store:
                source = FilesystemEventSource(artifact_store, event_store)
                events = source.generate_events(temp_dir)
                assert any(event.type == "FILE_CREATED" for event in events)
                assert all(event.source_kind == "filesystem" for event in events)

                create_text_file(file_path, "second version")
                events = source.generate_events(temp_dir)
                assert any(event.type == "FILE_MODIFIED" for event in events)

                os.remove(file_path)
                events = source.generate_events(temp_dir)
                assert any(event.type == "FILE_DELETED" for event in events)


def test_simulated_event_source_generates_simulated_events():
    definitions = [
        {
            "id": "evt-opened",
            "type": "FILE_OPENED",
            "timestamp": "2026-08-12T00:00:00+00:00",
            "artifact_id": 1,
            "payload": {"path": "/tmp/file.txt"},
            "event_confidence": 0.8,
        }
    ]
    source = SimulatedEventSource(definitions)
    events = source.generate_events()
    assert len(events) == 1
    assert events[0].type == "FILE_OPENED"
    assert events[0].source_kind == "simulated"
