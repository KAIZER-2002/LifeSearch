import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from src.artifacts.store import ArtifactStore, _format_timestamp
from .model import (
    Event,
    SUPPORTED_SIMULATED_EVENT_TYPES,
)
from .store import EventStore


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventSource(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def source_kind(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_events(self, *args, **kwargs) -> List[Event]:
        raise NotImplementedError


class FilesystemEventSource(EventSource):
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
        ".docx",
        ".png",
        ".jpg",
        ".jpeg",
    }
    SCREENSHOT_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    EXTENSION_TO_MIME = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    def __init__(self, artifact_store: ArtifactStore, event_store: Optional[EventStore] = None):
        self.artifact_store = artifact_store
        self.event_store = event_store

    @property
    def source_name(self) -> str:
        return "FilesystemEventSource"

    @property
    def source_kind(self) -> str:
        return "filesystem"

    def generate_events(self, folder_path: str) -> List[Event]:
        folder_path = os.path.abspath(folder_path)
        current_paths: List[str] = []
        events: List[Event] = []

        for root, _dirs, files in os.walk(folder_path):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in self.SUPPORTED_EXTENSIONS:
                    continue

                path = os.path.abspath(os.path.join(root, file_name))
                current_paths.append(path)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue

                size = stat.st_size
                modified_at = _format_timestamp(stat.st_mtime)
                artifact = self.artifact_store.get_artifact_by_path(path)
                if artifact is None:
                    event_type = (
                        "SCREENSHOT_DISCOVERED"
                        if extension in self.SCREENSHOT_EXTENSIONS
                        else "FILE_CREATED"
                    )
                    events.append(
                        Event(
                            id="",
                            type=event_type,
                            timestamp=_now_iso(),
                            source=self.source_name,
                            source_kind=self.source_kind,
                            artifact_id=None,
                            payload={
                                "path": path,
                                "mime_type": self.EXTENSION_TO_MIME.get(extension),
                                "created_at": _format_timestamp(stat.st_ctime),
                                "modified_at": modified_at,
                                "size": size,
                            },
                        )
                    )
                elif artifact["size"] != size or artifact["modified_at"] != modified_at:
                    events.append(
                        Event(
                            id="",
                            type="FILE_MODIFIED",
                            timestamp=_now_iso(),
                            source=self.source_name,
                            source_kind=self.source_kind,
                            artifact_id=int(artifact["id"]),
                            payload={
                                "path": path,
                                "size": size,
                                "modified_at": modified_at,
                            },
                        )
                    )
                elif self._should_emit_initial_creation_event(artifact, extension):
                    event_type = (
                        "SCREENSHOT_DISCOVERED"
                        if extension in self.SCREENSHOT_EXTENSIONS
                        else "FILE_CREATED"
                    )
                    events.append(
                        Event(
                            id="",
                            type=event_type,
                            timestamp=_now_iso(),
                            source=self.source_name,
                            source_kind=self.source_kind,
                            artifact_id=int(artifact["id"]),
                            payload={
                                "path": path,
                                "mime_type": self.EXTENSION_TO_MIME.get(extension),
                                "created_at": artifact["created_at"],
                                "modified_at": artifact["modified_at"],
                                "size": artifact["size"],
                            },
                        )
                    )

        known_artifacts = self.artifact_store.list_artifacts_in_folder(folder_path, include_missing=True)
        existing_paths = set(current_paths)
        for artifact in known_artifacts:
            artifact_path = artifact["path"]
            if artifact_path not in existing_paths and not artifact["missing"]:
                if not self._has_prior_deletion_event(int(artifact["id"])):
                    events.append(
                        Event(
                            id="",
                            type="FILE_DELETED",
                            timestamp=_now_iso(),
                            source=self.source_name,
                            source_kind=self.source_kind,
                            artifact_id=int(artifact["id"]),
                            payload={"path": artifact_path},
                        )
                    )

        return events

    def _should_emit_initial_creation_event(self, artifact: Any, extension: str) -> bool:
        if self.event_store is None:
            return False
        if artifact is None:
            return False
        if extension in self.SCREENSHOT_EXTENSIONS:
            creation_types = {"SCREENSHOT_DISCOVERED"}
        else:
            creation_types = {"FILE_CREATED"}
        return not any(
            event.type in creation_types
            for event in self.event_store.get_events_for_artifact(int(artifact["id"]))
        )

    def _has_prior_deletion_event(self, artifact_id: int) -> bool:
        if self.event_store is None:
            return False
        return any(
            event.type == "FILE_DELETED"
            for event in self.event_store.get_events_for_artifact(artifact_id)
        )


class SimulatedEventSource(EventSource):
    def __init__(self, event_definitions: Iterable[Dict[str, Any]]):
        self.event_definitions = list(event_definitions)

    @property
    def source_name(self) -> str:
        return "SimulatedEventSource"

    @property
    def source_kind(self) -> str:
        return "simulated"

    def generate_events(self) -> List[Event]:
        events: List[Event] = []
        for event_definition in self.event_definitions:
            event_type = event_definition["type"]
            if event_type not in SUPPORTED_SIMULATED_EVENT_TYPES:
                raise ValueError(f"Unsupported simulated event type: {event_type}")
            events.append(
                Event(
                    id=event_definition.get("id", ""),
                    type=event_type,
                    timestamp=event_definition.get("timestamp") or _now_iso(),
                    source=self.source_name,
                    source_kind=self.source_kind,
                    artifact_id=event_definition.get("artifact_id"),
                    payload=event_definition.get("payload", {}),
                    event_confidence=float(event_definition.get("event_confidence", 1.0)),
                )
            )
        return events
