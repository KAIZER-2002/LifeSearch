import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(value: str) -> str:
    if not value:
        return _now_iso()

    timestamp = value.strip()
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO 8601 timestamp: {value}") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.isoformat()


SUPPORTED_SOURCE_KINDS = {"filesystem", "simulated"}

SUPPORTED_FILESYSTEM_EVENT_TYPES = {
    "FILE_CREATED",
    "FILE_MODIFIED",
    "FILE_DELETED",
    "SCREENSHOT_DISCOVERED",
}

SUPPORTED_SIMULATED_EVENT_TYPES = {
    "FILE_OPENED",
    "FILE_DOWNLOADED",
    "APPLICATION_ACTIVE",
}


@dataclass(frozen=True)
class Event:
    id: str
    type: str
    timestamp: str
    source: str
    source_kind: str
    artifact_id: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    event_confidence: float = 1.0
    recorded_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", self.id or str(uuid.uuid4()))
        object.__setattr__(self, "timestamp", _normalize_timestamp(self.timestamp or _now_iso()))
        object.__setattr__(self, "recorded_at", _normalize_timestamp(self.recorded_at or _now_iso()))

        if self.source_kind not in SUPPORTED_SOURCE_KINDS:
            raise ValueError(f"Unsupported source_kind: {self.source_kind}")

        if not (0.0 <= self.event_confidence <= 1.0):
            raise ValueError("event_confidence must be between 0.0 and 1.0")

        try:
            json.dumps(self.payload, ensure_ascii=False)
        except TypeError as exc:
            raise TypeError("payload must be JSON serializable") from exc

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "timestamp": self.timestamp,
            "source": self.source,
            "source_kind": self.source_kind,
            "artifact_id": self.artifact_id,
            "payload": self.payload,
            "event_confidence": self.event_confidence,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        return cls(
            id=data.get("id") or str(uuid.uuid4()),
            type=data["type"],
            timestamp=data.get("timestamp") or _now_iso(),
            source=data["source"],
            source_kind=data["source_kind"],
            artifact_id=data.get("artifact_id"),
            payload=data.get("payload") or {},
            event_confidence=float(data.get("event_confidence", 1.0)),
            recorded_at=data.get("recorded_at") or _now_iso(),
        )

    @classmethod
    def from_row(cls, row: Any) -> "Event":
        payload = json.loads(row["payload"] or "{}")
        return cls(
            id=row["id"],
            type=row["type"],
            timestamp=row["timestamp"],
            source=row["source"],
            source_kind=row["source_kind"],
            artifact_id=row["artifact_id"],
            payload=payload,
            event_confidence=row["event_confidence"],
            recorded_at=row["recorded_at"],
        )
