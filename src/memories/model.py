from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Memory:
    id: str
    episode_ids: List[str]
    event_ids: List[str]
    artifact_ids: List[int]
    start_ts: str
    end_ts: str
    title: str
    topics: List[str]
    confidence: float
    evidence: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "episode_ids": list(self.episode_ids),
            "event_ids": list(self.event_ids),
            "artifact_ids": list(self.artifact_ids),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "title": self.title,
            "topics": list(self.topics),
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Memory":
        return Memory(
            id=data["id"],
            episode_ids=list(data["episode_ids"]),
            event_ids=list(data["event_ids"]),
            artifact_ids=list(data["artifact_ids"]),
            start_ts=data["start_ts"],
            end_ts=data["end_ts"],
            title=data["title"],
            topics=list(data["topics"]),
            confidence=float(data["confidence"]),
            evidence=[dict(item) for item in data["evidence"]],
        )
