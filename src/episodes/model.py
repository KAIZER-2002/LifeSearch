from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class EvidenceSignal:
    name: str
    present: bool
    contribution: float

@dataclass(frozen=True)
class EpisodeDecision:
    previous_event_id: str
    current_event_id: str
    gap_seconds: float
    signals: List[EvidenceSignal]
    score: float
    threshold: float
    decision: str
    reason: Optional[str] = None

@dataclass(frozen=True)
class Episode:
    id: str
    start_ts: str
    end_ts: str
    event_ids: List[str]
    artifact_ids: List[int]
    grouping_confidence: float
    title: str
    evidence: List[EpisodeDecision]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "event_ids": self.event_ids,
            "artifact_ids": self.artifact_ids,
            "grouping_confidence": self.grouping_confidence,
            "title": self.title,
            "evidence": [
                {
                    "previous_event_id": d.previous_event_id,
                    "current_event_id": d.current_event_id,
                    "gap_seconds": d.gap_seconds,
                    "signals": [
                        {"name": s.name, "present": s.present, "contribution": s.contribution}
                        for s in d.signals
                    ],
                    "score": d.score,
                    "threshold": d.threshold,
                    "decision": d.decision,
                    "reason": d.reason,
                }
                for d in self.evidence
            ],
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Episode":
        return Episode(
            id=data["id"],
            start_ts=data["start_ts"],
            end_ts=data["end_ts"],
            event_ids=list(data["event_ids"]),
            artifact_ids=list(data["artifact_ids"]),
            grouping_confidence=float(data["grouping_confidence"]),
            title=data["title"],
            evidence=[
                EpisodeDecision(
                    previous_event_id=item["previous_event_id"],
                    current_event_id=item["current_event_id"],
                    gap_seconds=float(item["gap_seconds"]),
                    signals=[
                        EvidenceSignal(**signal) for signal in item["signals"]
                    ],
                    score=float(item["score"]),
                    threshold=float(item["threshold"]),
                    decision=item["decision"],
                    reason=item.get("reason"),
                )
                for item in data["evidence"]
            ],
        )
