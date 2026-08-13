from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.episodes.model import Episode, EpisodeDecision, EvidenceSignal
from src.events.model import Event
from src.episodes.store import EpisodeStore

MAX_EPISODE_GAP_SECONDS = 30 * 60
MERGE_THRESHOLD = 0.75

EVENT_FAMILY_MAP = {
    "FILE_CREATED": "FILE_ACTIVITY",
    "FILE_MODIFIED": "FILE_ACTIVITY",
    "FILE_DELETED": "FILE_ACTIVITY",
    "FILE_OPENED": "FILE_INTERACTION",
    "FILE_DOWNLOADED": "FILE_INTERACTION",
    "SCREENSHOT_DISCOVERED": "SCREENSHOT",
    "APPLICATION_ACTIVE": "APPLICATION",
}

BROAD_DIRECTORIES = {
    "Downloads",
    "Documents",
    "Desktop",
    "Pictures",
    "Videos",
    "Music",
}

SCORING_WEIGHTS = {
    "same_artifact_id": 1.0,
    "same_specific_path": 0.6,
    "shared_basename": 0.45,
    "short_gap": 0.30,
    "same_event_family": 0.20,
    "application_continuity": 0.10,
}

MEANINGFUL_SIGNALS = {
    "same_artifact_id",
    "same_specific_path",
    "shared_basename",
    "same_event_family",
}

class EpisodeEngine:
    def __init__(self, max_episode_gap_seconds: int = MAX_EPISODE_GAP_SECONDS, merge_threshold: float = MERGE_THRESHOLD):
        self.max_episode_gap_seconds = max_episode_gap_seconds
        self.merge_threshold = merge_threshold

    @staticmethod
    def get_event_family(event_type: str) -> Optional[str]:
        return EVENT_FAMILY_MAP.get(event_type)

    @staticmethod
    def _parse_timestamp(timestamp: str) -> datetime:
        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _safe_path_components(path: Optional[str]) -> List[str]:
        if not path:
            return []
        normalized = Path(path).as_posix()
        return [segment for segment in normalized.split("/") if segment]

    @staticmethod
    def _file_basename(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        return Path(path).name

    @staticmethod
    def _common_path_prefix(a: str, b: str) -> List[str]:
        a_parts = EpisodeEngine._safe_path_components(a)
        b_parts = EpisodeEngine._safe_path_components(b)
        prefix = []
        for a_part, b_part in zip(a_parts, b_parts):
            if a_part != b_part:
                break
            prefix.append(a_part)
        return prefix

    @staticmethod
    def _is_broad_directory(path: str) -> bool:
        parts = EpisodeEngine._safe_path_components(path)
        return len(parts) == 1 and parts[0] in BROAD_DIRECTORIES

    @staticmethod
    def _same_specific_path(path_a: Optional[str], path_b: Optional[str]) -> bool:
        if not path_a or not path_b:
            return False
        prefix = EpisodeEngine._common_path_prefix(path_a, path_b)
        if len(prefix) < 2:
            return False
        last_shared = prefix[-1]
        if last_shared in BROAD_DIRECTORIES:
            return False
        return True

    @staticmethod
    def _extract_app(event: Event) -> Optional[str]:
        payload = event.payload or {}
        return payload.get("app") or payload.get("application")

    def _compute_signals(self, previous: Event, current: Event) -> List[EvidenceSignal]:
        previous_artifact = previous.artifact_id
        current_artifact = current.artifact_id
        same_artifact_id = previous_artifact is not None and previous_artifact == current_artifact

        previous_path = previous.payload.get("path") if previous.payload else None
        current_path = current.payload.get("path") if current.payload else None
        same_specific_path = self._same_specific_path(previous_path, current_path)
        shared_basename = (
            self._file_basename(previous_path) is not None
            and self._file_basename(previous_path) == self._file_basename(current_path)
        )
        previous_family = self.get_event_family(previous.type)
        current_family = self.get_event_family(current.type)
        same_event_family = previous_family is not None and previous_family == current_family
        same_application = (
            self._extract_app(previous) is not None
            and self._extract_app(previous) == self._extract_app(current)
        )

        gap_seconds = (self._parse_timestamp(current.timestamp) - self._parse_timestamp(previous.timestamp)).total_seconds()
        short_gap = 0 < gap_seconds <= self.max_episode_gap_seconds

        return [
            EvidenceSignal("same_artifact_id", same_artifact_id, SCORING_WEIGHTS["same_artifact_id"] if same_artifact_id else 0.0),
            EvidenceSignal("same_specific_path", same_specific_path, SCORING_WEIGHTS["same_specific_path"] if same_specific_path else 0.0),
            EvidenceSignal("shared_basename", shared_basename, SCORING_WEIGHTS["shared_basename"] if shared_basename else 0.0),
            EvidenceSignal("short_gap", short_gap, SCORING_WEIGHTS["short_gap"] if short_gap else 0.0),
            EvidenceSignal("same_event_family", same_event_family, SCORING_WEIGHTS["same_event_family"] if same_event_family else 0.0),
            EvidenceSignal("application_continuity", same_application, SCORING_WEIGHTS["application_continuity"] if same_application else 0.0),
        ]

    def _should_merge(self, previous: Event, current: Event, signals: List[EvidenceSignal], gap_seconds: float) -> EpisodeDecision:
        if gap_seconds > self.max_episode_gap_seconds:
            threshold = self.merge_threshold
            return EpisodeDecision(
                previous.id,
                current.id,
                gap_seconds,
                [signal for signal in signals if signal.present],
                score=sum(signal.contribution for signal in signals),
                threshold=threshold,
                decision="SPLIT",
                reason="gap_exceeded",
            )

        score = sum(signal.contribution for signal in signals)
        has_meaningful = any(signal.present and signal.name in MEANINGFUL_SIGNALS for signal in signals)
        if signals[0].present:
            return EpisodeDecision(
                previous.id,
                current.id,
                gap_seconds,
                [signal for signal in signals if signal.present],
                score=score,
                threshold=self.merge_threshold,
                decision="MERGED",
            )
        if score >= self.merge_threshold and has_meaningful:
            return EpisodeDecision(
                previous.id,
                current.id,
                gap_seconds,
                [signal for signal in signals if signal.present],
                score=score,
                threshold=self.merge_threshold,
                decision="MERGED",
            )
        return EpisodeDecision(
            previous.id,
            current.id,
            gap_seconds,
            [signal for signal in signals if signal.present],
            score=score,
            threshold=self.merge_threshold,
            decision="SPLIT",
            reason="insufficient_relationship",
        )

    def _compute_confidence(self, evidence: List[EpisodeDecision]) -> float:
        if not evidence:
            return 0.5
        score = sum(decision.score for decision in evidence) / len(evidence)
        return min(1.0, max(0.0, score / 1.3))

    def _build_title(self, event_ids: List[str], events_by_id: Dict[str, Event]) -> str:
        if not event_ids:
            return "Activity"
        paths: List[str] = []
        for event_id in event_ids:
            event = events_by_id[event_id]
            path = event.payload.get("path") if event.payload else None
            if path:
                paths.append(path)
        if len(paths) == 1:
            return f"Activity involving {Path(paths[0]).name}"
        basename_counts: Dict[str, int] = {}
        for path in paths:
            basename = Path(path).name
            basename_counts[basename] = basename_counts.get(basename, 0) + 1
        if basename_counts:
            most_common = max(basename_counts.items(), key=lambda item: (item[1], item[0]))[0]
            if len(basename_counts) == 1:
                return f"Activity involving {most_common}"
            if basename_counts[most_common] > 1:
                return f"Activity involving {most_common}"
        if paths:
            first_path = paths[0]
            return f"Activity involving {Path(first_path).name}"
        first_event = events_by_id[event_ids[0]]
        start = self._parse_timestamp(first_event.timestamp).strftime("%H:%M")
        last_event = events_by_id[event_ids[-1]]
        end = self._parse_timestamp(last_event.timestamp).strftime("%H:%M")
        return f"Activity — {start}–{end}"

    @staticmethod
    def _episode_id_from_components(start_ts: str, end_ts: str, event_ids: Sequence[str]) -> str:
        hasher = hashlib.sha256()
        joined = f"{start_ts}|{end_ts}|{','.join(event_ids)}"
        hasher.update(joined.encode("utf-8"))
        return hasher.hexdigest()

    def group_events(self, events: Sequence[Event]) -> List[Episode]:
        if not events:
            return []

        ordered = sorted(events, key=lambda event: (self._parse_timestamp(event.timestamp), event.id))
        unique: List[Event] = []
        seen: set[str] = set()
        for event in ordered:
            if event.id in seen:
                continue
            seen.add(event.id)
            unique.append(event)

        episodes: List[Episode] = []
        current_events: List[Event] = [unique[0]]
        current_evidence: List[EpisodeDecision] = []

        for previous, current in zip(unique, unique[1:]):
            gap_seconds = (self._parse_timestamp(current.timestamp) - self._parse_timestamp(previous.timestamp)).total_seconds()
            signals = self._compute_signals(previous, current)
            decision = self._should_merge(previous, current, signals, gap_seconds)
            if decision.decision == "MERGED":
                current_events.append(current)
                current_evidence.append(decision)
            else:
                episode = self._build_episode(current_events, current_evidence)
                episodes.append(episode)
                current_events = [current]
                current_evidence = []
                current_artifact_ids = [current.artifact_id] if current.artifact_id else []

        episode = self._build_episode(current_events, current_evidence)
        episodes.append(episode)
        return episodes

    def _build_episode(self, events: List[Event], evidence: List[EpisodeDecision]) -> Episode:
        event_ids = [event.id for event in events]
        artifact_ids: List[str] = []
        for event in events:
            if event.artifact_id and event.artifact_id not in artifact_ids:
                artifact_ids.append(event.artifact_id)
        start_ts = events[0].timestamp
        end_ts = events[-1].timestamp
        title = self._build_title(event_ids, {event.id: event for event in events})
        grouping_confidence = self._compute_confidence(evidence)
        episode_id = self._episode_id_from_components(start_ts, end_ts, event_ids)
        return Episode(
            id=episode_id,
            start_ts=start_ts,
            end_ts=end_ts,
            event_ids=event_ids,
            artifact_ids=artifact_ids,
            grouping_confidence=grouping_confidence,
            title=title,
            evidence=evidence,
        )

    def detect_and_persist(self, event_store: Any, episode_store: EpisodeStore, start_ts: Optional[str] = None, end_ts: Optional[str] = None) -> List[Episode]:
        events = event_store.query_events(start_ts=start_ts, end_ts=end_ts)
        episodes = self.group_events(events)
        if start_ts is not None and end_ts is not None:
            if episodes:
                episode_store.save_episodes(episodes, start_ts, end_ts)
            else:
                episode_store.delete_episodes_in_range(start_ts, end_ts)
        elif episodes:
            episode_store.save_episodes(episodes)
        return episodes
