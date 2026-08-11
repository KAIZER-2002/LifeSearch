from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Union

from src.episodes.model import Episode
from src.events.model import Event
from src.memories.model import Memory

GENERIC_TOKENS = {
    "pdf",
    "py",
    "md",
    "docs",
    "docx",
    "download",
    "downloads",
    "file",
    "files",
    "desktop",
    "documents",
    "pictures",
    "videos",
    "music",
    "jpg",
    "jpeg",
    "png",
    "txt",
    "screenshot",
    "screenshotdiscovered",
    "application",
    "active",
    "project",
    "projects",
    "users",
    "user",
    "workspace",
    "workspaces",
    "c",
    "d",
    "e",
    "appdata",
    "local",
    "temp",
    "tmp",
    "home",
    "usr",
    "var",
}

BROAD_DIRECTORIES = {
    "Downloads",
    "Documents",
    "Desktop",
    "Pictures",
    "Videos",
    "Music",
    "Users",
    "User",
    "Project",
    "Projects",
    "Workspace",
    "Workspaces",
    "Home",
    "AppData",
    "Local",
    "Temp",
    "Tmp",
    "C",
    "D",
    "E",
}


TOPIC_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _normalize_token(token: str) -> str:
    return token.lower().strip()


def _is_valid_topic_token(token: str) -> bool:
    token = _normalize_token(token)
    if len(token) < 3:
        return False
    if token.isdigit():
        return False
    if token in GENERIC_TOKENS:
        return False
    return True


def _split_into_tokens(text: str) -> List[str]:
    return [token for token in TOPIC_TOKEN_PATTERN.findall(text) if _is_valid_topic_token(token)]


def _path_segments(path: str) -> List[str]:
    if not path:
        return []
    normalized = Path(path).as_posix()
    return [segment for segment in normalized.split("/") if segment]


def _filename_tokens(path: str) -> List[str]:
    name = Path(path).name
    parts = [p for p in name.split(".") if p]
    if len(parts) > 1:
        parts = parts[:-1]
    tokens: List[str] = []
    for part in reversed(parts):
        for token in _split_into_tokens(part):
            if token not in tokens:
                tokens.append(token)
    return tokens


def _path_topic_tokens(path: str) -> List[str]:
    segments = _path_segments(path)
    user_dirs: set[str] = set()
    for i, seg in enumerate(segments[:-1]):
        if seg.lower() in {"users", "user", "home"}:
            if i + 1 < len(segments) - 1:
                user_dirs.add(segments[i + 1].lower())

    topics: List[str] = []
    for segment in reversed(segments[:-1]):
        normalized = segment.lower().strip()
        if normalized in {token.lower() for token in BROAD_DIRECTORIES} or normalized in user_dirs or normalized in GENERIC_TOKENS:
            continue
        for token in _split_into_tokens(normalized):
            if token not in topics:
                topics.append(token)
    return topics



def _title_for_topics(topics: List[str]) -> str:
    if not topics:
        raise ValueError("topics must not be empty")
    if len(topics) == 1:
        return f"Activity involving {topics[0]}"
    if len(topics) == 2:
        return f"Activity involving {topics[0]} and {topics[1]}"
    return f"Activity involving {topics[0]} and {topics[1]}"


def _time_range_title(start_ts: str, end_ts: str) -> str:
    from datetime import datetime, timezone

    def _parse_timestamp(ts: str):
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    start = _parse_timestamp(start_ts).strftime("%H:%M")
    end = _parse_timestamp(end_ts).strftime("%H:%M")
    return f"Activity — {start}–{end}"


class EventProvider(Protocol):
    def get_event(self, event_id: str) -> Optional[Event]:
        ...


class MemoryBuilder:
    def __init__(self, event_provider: Union[EventProvider, Dict[str, Event]]):
        self.event_provider = event_provider

    @staticmethod
    def _parse_timestamp(timestamp: str):
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(timestamp)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _get_event(self, event_id: str) -> Optional[Event]:
        if hasattr(self.event_provider, "get_event"):
            return self.event_provider.get_event(event_id)
        return self.event_provider.get(event_id)

    def _collect_event_evidence(self, episode: Episode) -> List[Event]:
        events: List[Event] = []
        for event_id in episode.event_ids:
            event = self._get_event(event_id)
            if event is not None:
                events.append(event)
        return events

    def _extract_topic_sources(self, events: List[Event]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        sources: List[Dict[str, Any]] = []

        for event in events:
            path = event.payload.get("path") if event.payload else None
            if path:
                for token in _filename_tokens(path):
                    if token not in seen:
                        seen.add(token)
                        sources.append({"topic": token, "source": Path(path).name, "source_type": "filename"})
                for token in _path_topic_tokens(path):
                    if token not in seen:
                        seen.add(token)
                        sources.append({"topic": token, "source": path, "source_type": "path_segment"})

            tags = event.payload.get("tags") if event.payload else None
            if isinstance(tags, Iterable) and not isinstance(tags, str):
                for tag in tags:
                    if isinstance(tag, str):
                        token = _normalize_token(tag)
                        if _is_valid_topic_token(token) and token not in seen:
                            seen.add(token)
                            sources.append({"topic": token, "source": tag, "source_type": "event_tag"})

        return sources

    def _build_title(self, topic_sources: List[Dict[str, Any]], start_ts: str, end_ts: str) -> str:
        if not topic_sources:
            return _time_range_title(start_ts, end_ts)

        filename_topics = [source for source in topic_sources if source["source_type"] == "filename"]
        if filename_topics:
            filenames: List[str] = []
            seen: set[str] = set()
            for source in filename_topics:
                name = source["source"]
                if name not in seen:
                    seen.add(name)
                    filenames.append(name)
            if len(filenames) == 1:
                return f"Activity involving {filenames[0]}"
            return f"Activity involving {filenames[0]} and {filenames[1]}"

        topic_names = [source["topic"] for source in topic_sources[:2]]
        if topic_names:
            if len(topic_names) == 1:
                return f"Activity involving {topic_names[0]}"
            return f"Activity involving {topic_names[0]} and {topic_names[1]}"

        return _time_range_title(start_ts, end_ts)

    def _compute_confidence(self, episode: Episode, topics: List[str]) -> float:
        base = episode.grouping_confidence
        bonus = 0.0
        if episode.artifact_ids:
            bonus += 0.1
        if topics:
            bonus += 0.05
        confidence = min(1.0, max(0.0, base + bonus))
        return confidence

    def build(self, episode: Episode) -> Optional[Memory]:
        if not episode.event_ids:
            return None

        events = self._collect_event_evidence(episode)
        if not events:
            return None

        topic_sources = self._extract_topic_sources(events)
        topics = [source["topic"] for source in topic_sources]
        title = self._build_title(topic_sources, episode.start_ts, episode.end_ts)
        confidence = self._compute_confidence(episode, topics)

        evidence = [
            {
                "episode_id": episode.id,
                "event_ids": list(episode.event_ids),
                "artifact_ids": list(episode.artifact_ids),
                "title_basis": "artifact_names" if any(source["source_type"] == "filename" for source in topic_sources) else "topics" if topics else "time_range",
                "topic_sources": topic_sources,
                "grouping_confidence": episode.grouping_confidence,
            }
        ]

        return Memory(
            id=f"memory_{episode.id}",
            episode_ids=[episode.id],
            event_ids=list(episode.event_ids),
            artifact_ids=list(episode.artifact_ids),
            start_ts=episode.start_ts,
            end_ts=episode.end_ts,
            title=title,
            topics=topics,
            confidence=confidence,
            evidence=evidence,
        )
