import json
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional


class EpisodeEngine:
    def __init__(self, db_path: str, time_gap_ms: int = 30 * 60 * 1000):
        self.db_path = db_path
        self.time_gap_ms = time_gap_ms
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.events: List[Dict[str, Any]] = []
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                start_ts INTEGER,
                end_ts INTEGER,
                dominant_entities TEXT,
                inferred_title TEXT,
                confidence REAL,
                event_ids TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_start ON episodes(start_ts)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_end ON episodes(end_ts)")
        self.conn.commit()

    def add_event(self, event: Dict[str, Any]) -> None:
        if not isinstance(event, dict):
            raise ValueError("Event must be a dictionary")
        self.events.append(event)
        self.events.sort(key=lambda event_item: event_item.get("timestamp", 0))

    def detect_episodes(
        self,
        from_ts: Optional[int] = None,
        to_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        filtered_events = [
            event
            for event in self.events
            if (from_ts is None or event.get("timestamp", 0) >= from_ts)
            and (to_ts is None or event.get("timestamp", 0) <= to_ts)
        ]

        if not filtered_events:
            return []

        episodes: List[Dict[str, Any]] = []
        current_chunk: List[Dict[str, Any]] = [filtered_events[0]]

        for event in filtered_events[1:]:
            previous_event = current_chunk[-1]
            gap = event.get("timestamp", 0) - previous_event.get("timestamp", 0)
            if gap > self.time_gap_ms:
                episodes.append(self._build_episode(current_chunk))
                current_chunk = [event]
            else:
                current_chunk.append(event)

        episodes.append(self._build_episode(current_chunk))
        return episodes

    def _build_episode(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        entities = []
        for event in events:
            entities.extend(event.get("linked_entities", []) or [])
            if event.get("type") == "BrowserVisited" and event.get("metadata", {}).get("url"):
                entities.append(event["metadata"]["url"])

        dominant_entities = sorted(set(entities), key=lambda x: entities.count(x), reverse=True)
        inferred_title = self._infer_title(events, dominant_entities)
        confidence = self._infer_confidence(events)

        episode = {
            "id": str(uuid.uuid4()),
            "start_ts": min(event.get("timestamp", 0) for event in events),
            "end_ts": max(event.get("timestamp", 0) for event in events),
            "dominant_entities": dominant_entities,
            "inferred_title": inferred_title,
            "confidence": confidence,
            "events": events,
            "event_ids": [event.get("id") for event in events],
        }
        self._persist_episode(episode)
        return episode

    def _infer_title(self, events: List[Dict[str, Any]], dominant_entities: List[str]) -> str:
        if dominant_entities:
            first_entity = dominant_entities[0]
            if isinstance(first_entity, str) and first_entity.startswith("http"):
                return f"Browsing session around {first_entity}"
            return f"Activity focused on {first_entity}"

        event_types = {event.get("type") for event in events}
        if "FileDownloaded" in event_types:
            return "Download and edit session"
        if events:
            return f"Episode around {events[0].get('type')}"
        return "Activity episode"

    def _infer_confidence(self, events: List[Dict[str, Any]]) -> float:
        if not events:
            return 0.0
        base = 0.5 + min(0.5, len(events) * 0.1)
        return min(1.0, base)

    def _persist_episode(self, episode: Dict[str, Any]) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO episodes (id, start_ts, end_ts, dominant_entities, inferred_title, confidence, event_ids)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                episode["id"],
                episode["start_ts"],
                episode["end_ts"],
                json.dumps(episode.get("dominant_entities", []), ensure_ascii=False),
                episode.get("inferred_title"),
                episode.get("confidence"),
                json.dumps(episode.get("event_ids", []), ensure_ascii=False),
            ],
        )
        self.conn.commit()

    def get_episodes(self) -> List[Dict[str, Any]]:
        cursor = self.conn.execute("SELECT * FROM episodes ORDER BY start_ts ASC")
        rows = cursor.fetchall()
        return [self._row_to_episode(row) for row in rows]

    def _row_to_episode(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "start_ts": row["start_ts"],
            "end_ts": row["end_ts"],
            "dominant_entities": json.loads(row["dominant_entities"] or "[]"),
            "inferred_title": row["inferred_title"],
            "confidence": row["confidence"],
            "event_ids": json.loads(row["event_ids"] or "[]"),
        }

    def close(self) -> None:
        self.conn.close()
