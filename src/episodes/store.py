from __future__ import annotations

import json
import sqlite3
from typing import Iterable, List, Optional

from src.episodes.model import Episode
from src.artifacts.store import ArtifactStore

class EpisodeStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.default_db_path()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._ensure_schema()

    @staticmethod
    def default_db_path() -> str:
        return ArtifactStore.default_db_path()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                event_ids TEXT NOT NULL,
                artifact_ids TEXT NOT NULL,
                grouping_confidence REAL NOT NULL,
                title TEXT NOT NULL,
                evidence TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_episodes_start_end ON episodes(start_ts, end_ts)"
        )
        self.conn.commit()

    def save_episodes(self, episodes: Iterable[Episode], range_start_ts: Optional[str] = None, range_end_ts: Optional[str] = None) -> None:
        if range_start_ts is not None and range_end_ts is not None:
            self.delete_episodes_in_range(range_start_ts, range_end_ts)

        for episode in episodes:
            self.conn.execute(
                "INSERT OR REPLACE INTO episodes (id, start_ts, end_ts, event_ids, artifact_ids, grouping_confidence, title, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.id,
                    episode.start_ts,
                    episode.end_ts,
                    json.dumps(episode.event_ids, separators=(",", ":")),
                    json.dumps(episode.artifact_ids, separators=(",", ":")),
                    episode.grouping_confidence,
                    episode.title,
                    json.dumps(episode.to_dict()["evidence"], separators=(",", ":")),
                ),
            )
        self.conn.commit()

    def delete_episodes_in_range(self, start_ts: str, end_ts: str) -> None:
        self.conn.execute(
            "DELETE FROM episodes WHERE NOT (end_ts < ? OR start_ts > ?)",
            (start_ts, end_ts),
        )
        self.conn.commit()

    def get_episodes(self) -> List[Episode]:
        cursor = self.conn.execute("SELECT id, start_ts, end_ts, event_ids, artifact_ids, grouping_confidence, title, evidence FROM episodes ORDER BY start_ts, end_ts")
        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_in_time_range(self, start_ts: str, end_ts: str) -> List[Episode]:
        cursor = self.conn.execute(
            "SELECT id, start_ts, end_ts, event_ids, artifact_ids, grouping_confidence, title, evidence FROM episodes WHERE NOT (end_ts < ? OR start_ts > ?) ORDER BY start_ts, end_ts",
            (start_ts, end_ts),
        )
        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def get_episodes_for_artifact(self, artifact_id: int) -> List[Episode]:
        episodes = self.get_episodes()
        return [episode for episode in episodes if artifact_id in episode.artifact_ids]

    def _row_to_episode(self, row: tuple) -> Episode:
        _id, start_ts, end_ts, event_ids_json, artifact_ids_json, grouping_confidence, title, evidence_json = row
        data = {
            "id": _id,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "event_ids": json.loads(event_ids_json),
            "artifact_ids": json.loads(artifact_ids_json),
            "grouping_confidence": grouping_confidence,
            "title": title,
            "evidence": json.loads(evidence_json),
        }
        return Episode.from_dict(data)

    def close(self) -> None:
        self.conn.close()
