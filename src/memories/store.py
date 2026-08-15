from __future__ import annotations

import json
import logging
import sqlite3
from typing import Iterable, List, Optional

from src.artifacts.store import ArtifactStore
from src.memories.model import Memory


class MemoryStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.default_db_path()
        self.conn = sqlite3.connect(self.db_path)
        # Durability hardening (C8-5A): WAL + foreign-key enforcement.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as _durable_exc:
            logging.getLogger(__name__).warning(
                "SQLite durability PRAGMA (WAL/foreign_keys) failed; "
                "durability guarantees may not hold: %s",
                _durable_exc,
            )
        self._ensure_schema()

    @staticmethod
    def default_db_path() -> str:
        return ArtifactStore.default_db_path()

    def _ensure_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                start_ts TEXT NOT NULL,
                end_ts TEXT NOT NULL,
                episode_ids TEXT NOT NULL,
                event_ids TEXT NOT NULL,
                artifact_ids TEXT NOT NULL,
                title TEXT NOT NULL,
                topics TEXT NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_start_end ON memories(start_ts, end_ts)"
        )
        self.conn.commit()

    def save_memories(self, memories: Iterable[Memory], range_start_ts: Optional[str] = None, range_end_ts: Optional[str] = None) -> None:
        if range_start_ts is not None and range_end_ts is not None:
            self.delete_memories_in_range(range_start_ts, range_end_ts)

        for memory in memories:
            self.conn.execute(
                "INSERT OR REPLACE INTO memories (id, start_ts, end_ts, episode_ids, event_ids, artifact_ids, title, topics, confidence, evidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    memory.id,
                    memory.start_ts,
                    memory.end_ts,
                    json.dumps(memory.episode_ids, separators=(",", ":")),
                    json.dumps(memory.event_ids, separators=(",", ":")),
                    json.dumps(memory.artifact_ids, separators=(",", ":")),
                    memory.title,
                    json.dumps(memory.topics, separators=(",", ":")),
                    memory.confidence,
                    json.dumps(memory.evidence, separators=(",", ":")),
                ),
            )
        self.conn.commit()

    def delete_memories_in_range(self, start_ts: str, end_ts: str) -> None:
        self.conn.execute(
            "DELETE FROM memories WHERE NOT (end_ts < ? OR start_ts > ?)",
            (start_ts, end_ts),
        )
        self.conn.commit()

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        cursor = self.conn.execute(
            "SELECT id, start_ts, end_ts, episode_ids, event_ids, artifact_ids, title, topics, confidence, evidence FROM memories WHERE id = ?",
            (memory_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_memories(self) -> List[Memory]:
        cursor = self.conn.execute("SELECT id, start_ts, end_ts, episode_ids, event_ids, artifact_ids, title, topics, confidence, evidence FROM memories ORDER BY start_ts, end_ts")
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def get_memories_in_time_range(self, start_ts: str, end_ts: str) -> List[Memory]:
        cursor = self.conn.execute(
            "SELECT id, start_ts, end_ts, episode_ids, event_ids, artifact_ids, title, topics, confidence, evidence FROM memories WHERE NOT (end_ts < ? OR start_ts > ?) ORDER BY start_ts, end_ts",
            (start_ts, end_ts),
        )
        return [self._row_to_memory(row) for row in cursor.fetchall()]

    def get_memories_for_episode(self, episode_id: str) -> List[Memory]:
        return [memory for memory in self.get_memories() if episode_id in memory.episode_ids]

    def get_memories_for_artifact(self, artifact_id: int) -> List[Memory]:
        return [memory for memory in self.get_memories() if artifact_id in memory.artifact_ids]

    def prune_dangling_memories(
        self, present_artifact_ids: set, present_episode_ids: set
    ) -> int:
        """Delete memories with no surviving artifact and no surviving episode (C8-5B).

        Mirrors prune_dangling_episodes: a memory derived from multiple sources is
        kept if any supporting source still exists.
        """
        removed = 0
        for mem in self.get_memories():
            has_artifact = bool(set(mem.artifact_ids) & present_artifact_ids)
            has_episode = bool(set(mem.episode_ids) & present_episode_ids)
            if not has_artifact and not has_episode:
                self.conn.execute("DELETE FROM memories WHERE id = ?", (mem.id,))
                removed += 1
        if removed:
            self.conn.commit()
        return removed

    def _row_to_memory(self, row: tuple) -> Memory:
        _id, start_ts, end_ts, episode_ids_json, event_ids_json, artifact_ids_json, title, topics_json, confidence, evidence_json = row
        data = {
            "id": _id,
            "episode_ids": json.loads(episode_ids_json),
            "event_ids": json.loads(event_ids_json),
            "artifact_ids": json.loads(artifact_ids_json),
            "start_ts": start_ts,
            "end_ts": end_ts,
            "title": title,
            "topics": json.loads(topics_json),
            "confidence": confidence,
            "evidence": json.loads(evidence_json),
        }
        return Memory.from_dict(data)

    def close(self) -> None:
        self.conn.close()
