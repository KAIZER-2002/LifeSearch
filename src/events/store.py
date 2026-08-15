import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

from .model import Event


class EventStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.default_db_path()
        self._ensure_data_folder()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        # Durability hardening (C8-5A): WAL + foreign-key enforcement.
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as _durable_exc:
            logging.getLogger(__name__).warning(
                "SQLite durability PRAGMA (WAL/foreign_keys) failed; "
                "durability guarantees may not hold: %s",
                _durable_exc,
            )
        self._initialize_schema()

    @staticmethod
    def default_db_path() -> str:
        data_folder = os.path.join(os.path.expanduser("~"), ".lifesearch")
        os.makedirs(data_folder, exist_ok=True)
        return os.path.join(data_folder, "lifesearch.db")

    def _ensure_data_folder(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _initialize_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                artifact_id INTEGER,
                payload TEXT NOT NULL DEFAULT '{}',
                event_confidence REAL NOT NULL DEFAULT 1.0,
                recorded_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
            CREATE INDEX IF NOT EXISTS idx_events_source_kind ON events(source_kind);
            CREATE INDEX IF NOT EXISTS idx_events_artifact_id ON events(artifact_id);
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "EventStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def append_event(self, event: Event) -> None:
        payload_json = json.dumps(event.payload, ensure_ascii=False)
        try:
            self.conn.execute(
                "INSERT INTO events (id, type, timestamp, source, source_kind, artifact_id, payload, event_confidence, recorded_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    event.id,
                    event.type,
                    event.timestamp,
                    event.source,
                    event.source_kind,
                    event.artifact_id,
                    payload_json,
                    event.event_confidence,
                    event.recorded_at,
                ],
            )
            self.conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Event with id {event.id} already exists") from exc

    def get_event(self, event_id: str) -> Optional[Event]:
        cursor = self.conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return Event.from_row(row)

    def query_events(
        self,
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        source_kind: Optional[str] = None,
        artifact_id: Optional[int] = None,
        start_ts: Optional[str] = None,
        end_ts: Optional[str] = None,
        limit: Optional[int] = None,
        order_desc: bool = False,
    ) -> List[Event]:
        query = "SELECT * FROM events"
        clauses: List[str] = []
        params: List[Any] = []

        if event_type is not None:
            clauses.append("type = ?")
            params.append(event_type)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if source_kind is not None:
            clauses.append("source_kind = ?")
            params.append(source_kind)
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            params.append(artifact_id)
        if start_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(end_ts)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp " + ("DESC" if order_desc else "ASC")
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        cursor = self.conn.execute(query, params)
        return [Event.from_row(row) for row in cursor.fetchall()]

    def get_events_for_artifact(self, artifact_id: int) -> List[Event]:
        return self.query_events(artifact_id=artifact_id)

    def delete_events_for_artifact(self, artifact_id: int) -> None:
        """Remove all events bound to an artifact (C8-5B orphan cleanup)."""
        self.conn.execute("DELETE FROM events WHERE artifact_id = ?", (artifact_id,))
        self.conn.commit()

    def get_events_by_type(self, event_type: str) -> List[Event]:
        return self.query_events(event_type=event_type)

    def get_events_by_source(self, source: str) -> List[Event]:
        return self.query_events(source=source)

    def get_events_in_time_range(self, start_ts: str, end_ts: str) -> List[Event]:
        return self.query_events(start_ts=start_ts, end_ts=end_ts)

    def get_recent_events(self, limit: int = 20) -> List[Event]:
        return self.query_events(limit=limit, order_desc=True)
