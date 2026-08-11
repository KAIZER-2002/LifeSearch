import json
import os
import sqlite3
from typing import Any, Dict, List, Optional


class EventStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                type TEXT,
                timestamp INTEGER,
                source TEXT,
                artifact_id TEXT,
                app TEXT,
                raw_payload TEXT,
                metadata TEXT,
                confidence REAL,
                tags TEXT,
                linked_entities TEXT
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_artifact ON events(artifact_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_app ON events(app)")
        self.conn.commit()

    def append_event(self, event: Dict[str, Any]) -> None:
        payload_json = json.dumps(event.get("raw_payload", {}), ensure_ascii=False)
        metadata_json = json.dumps(event.get("metadata", {}), ensure_ascii=False)
        tags_json = json.dumps(event.get("tags", []), ensure_ascii=False)
        entities_json = json.dumps(event.get("linked_entities", []), ensure_ascii=False)

        self.conn.execute(
            "INSERT OR REPLACE INTO events (id, type, timestamp, source, artifact_id, app, raw_payload, metadata, confidence, tags, linked_entities)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                event["id"],
                event.get("type"),
                event.get("timestamp"),
                event.get("source"),
                event.get("artifact_id"),
                event.get("app"),
                payload_json,
                metadata_json,
                event.get("confidence"),
                tags_json,
                entities_json,
            ],
        )
        self.conn.commit()

    def query_events(
        self,
        event_type: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        app: Optional[str] = None,
        artifact_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM events"
        clauses: List[str] = []
        params: List[Any] = []

        if event_type is not None:
            clauses.append("type = ?")
            params.append(event_type)
        if start_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(end_ts)
        if app is not None:
            clauses.append("app = ?")
            params.append(app)
        if artifact_id is not None:
            clauses.append("artifact_id = ?")
            params.append(artifact_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp ASC"

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "id": row["id"],
            "type": row["type"],
            "timestamp": row["timestamp"],
            "source": row["source"],
            "artifact_id": row["artifact_id"],
            "app": row["app"],
            "raw_payload": json.loads(row["raw_payload"] or "{}"),
            "metadata": json.loads(row["metadata"] or "{}"),
            "confidence": row["confidence"],
            "tags": json.loads(row["tags"] or "[]"),
            "linked_entities": json.loads(row["linked_entities"] or "[]"),
        }

    def close(self) -> None:
        self.conn.close()
