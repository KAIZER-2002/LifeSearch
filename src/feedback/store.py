"""
FeedbackStore — isolated, local-first capture of user ranking signals.

Stores feedback in the EXISTING Life Search database, reusing the
``re_rank_feedback`` and ``accesses`` tables defined in ``sqlite_schema.sql``
(CREATE TABLE IF NOT EXISTS is used so the store is self-contained and
testable against a fresh database).

Design constraints (C4):
- Opens its OWN SQLite connection to the existing database file.
- WAL + busy_timeout for safe concurrent access from HTTP request threads.
- Short, parameterized transactions only (no SQL string interpolation).
- Validates action values; records query, document_id, action, timestamp.
- Does NOT introduce a new database engine or dependency.
- No re-ranking logic lives here (consumed by a later milestone).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Actions supported by the C4 API contract (api_spec.yaml FeedbackRequest).
VALID_ACTIONS = ("click", "ignore", "pin")

# Access-like events also logged to the `accesses` table for future recency/
# frequency signals. `ignore` is a negative ranking signal only.
_ACCESS_ACTIONS = ("click", "pin")


class FeedbackStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema (reuses the existing feedback/access tables verbatim)
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS re_rank_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT,
                    document_id TEXT,
                    action TEXT,
                    timestamp INTEGER
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS accesses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    access_type TEXT,
                    timestamp INTEGER
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def record(self, query: str, document_id: str, action: str) -> None:
        action = self._validate_action(action)
        ts = int(time.time() * 1000)
        with self._lock:
            try:
                with self._conn:
                    cur = self._conn.cursor()
                    cur.execute(
                        "INSERT INTO re_rank_feedback "
                        "(query_text, document_id, action, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        (query, document_id, action, ts),
                    )
                    if action in _ACCESS_ACTIONS:
                        cur.execute(
                            "INSERT INTO accesses (document_id, access_type, timestamp) "
                            "VALUES (?, ?, ?)",
                            (document_id, action, ts),
                        )
            except sqlite3.Error:
                # Surface to caller; the HTTP layer converts to a safe 500.
                raise

    @staticmethod
    def _validate_action(action: str) -> str:
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid feedback action: {action!r}")
        return action

    # ------------------------------------------------------------------
    # Read helpers (used by tests and a future ranking consumer)
    # ------------------------------------------------------------------
    def get_feedback(
        self,
        document_id: Optional[str] = None,
        query: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            params.append(document_id)
        if query is not None:
            clauses.append("query_text = ?")
            params.append(query)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        sql = (
            "SELECT id, query_text, document_id, action, timestamp "
            "FROM re_rank_feedback" + where + " ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "query": r["query_text"],
                "document_id": r["document_id"],
                "action": r["action"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    def get_counts(self, document_id: Optional[str] = None) -> Dict[str, int]:
        clauses: List[str] = []
        params: List[Any] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            params.append(document_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = "SELECT action, COUNT(*) FROM re_rank_feedback" + where + " GROUP BY action"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {r["action"]: r["COUNT(*)"] for r in rows}

    def get_accesses(
        self, document_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if document_id is not None:
            clauses.append("document_id = ?")
            params.append(document_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(int(limit))
        sql = (
            "SELECT id, document_id, access_type, timestamp "
            "FROM accesses" + where + " ORDER BY id DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": r["id"],
                "document_id": r["document_id"],
                "access_type": r["access_type"],
                "timestamp": r["timestamp"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
