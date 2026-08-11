import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _format_timestamp(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.default_db_path()
        self._ensure_data_folder()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
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
            CREATE TABLE IF NOT EXISTS artifacts (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                indexed_at TEXT NOT NULL,
                content_hash TEXT,
                extracted_text TEXT,
                extract_error TEXT,
                missing INTEGER NOT NULL DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS artifact_fts USING fts5(
                path,
                file_name,
                mime_type,
                extracted_text,
                tokenize = 'porter'
            );
            """
        )
        self.conn.commit()

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def artifact_count(self, include_missing: bool = False) -> int:
        query = "SELECT COUNT(*) FROM artifacts"
        if not include_missing:
            query += " WHERE missing = 0"
        result = self.conn.execute(query).fetchone()
        return int(result[0])

    def get_artifact_by_path(self, path: str) -> Optional[sqlite3.Row]:
        normalized = os.path.abspath(path)
        result = self.conn.execute(
            "SELECT * FROM artifacts WHERE path = ?", (normalized,)
        ).fetchone()
        return result

    def get_artifact(self, artifact_id: int) -> Optional[sqlite3.Row]:
        result = self.conn.execute(
            "SELECT * FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()
        return result

    def artifact_needs_index(self, path: str, size: int, modified_at: str) -> bool:
        row = self.get_artifact_by_path(path)
        if row is None:
            return True
        if row["missing"]:
            return True
        if row["size"] != size or row["modified_at"] != modified_at:
            return True
        return False

    def upsert_artifact(
        self,
        path: str,
        file_name: str,
        mime_type: str,
        size: int,
        created_at: str,
        modified_at: str,
        extracted_text: str,
        content_hash: Optional[str],
        extract_error: Optional[str],
    ) -> int:
        normalized = os.path.abspath(path)
        indexed_at = _now_iso()
        cursor = self.conn.execute(
            "INSERT INTO artifacts (path, file_name, mime_type, size, created_at, modified_at, indexed_at, content_hash, extracted_text, extract_error, missing) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)"
            " ON CONFLICT(path) DO UPDATE SET file_name = excluded.file_name, mime_type = excluded.mime_type, size = excluded.size, created_at = excluded.created_at, modified_at = excluded.modified_at, indexed_at = excluded.indexed_at, content_hash = excluded.content_hash, extracted_text = excluded.extracted_text, extract_error = excluded.extract_error, missing = 0",
            (
                normalized,
                file_name,
                mime_type,
                size,
                created_at,
                modified_at,
                indexed_at,
                content_hash,
                extracted_text,
                extract_error,
            ),
        )
        self.conn.commit()
        row = self.get_artifact_by_path(normalized)
        artifact_id = int(row["id"])
        self._upsert_fts(
            artifact_id,
            normalized,
            file_name,
            mime_type,
            extracted_text,
        )
        return artifact_id

    def _upsert_fts(
        self,
        artifact_id: int,
        path: str,
        file_name: str,
        mime_type: str,
        extracted_text: str,
    ) -> None:
        self.conn.execute(
            "DELETE FROM artifact_fts WHERE rowid = ?",
            (artifact_id,),
        )
        self.conn.execute(
            "INSERT INTO artifact_fts(rowid, path, file_name, mime_type, extracted_text) VALUES (?, ?, ?, ?, ?)",
            (artifact_id, path, file_name, mime_type, extracted_text),
        )
        self.conn.commit()

    def search_artifacts(self, query: str, limit: int = 20, mime_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
        sql = (
            "SELECT a.id, a.path, a.file_name, a.mime_type, a.size, a.modified_at, "
            "bm25(artifact_fts) AS rank, snippet(artifact_fts, 3, '[', ']', '...', 10) AS snippet "
            "FROM artifact_fts "
            "JOIN artifacts a ON artifact_fts.rowid = a.id "
            "WHERE artifact_fts MATCH ? AND a.missing = 0 "
        )
        params: List[Any] = [query]
        if mime_type_filter:
            sql += " AND a.mime_type LIKE ?"
            params.append(f"%{mime_type_filter}%")
        sql += " ORDER BY rank, a.path LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def mark_missing_artifacts(self, current_paths: Iterable[str]) -> None:
        current_set = {os.path.abspath(path) for path in current_paths}
        existing = self.conn.execute("SELECT path FROM artifacts").fetchall()
        for row in existing:
            path = row["path"]
            if path not in current_set and not self.is_artifact_missing(path):
                self.conn.execute("UPDATE artifacts SET missing = 1 WHERE path = ?", (path,))
        for path in current_set:
            self.conn.execute("UPDATE artifacts SET missing = 0 WHERE path = ?", (path,))
        self.conn.commit()

    def is_artifact_missing(self, path: str) -> bool:
        row = self.get_artifact_by_path(path)
        return bool(row and row["missing"])

    def list_artifacts_in_folder(self, folder_path: str, include_missing: bool = False) -> List[sqlite3.Row]:
        normalized_folder = os.path.abspath(folder_path)
        normalized_folder = normalized_folder.rstrip(os.sep) + os.sep
        query = "SELECT * FROM artifacts WHERE path LIKE ?"
        params: List[Any] = [normalized_folder + "%"]
        if not include_missing:
            query += " AND missing = 0"
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()

    def status(self) -> Dict[str, Any]:
        total = self.artifact_count(include_missing=True)
        present = self.artifact_count(include_missing=False)
        missing = total - present
        return {
            "total_artifacts": total,
            "available_artifacts": present,
            "missing_artifacts": missing,
            "database_path": self.db_path,
        }
