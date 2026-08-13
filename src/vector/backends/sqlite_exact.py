from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..chunker import TextChunk
from ..types import ChunkMatch

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteExactBackend:
    """Canonical vector store.

    Responsibilities (per the Vector Production Hardening decision):
      * Source of truth for chunk text + raw embedding BLOBs.
      * Exact cosine search (permanent fallback for ANN failures).
      * Recovery source for rebuilding the HNSW cache.

    A stable, monotonic integer ``idx_label`` is allocated per chunk so the
    ANN backend can use it as its vector id without relying on Python's
    non-stable ``hash()``.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._initialize_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _initialize_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vector_chunks (
                chunk_id TEXT PRIMARY KEY,
                artifact_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                source_type TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                text TEXT NOT NULL,
                model_id TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                idx_label INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_vector_chunks_artifact ON vector_chunks(artifact_id);
            CREATE INDEX IF NOT EXISTS idx_vector_chunks_model ON vector_chunks(model_id, dimension);
            CREATE INDEX IF NOT EXISTS idx_vector_chunks_label ON vector_chunks(idx_label);
            CREATE TABLE IF NOT EXISTS vector_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
            """
        )
        # Migrate databases created before idx_label existed.
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(vector_chunks)")]
        if "idx_label" not in cols:
            self.conn.execute("ALTER TABLE vector_chunks ADD COLUMN idx_label INTEGER")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Writes (canonical)
    # ------------------------------------------------------------------
    def save_chunks(
        self,
        artifact_id: int,
        chunks: List[TextChunk],
        embeddings: List[List[float]],
        model_id: str,
        dimension: int,
    ) -> None:
        if not chunks or not embeddings or len(chunks) != len(embeddings):
            return

        for emb in embeddings:
            if len(emb) != dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {dimension}, got {len(emb)}"
                )

        # Reserve labels and perform the replace inside a single transaction
        # (avoids a commit per label allocation on top of the replace commit).
        labels = []
        now_str = _now_iso()

        try:
            self.conn.execute("BEGIN TRANSACTION")
            row = self.conn.execute(
                "SELECT value FROM vector_meta WHERE key='hnsw_hwm'"
            ).fetchone()
            start = int(row["value"]) if row else 0
            for _ in chunks:
                start += 1
                labels.append(start)
            self.conn.execute(
                "INSERT INTO vector_meta(key, value) VALUES('hnsw_hwm', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (start,),
            )
            self.conn.execute(
                "DELETE FROM vector_chunks WHERE artifact_id = ?", (artifact_id,)
            )
            for chunk, emb, label in zip(chunks, embeddings, labels):
                blob = np.array(emb, dtype=np.float32).tobytes()
                self.conn.execute(
                    """
                    INSERT INTO vector_chunks (
                        chunk_id, artifact_id, chunk_index, source_type,
                        start_char, end_char, text, model_id, dimension,
                        embedding, created_at, idx_label
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.artifact_id,
                        chunk.chunk_index,
                        chunk.source_type,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.text,
                        model_id,
                        dimension,
                        blob,
                        now_str,
                        label,
                    ),
                )
            self.conn.commit()
        except Exception as exc:
            self.conn.rollback()
            raise RuntimeError(
                f"Atomic vector replacement failed for artifact {artifact_id}: {exc}"
            ) from exc

    def delete_artifact_chunks(self, artifact_id: int) -> None:
        self.conn.execute(
            "DELETE FROM vector_chunks WHERE artifact_id = ?", (artifact_id,)
        )
        self.conn.commit()

    def delete_model_chunks(self, model_id: str) -> None:
        self.conn.execute("DELETE FROM vector_chunks WHERE model_id = ?", (model_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Label / chunk lookups (used by the facade + ANN rebuild)
    # ------------------------------------------------------------------
    def get_labels_for_artifact(self, artifact_id: int) -> List[int]:
        rows = self.conn.execute(
            "SELECT idx_label FROM vector_chunks WHERE artifact_id = ?", (artifact_id,)
        ).fetchall()
        return [int(r["idx_label"]) for r in rows if r["idx_label"] is not None]

    def get_chunks_for_artifact(self, artifact_id: int) -> List[Tuple[int, List[float]]]:
        rows = self.conn.execute(
            "SELECT idx_label, embedding FROM vector_chunks WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchall()
        return self._decode_rows(rows)

    def get_all_chunks_for_model(
        self, model_id: str, dimension: int
    ) -> List[Tuple[int, List[float]]]:
        rows = self.conn.execute(
            "SELECT idx_label, embedding FROM vector_chunks WHERE model_id = ? AND dimension = ?",
            (model_id, dimension),
        ).fetchall()
        return self._decode_rows(rows)

    def get_chunks_by_labels(self, labels: List[int]) -> Dict[int, Dict[str, Any]]:
        if not labels:
            return {}
        placeholders = ",".join("?" * len(labels))
        rows = self.conn.execute(
            f"SELECT chunk_id, artifact_id, text, source_type, chunk_index, idx_label "
            f"FROM vector_chunks WHERE idx_label IN ({placeholders})",
            [int(l) for l in labels],
        ).fetchall()
        return {
            int(r["idx_label"]): {
                "chunk_id": str(r["chunk_id"]),
                "artifact_id": int(r["artifact_id"]),
                "text": str(r["text"]),
                "source_type": str(r["source_type"]),
                "chunk_index": int(r["chunk_index"]),
            }
            for r in rows
        }

    @staticmethod
    def _decode_rows(rows) -> List[Tuple[int, List[float]]]:
        out: List[Tuple[int, List[float]]] = []
        for r in rows:
            if r["idx_label"] is None:
                continue
            arr = np.frombuffer(r["embedding"], dtype=np.float32)
            out.append((int(r["idx_label"]), arr.tolist()))
        return out

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def stored_model_ids(self) -> List[str]:
        return [r[0] for r in self.conn.execute("SELECT DISTINCT model_id FROM vector_chunks").fetchall()]

    def count_chunks(self, model_id: Optional[str] = None) -> int:
        if model_id:
            return self.conn.execute(
                "SELECT COUNT(*) FROM vector_chunks WHERE model_id = ?", (model_id,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM vector_chunks").fetchone()[0]

    def search_semantic_chunks(
        self,
        query_embedding: List[float],
        model_id: str,
        dimension: int,
        top_k: int = 20,
        min_similarity: float = 0.50,
    ) -> List[ChunkMatch]:
        """Exact cosine search over stored vectors (permanent fallback)."""
        if not query_embedding or len(query_embedding) != dimension:
            return []

        rows = self.conn.execute(
            "SELECT chunk_id, artifact_id, chunk_index, source_type, text, embedding "
            "FROM vector_chunks WHERE model_id = ? AND dimension = ?",
            (model_id, dimension),
        ).fetchall()
        if not rows:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-9:
            return []
        q = q / q_norm

        matrix_rows = []
        metadata_rows = []
        for r in rows:
            arr = np.frombuffer(r["embedding"], dtype=np.float32)
            if len(arr) == dimension:
                matrix_rows.append(arr)
                metadata_rows.append(r)
        if not matrix_rows:
            return []

        matrix = np.vstack(matrix_rows)
        norms = np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        norm_matrix = matrix / norms
        similarities = np.dot(norm_matrix, q)

        matches: List[ChunkMatch] = []
        for idx, sim in enumerate(similarities):
            sim_val = float(sim)
            if sim_val >= min_similarity:
                r = metadata_rows[idx]
                matches.append(
                    ChunkMatch(
                        chunk_id=str(r["chunk_id"]),
                        artifact_id=int(r["artifact_id"]),
                        text=str(r["text"]),
                        similarity=sim_val,
                        source_type=str(r["source_type"]),
                        chunk_index=int(r["chunk_index"]),
                    )
                )
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches[:top_k]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
