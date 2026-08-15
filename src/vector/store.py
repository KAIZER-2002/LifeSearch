from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from .backends.base import VectorIndexBackend
from .backends.sqlite_exact import SQLiteExactBackend
from .persistence import vector_index_path
from .types import ChunkMatch

logger = logging.getLogger(__name__)

try:
    from .backends.hnsw import HNSWVectorIndex

    _HNSW_AVAILABLE = True
except ImportError:  # hnswlib not installed -> exact-only mode
    HNSWVectorIndex = None
    _HNSW_AVAILABLE = False


class VectorStore:
    """Facade over the vector subsystem.

    Public contract (unchanged from the pre-hardening implementation):
      * save_chunks(artifact_id, chunks, embeddings, model_id, dimension)
      * delete_artifact_chunks(artifact_id)
      * search_semantic_chunks(query_embedding, model_id, dimension, top_k, min_similarity)
      * count_chunks(model_id=None)
      * close()

    Behaviour:
      * Canonical vector data is always written to SQLite (source of truth).
      * HNSW is synchronised as a derived, rebuildable cache.
      * Queries use HNSW when available, otherwise exact SQLite.
      * ANY ANN failure degrades transparently to exact SQLite search.
      * SQLite canonical data is never lost because of an ANN failure.

    SearchEngine / ArtifactScanner depend only on this contract and remain
    backend-agnostic.
    """

    def __init__(
        self,
        db_path: str,
        ann_enabled: Optional[bool] = None,
        ef_search: int = 64,
    ) -> None:
        self.db_path = db_path
        self.sqlite = SQLiteExactBackend(db_path, check_same_thread=False)
        self.ann_enabled = bool(ann_enabled) if ann_enabled is not None else _HNSW_AVAILABLE
        self.ef_search = ef_search
        self._ann_cache: Dict[Tuple[str, int], Optional[VectorIndexBackend]] = {}
        self._dirty = False
        self._pending_adds = 0
        self._flush_every = 2000

    # ------------------------------------------------------------------
    # Canonical API (stable)
    # ------------------------------------------------------------------
    def save_chunks(
        self,
        artifact_id: int,
        chunks: List[Any],
        embeddings: List[List[float]],
        model_id: str,
        dimension: int,
    ) -> None:
        # Capture the current ANN labels for this artifact BEFORE the SQLite
        # replace deletes them, so we can mark the old vectors deleted.
        old_labels = []
        if self.ann_enabled:
            try:
                old_labels = self.sqlite.get_labels_for_artifact(artifact_id)
            except Exception:
                old_labels = []
        # Canonical write is authoritative; propagate errors to callers.
        self.sqlite.save_chunks(artifact_id, chunks, embeddings, model_id, dimension)
        self._sync_save(artifact_id, model_id, dimension, old_labels)

    def delete_artifact_chunks(self, artifact_id: int) -> None:
        labels = self.sqlite.get_labels_for_artifact(artifact_id)
        self.sqlite.delete_artifact_chunks(artifact_id)
        if self.ann_enabled:
            try:
                for ann in self._ann_cache.values():
                    if ann is not None:
                        ann.mark_deleted(labels)
                # The in-memory index was mutated by deletion; mark it dirty so
                # the deletion is persisted on the next flush/close. Without this,
                # mark_deleted was lost on close() because flush() early-returns
                # when not dirty (HNSW deletion persistence fix).
                if labels:
                    self._dirty = True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"ANN delete sync failed for artifact {artifact_id}: {exc}")

    def search_semantic_chunks(
        self,
        query_embedding: List[float],
        model_id: str,
        dimension: int,
        top_k: int = 20,
        min_similarity: float = 0.50,
    ) -> List[ChunkMatch]:
        if self.ann_enabled:
            try:
                ann = self._ensure_ann(model_id, dimension)
                if ann is not None and ann.count() > 0:
                    lbl_sim = ann.search(query_embedding, top_k)
                    if lbl_sim:
                        rows = self.sqlite.get_chunks_by_labels([l for l, _ in lbl_sim])
                        results: List[ChunkMatch] = []
                        for lab, sim in lbl_sim:
                            row = rows.get(lab)
                            if row is None:
                                # label was deleted/missing in SQLite -> skip
                                continue
                            if sim < min_similarity:
                                continue
                            results.append(
                                ChunkMatch(
                                    chunk_id=row["chunk_id"],
                                    artifact_id=row["artifact_id"],
                                    text=row["text"],
                                    similarity=sim,
                                    source_type=row["source_type"],
                                    chunk_index=row["chunk_index"],
                                )
                            )
                        return results[:top_k]
            except Exception as exc:
                logger.warning(f"ANN search failed, falling back to exact: {exc}")
        # Exact SQLite fallback (also covers missing/corrupt/unavailable ANN).
        return self.sqlite.search_semantic_chunks(
            query_embedding, model_id, dimension, top_k, min_similarity
        )

    def count_chunks(self, model_id: Optional[str] = None) -> int:
        return self.sqlite.count_chunks(model_id)

    # ------------------------------------------------------------------
    # Model / lifecycle helpers
    # ------------------------------------------------------------------
    def stored_model_ids(self) -> List[str]:
        return self.sqlite.stored_model_ids()

    def delete_model_chunks(self, model_id: str) -> None:
        """Explicit cleanup of all vectors for a given embedding model."""
        self.sqlite.delete_model_chunks(model_id)
        for key in list(self._ann_cache.keys()):
            if key[0] == model_id:
                self._ann_cache.pop(key, None)

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
        try:
            self.sqlite.close()
        except Exception:
            pass
        self._ann_cache.clear()

    # ------------------------------------------------------------------
    # ANN management (internal)
    # ------------------------------------------------------------------
    def _sync_save(
        self, artifact_id: int, model_id: str, dimension: int, old_labels: List[int]
    ) -> None:
        if not self.ann_enabled:
            return
        try:
            ann = self._ensure_ann(model_id, dimension)
            if ann is None:
                return
            if old_labels:
                ann.mark_deleted(old_labels)
            new = self.sqlite.get_chunks_for_artifact(artifact_id)
            if new:
                # Skip labels already present (e.g. after a rebuild-from-SQLite
                # that already ingested them) to avoid duplicate vectors.
                # Use the backend abstraction instead of peeking at HNSW internals.
                to_add = [(l, e) for l, e in new if not ann.contains(l)]
                if to_add:
                    ann.add_items([l for l, _ in to_add], [e for _, e in to_add])
                    self._dirty = True
                    self._pending_adds += len(to_add)
                    if self._pending_adds >= self._flush_every:
                        self.flush()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"ANN sync failed for artifact {artifact_id}: {exc}")

    def flush(self) -> None:
        """Persist any dirty ANN indexes to disk.

        The in-memory ANN index is authoritative within a process; saving is
        deferred (and periodic) so bulk indexing does not rewrite the whole
        index file on every artifact. Call before process exit / on close.
        """
        if not self._dirty:
            return
        for ann in self._ann_cache.values():
            if ann is not None:
                try:
                    ann.save()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(f"ANN flush failed: {exc}")
        self._dirty = False
        self._pending_adds = 0

    def _ensure_ann(self, model_id: str, dimension: int) -> Optional[VectorIndexBackend]:
        key = (model_id, dimension)
        if key in self._ann_cache:
            return self._ann_cache[key]
        if not self.ann_enabled or HNSWVectorIndex is None:
            self._ann_cache[key] = None
            return None

        path = vector_index_path(self.db_path, model_id, dimension)
        ann = HNSWVectorIndex(path, dimension, model_id, ef_search=self.ef_search)
        try:
            if os.path.exists(path):
                if not ann.load():
                    # Corrupt/incompatible on-disk index -> rebuild from SQLite.
                    self._rebuild_ann(ann, model_id, dimension)
            else:
                if self.sqlite.count_chunks(model_id) > 0:
                    # Missing index but canonical data exists: rebuild it.
                    self._rebuild_ann(ann, model_id, dimension)
                else:
                    # No data yet: start empty; incremental adds populate it.
                    ann._init_empty(1024)
            self._ann_cache[key] = ann
            return ann
        except Exception as exc:
            logger.warning(f"ANN init failed for {key}, using exact search: {exc}")
            self._ann_cache[key] = None
            return None

    def _rebuild_ann(self, ann: VectorIndexBackend, model_id: str, dimension: int) -> None:
        items = self.sqlite.get_all_chunks_for_model(model_id, dimension)
        ann.rebuild_from(items)


# Re-export for backwards-compatible imports (tests, __init__).
__all__ = ["VectorStore", "ChunkMatch"]
