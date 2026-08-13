from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np

from .base import VectorIndexBackend

logger = logging.getLogger(__name__)

try:
    import hnswlib

    _HNSW_INSTALLED = True
except ImportError:  # pragma: no cover - exercised only without hnswlib
    hnswlib = None
    _HNSW_INSTALLED = False


class HNSWVectorIndex(VectorIndexBackend):
    """hnswlib-backed ANN index.

    Design constraints (Vector Production Hardening):
      * cosine similarity, configurable dimension / M / ef_construction / ef_search
      * stable integer labels (provided by the SQLite backend)
      * incremental insertion, deletion via hnswlib mark_deleted
      * persistent index file + sidecar metadata
      * dimension / model_id validated; never silently accepts mismatches

    hnswlib's cosine space returns ``distance = 1 - cosine_similarity``,
    so ``similarity = 1 - distance``. Vectors are normalised on both insert
    and query for consistency.
    """

    def __init__(
        self,
        index_path: str,
        dimension: int,
        model_id: str,
        space: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
    ) -> None:
        if not _HNSW_INSTALLED:
            raise RuntimeError("hnswlib is not installed")
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if M <= 0 or ef_construction <= 0 or ef_search <= 0:
            raise ValueError("M, ef_construction and ef_search must be positive")

        self.index_path = index_path
        self.meta_path = index_path + ".meta.json"
        self.dimension = dimension
        self.model_id = model_id
        self.space = space
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._index = None
        self._labels = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize(self, v: List[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        if a.ndim != 1 or a.shape[0] == 0:
            raise ValueError("embedding must be a non-empty 1-D vector")
        if a.shape[0] != self.dimension:
            raise ValueError(f"vector dim {a.shape[0]} != expected {self.dimension}")
        n = float(np.linalg.norm(a))
        if n < 1e-9:
            return a
        return a / n

    def _init_empty(self, capacity: int) -> None:
        cap = max(int(capacity), 1)
        self._index = hnswlib.Index(space=self.space, dim=self.dimension)
        self._index.init_index(
            max_elements=cap, ef_construction=self.ef_construction, M=self.M
        )
        self._index.set_ef(self.ef_search)

    def _ensure_capacity(self, additional: int) -> None:
        if self._index is None:
            self._init_empty(max(additional, 1024))
            return
        used = self._index.get_current_count()
        max_elements = self._index.get_max_elements()
        if max_elements - used < additional:
            new_cap = max(
                int(max_elements * 2),
                used + additional + 64,
            )
            self._index.resize_index(new_cap)

    @staticmethod
    def _sim_from_distance(distance: float) -> float:
        return 1.0 - float(distance)

    # ------------------------------------------------------------------
    # VectorIndexBackend interface
    # ------------------------------------------------------------------
    def add_items(self, labels: List[int], embeddings: List[List[float]]) -> None:
        if not labels:
            return
        if len(labels) != len(embeddings):
            raise ValueError("labels and embeddings length mismatch")
        for e in embeddings:
            if len(e) != self.dimension:
                raise ValueError(f"vector dim {len(e)} != expected {self.dimension}")
        norms = [self._normalize(e) for e in embeddings]
        arr = np.stack(norms).astype(np.float32)
        lbls = np.asarray(labels, dtype=np.int64)
        self._ensure_capacity(len(labels))
        self._index.add_items(arr, lbls)
        self._labels.update(int(x) for x in labels)

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        if self._index is None or self._index.get_current_count() == 0:
            return []
        q = self._normalize(query_embedding)
        # hnswlib raises if k exceeds the number of *alive* (non-deleted)
        # elements. Retry with a smaller k to stay robust.
        total = self._index.get_current_count()
        k = max(1, min(int(top_k), total))
        labels = distances = None
        last_err: Optional[Exception] = None
        for attempt in (k, max(1, k // 2), 1):
            try:
                labels, distances = self._index.knn_query(
                    np.stack([q]).astype(np.float32), k=attempt
                )
                break
            except RuntimeError as exc:
                last_err = exc
                continue
        if labels is None:
            logger.warning(f"hnsw search failed: {last_err}")
            return []
        out: List[Tuple[int, float]] = []
        for lab, dist in zip(labels[0], distances[0]):
            out.append((int(lab), self._sim_from_distance(dist)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def mark_deleted(self, labels: List[int]) -> None:
        if not labels or self._index is None:
            return
        try:
            for lab in labels:
                self._index.mark_deleted(int(lab))
            self._labels.difference_update(int(x) for x in labels)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"hnsw mark_deleted failed: {exc}")

    def contains(self, label: int) -> bool:
        return int(label) in self._labels

    def save(self) -> None:
        if self._index is None:
            return
        try:
            parent = os.path.dirname(self.index_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._index.save_index(self.index_path)
            with open(self.meta_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"dimension": self.dimension, "model_id": self.model_id},
                    fh,
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"hnsw save failed: {exc}")

    def load(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            return False
        try:
            with open(self.meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("dimension") != self.dimension or meta.get("model_id") != self.model_id:
                return False
            self._index = hnswlib.Index(space=self.space, dim=self.dimension)
            self._index.load_index(self.index_path)
            self._index.set_ef(self.ef_search)
            self._labels = set()  # labels unknown after load; add_items tracks new ones
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"hnsw load failed: {exc}")
            return False

    def rebuild_from(self, items: List[Tuple[int, List[float]]]) -> None:
        # Drop any stale index files before rebuilding.
        for p in (self.index_path, self.meta_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        self._index = None
        if not items:
            self._init_empty(1024)
            self._labels = set()
            self.save()
            return
        labels = [it[0] for it in items]
        norms = [self._normalize(e) for e in [it[1] for it in items]]
        arr = np.stack(norms).astype(np.float32)
        lbls = np.asarray(labels, dtype=np.int64)
        self._init_empty(max(len(labels), 1024))
        self._index.add_items(arr, lbls)
        self._labels = set(int(x) for x in labels)
        self.save()

    def count(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.get_current_count())

    def close(self) -> None:
        self._index = None
