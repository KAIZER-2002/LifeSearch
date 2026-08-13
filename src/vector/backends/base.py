from __future__ import annotations

from typing import List, Optional, Tuple


class VectorIndexBackend:
    """Minimal ANN / index backend abstraction.

    Operates purely on *stable integer labels* and (label, similarity) pairs.
    It does NOT know about chunk text, artifact ids, or model ids -- those are
    the facade's responsibility. This keeps the abstraction small and lets
    SQLite remain the canonical store of truth while the ANN index is a
    rebuildable cache.
    """

    def add_items(self, labels: List[int], embeddings: List[List[float]]) -> None:
        raise NotImplementedError

    def search(self, query_embedding: List[float], top_k: int) -> List[Tuple[int, float]]:
        """Return (label, similarity) pairs sorted by similarity descending."""
        raise NotImplementedError

    def mark_deleted(self, labels: List[int]) -> None:
        raise NotImplementedError

    def contains(self, label: int) -> bool:
        """Whether a label is currently present in the index (excludes deleted)."""
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> bool:
        """Load a previously saved index. Return True on success."""
        raise NotImplementedError

    def rebuild_from(self, items: List[Tuple[int, List[float]]]) -> None:
        """Replace the index contents with the given (label, embedding) items."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass
