from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChunkMatch:
    """A single semantic match returned by the vector store.

    Kept in a neutral module so that SQLite / HNSW backends and the
    VectorStore facade can all import it without creating import cycles.
    """

    chunk_id: str
    artifact_id: int
    text: str
    similarity: float
    source_type: str
    chunk_index: int
