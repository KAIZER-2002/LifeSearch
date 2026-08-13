from __future__ import annotations

import hashlib
import os
import re

# Local-first application data root (mirrors ArtifactStore / ModelManager
# conventions that already use ~/.lifesearch).
APP_DATA_ROOT = os.path.join(os.path.expanduser("~"), ".lifesearch")


def vector_index_root() -> str:
    """Directory that holds all persisted ANN indexes."""
    path = os.path.join(APP_DATA_ROOT, "vector_index")
    os.makedirs(path, exist_ok=True)
    return path


def vector_index_dir(db_path: str) -> str:
    """Stable, cross-platform directory per database/profile.

    Derived from the absolute database path so it is deterministic across
    processes and does not leak filesystem characters into directory names.
    """
    digest = hashlib.sha256(os.path.abspath(db_path).encode("utf-8")).hexdigest()[:16]
    path = os.path.join(vector_index_root(), digest)
    os.makedirs(path, exist_ok=True)
    return path


def vector_index_path(db_path: str, model_id: str, dimension: int) -> str:
    """On-disk HNSW index file, keyed by model_id + dimension.

    Keying by (model_id, dimension) guarantees that vectors from different
    embedding models never share an index and can never be cross-queried.
    """
    safe_model = re.sub(r"[^A-Za-z0-9_.-]", "_", str(model_id))
    return os.path.join(vector_index_dir(db_path), f"{safe_model}_{dimension}.hnsw")
