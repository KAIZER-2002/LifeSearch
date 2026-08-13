import os
import tempfile

import pytest

from src.vector.backends.sqlite_exact import SQLiteExactBackend
from src.vector.chunker import TextChunk


def _tmp():
    d = tempfile.TemporaryDirectory()
    return _Ctx(d)


class _Ctx:
    def __init__(self, d):
        self._d = d

    def __enter__(self):
        return os.path.join(self._d.name, "ls.db")

    def __exit__(self, *a):
        self._d.cleanup()


def _chunk(artifact_id, text, chunk_id=None):
    cid = chunk_id or f"{artifact_id}_c0"
    return TextChunk(chunk_id=cid, artifact_id=artifact_id, text=text, start_char=0, end_char=len(text), chunk_index=0)


def test_sqlite_save_and_exact_search():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        b.save_chunks(1, [_chunk(1, "MongoDB connection refused")], [[1.0, 0.0, 0.0]], "m-v1", 3)
        b.save_chunks(2, [_chunk(2, "Qdrant vector guide")], [[0.0, 1.0, 0.0]], "m-v1", 3)
        res = b.search_semantic_chunks([0.9, 0.1, 0.0], "m-v1", 3, top_k=5, min_similarity=0.5)
        assert len(res) == 1
        assert res[0].artifact_id == 1
        assert res[0].similarity > 0.8
        b.close()


def test_sqlite_dimension_validation():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        with pytest.raises(ValueError, match="dimension mismatch"):
            b.save_chunks(1, [_chunk(1, "x")], [[1.0, 0.0]], "m-v1", 3)
        b.close()


def test_sqlite_model_id_isolation():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        b.save_chunks(1, [_chunk(1, "x")], [[1.0, 0.0, 0.0]], "model-A", 3)
        res = b.search_semantic_chunks([1.0, 0.0, 0.0], "model-B", 3)
        assert res == []
        b.close()


def test_sqlite_atomic_replacement():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        b.save_chunks(1, [_chunk(1, "initial")], [[1.0, 0.0, 0.0]], "m-v1", 3)
        assert b.count_chunks() == 1
        b.save_chunks(1, [_chunk(1, "updated")], [[0.0, 1.0, 0.0]], "m-v1", 3)
        assert b.count_chunks() == 1
        b.close()


def test_sqlite_lookup_helpers():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        b.save_chunks(1, [_chunk(1, "a")], [[1.0, 0.0, 0.0]], "m-v1", 3)
        labels = b.get_labels_for_artifact(1)
        assert len(labels) == 1
        by_label = b.get_chunks_by_labels(labels)
        assert labels[0] in by_label
        assert by_label[labels[0]]["artifact_id"] == 1
        b.close()


def test_sqlite_stored_model_ids():
    with _tmp() as db:
        b = SQLiteExactBackend(db)
        b.save_chunks(1, [_chunk(1, "a")], [[1.0, 0.0, 0.0]], "model-A", 3)
        b.save_chunks(2, [_chunk(2, "b")], [[0.0, 1.0, 0.0]], "model-B", 3)
        assert set(b.stored_model_ids()) == {"model-A", "model-B"}
        b.close()
