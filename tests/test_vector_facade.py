import os

import pytest

from src.vector.chunker import TextChunk
from src.vector.store import VectorStore
from src.vector.persistence import vector_index_path


def _chunk(aid, text, cid=None):
    cid = cid or f"{aid}_c0"
    return TextChunk(chunk_id=cid, artifact_id=aid, text=text, start_char=0, end_char=len(text), chunk_index=0)


def _make_store(tmp_path, ann_enabled=True):
    db = str(tmp_path / "ls.db")
    return VectorStore(db, ann_enabled=ann_enabled)


def _seed(store):
    store.save_chunks(1, [_chunk(1, "MongoDB connection refused")], [[1.0, 0.0, 0.0]], "m-v1", 3)
    store.save_chunks(2, [_chunk(2, "Qdrant vector guide")], [[0.0, 1.0, 0.0]], "m-v1", 3)


def _search_exact(store):
    return store.search_semantic_chunks([0.9, 0.1, 0.0], "m-v1", 3, top_k=5, min_similarity=0.5)


def test_facade_contract_save_search_delete(tmp_path):
    store = _make_store(tmp_path)
    _seed(store)
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.delete_artifact_chunks(1)
    assert store.count_chunks() == 1
    assert _search_exact(store) == []
    store.close()


def test_facade_exact_only_fallback(tmp_path):
    store = _make_store(tmp_path, ann_enabled=False)
    _seed(store)
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.close()


def test_facade_ann_equals_exact(tmp_path):
    exact = _make_store(tmp_path, ann_enabled=False)
    ann = _make_store(tmp_path, ann_enabled=True)
    _seed(exact)
    _seed(ann)
    r_exact = {r.artifact_id for r in _search_exact(exact)}
    r_ann = {r.artifact_id for r in _search_exact(ann)}
    assert r_exact == r_ann == {1}
    exact.close()
    ann.close()


def test_facade_ann_failure_falls_back_to_exact(tmp_path, monkeypatch):
    store = _make_store(tmp_path, ann_enabled=True)
    _seed(store)

    def _boom(*a, **k):
        raise RuntimeError("simulated ANN failure")

    monkeypatch.setattr(store, "_ensure_ann", _boom)
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.close()


def test_facade_missing_ann_file_rebuilds(tmp_path):
    store = _make_store(tmp_path, ann_enabled=True)
    _seed(store)
    # Remove the persisted ANN index; search must rebuild from SQLite.
    p = vector_index_path(store.db_path, "m-v1", 3)
    if os.path.exists(p):
        os.remove(p)
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.close()


def test_facade_corrupt_ann_file_falls_back(tmp_path):
    store = _make_store(tmp_path, ann_enabled=True)
    _seed(store)
    p = vector_index_path(store.db_path, "m-v1", 3)
    with open(p, "w") as f:
        f.write("corrupted-bytes")
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.close()


def test_facade_model_isolation(tmp_path):
    store = _make_store(tmp_path, ann_enabled=True)
    store.save_chunks(1, [_chunk(1, "alpha")], [[1.0, 0.0, 0.0]], "model-A", 3)
    store.save_chunks(2, [_chunk(2, "beta")], [[0.0, 1.0, 0.0]], "model-B", 3)
    ra = store.search_semantic_chunks([1.0, 0.0, 0.0], "model-A", 3, top_k=5, min_similarity=0.5)
    rb = store.search_semantic_chunks([0.0, 1.0, 0.0], "model-B", 3, top_k=5, min_similarity=0.5)
    assert [r.artifact_id for r in ra] == [1]
    assert [r.artifact_id for r in rb] == [2]
    store.close()


def test_facade_dimension_mismatch_isolation(tmp_path):
    store = _make_store(tmp_path, ann_enabled=True)
    store.save_chunks(1, [_chunk(1, "x")], [[1.0, 0.0, 0.0]], "m", 3)
    # Querying with wrong dimension must not crash and must return nothing.
    res = store.search_semantic_chunks([1.0, 0.0], "m", 2, top_k=5)
    assert res == []
    store.close()


def test_facade_rebuild_from_sqlite(tmp_path):
    store = _make_store(tmp_path, ann_enabled=True)
    _seed(store)
    # Delete ANN files entirely and force a fresh ensure via search.
    import shutil

    from src.vector.persistence import vector_index_dir

    shutil.rmtree(vector_index_dir(store.db_path), ignore_errors=True)
    res = _search_exact(store)
    assert len(res) == 1 and res[0].artifact_id == 1
    store.close()
