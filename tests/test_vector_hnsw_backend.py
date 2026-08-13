import os

import numpy as np
import pytest

from src.vector.backends.hnsw import HNSWVectorIndex


def _norm(v):
    a = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return (a / n).tolist() if n > 0 else a.tolist()


def _new_index(path, dim=3):
    return HNSWVectorIndex(path, dim, "test-model", ef_search=64)


def test_hnsw_cosine_correctness(tmp_path):
    idx = _new_index(str(tmp_path / "i.hnsw"))
    idx.add_items([1, 2], [_norm([1.0, 0.0, 0.0]), _norm([0.0, 1.0, 0.0])])
    res = idx.search(_norm([0.9, 0.1, 0.0]), top_k=2)
    assert len(res) == 2
    labels = [l for l, _ in res]
    assert labels[0] == 1  # most similar
    assert res[0][1] > 0.9  # cosine ~0.994
    idx.close()


def test_hnsw_dimension_validation(tmp_path):
    idx = _new_index(str(tmp_path / "i.hnsw"))
    with pytest.raises(ValueError):
        idx.add_items([1], [[1.0, 0.0]])  # wrong dim
    idx.close()


def test_hnsw_incremental_add(tmp_path):
    idx = _new_index(str(tmp_path / "i.hnsw"))
    idx.add_items(list(range(5)), [_norm([float(i) + 1.0, 0.0, 0.0]) for i in range(5)])
    assert idx.count() == 5
    idx.add_items(list(range(5, 10)), [_norm([0.0, float(i) + 1.0, 0.0]) for i in range(5, 10)])
    assert idx.count() == 10
    idx.close()


def test_hnsw_mark_deleted(tmp_path):
    idx = _new_index(str(tmp_path / "i.hnsw"))
    idx.add_items([1, 2], [_norm([1.0, 0.0, 0.0]), _norm([0.0, 1.0, 0.0])])
    idx.mark_deleted([2])
    res = idx.search(_norm([0.9, 0.1, 0.0]), top_k=2)
    labels = [l for l, _ in res]
    assert 2 not in labels
    assert 1 in labels
    idx.close()


def test_hnsw_persistence_and_reload(tmp_path):
    p = str(tmp_path / "i.hnsw")
    idx = _new_index(p)
    idx.add_items([1, 2], [_norm([1.0, 0.0, 0.0]), _norm([0.0, 1.0, 0.0])])
    idx.save()
    idx.close()

    idx2 = _new_index(p)
    assert idx2.load() is True
    res = idx2.search(_norm([0.9, 0.1, 0.0]), top_k=2)
    assert [l for l, _ in res][0] == 1
    idx2.close()


def test_hnsw_corrupt_load_returns_false(tmp_path):
    p = str(tmp_path / "i.hnsw")
    with open(p, "w") as f:
        f.write("not a real index")
    with open(p + ".meta.json", "w") as f:
        f.write('{"dimension": 3, "model_id": "test-model"}')
    idx = _new_index(p)
    assert idx.load() is False
    idx.close()


def test_hnsw_dimension_mismatch_on_load(tmp_path):
    p = str(tmp_path / "i.hnsw")
    idx = _new_index(p, dim=3)
    idx.add_items([1], [_norm([1.0, 0.0, 0.0])])
    idx.save()
    idx.close()
    # A different dimension instance must reject the on-disk index.
    idx2 = HNSWVectorIndex(p, 384, "test-model")
    assert idx2.load() is False
    idx2.close()
