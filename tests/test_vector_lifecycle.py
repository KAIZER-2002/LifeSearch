import hashlib
import os

import numpy as np

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.vector.store import VectorStore


class FakeEmbeddingEngine:
    """Deterministic, model-id-tagged fake engine (no real model needed)."""

    def __init__(self, model_id="fake-384"):
        self.model_id = model_id
        self.dimension = 384

    def embed_text(self, text):
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384).astype(np.float32)
        n = np.linalg.norm(v)
        return (v / n).tolist() if n > 0 else v.tolist()

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]


def _scanner(db, engine):
    store = ArtifactStore(db)
    vs = VectorStore(db)
    sc = ArtifactScanner(store, Extractor(NullOCREngine()), vector_store=vs, embedding_engine=engine)
    return store, vs, sc


def test_scanner_index_and_delete_lifecycle(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    f = folder / "doc1.txt"
    f.write_text("MongoDB connection refused localhost 27017")

    db = str(tmp_path / "ls.db")
    store, vs, sc = _scanner(db, FakeEmbeddingEngine())

    sc.index_folder(str(folder))
    assert vs.count_chunks() >= 1

    engine = FakeEmbeddingEngine()
    q = engine.embed_text("MongoDB connection refused localhost 27017")
    res = vs.search_semantic_chunks(q, "fake-384", 384, top_k=5, min_similarity=0.3)
    assert len(res) >= 1

    # Remove the file and re-scan -> orphan vectors must be cleaned up.
    os.remove(str(f))
    sc.index_folder(str(folder))

    assert vs.count_chunks() == 0
    res_after = vs.search_semantic_chunks(q, "fake-384", 384, top_k=5, min_similarity=0.3)
    assert res_after == []

    store.close()
    vs.close()


def test_scanner_model_change_isolation(tmp_path):
    folder = tmp_path / "files"
    folder.mkdir()
    f = folder / "doc1.txt"
    f.write_text("vector search indexing notes")

    db = str(tmp_path / "ls.db")
    store, vs, sc = _scanner(db, FakeEmbeddingEngine(model_id="model-A-384"))
    sc.index_folder(str(folder))
    assert vs.stored_model_ids() == ["model-A-384"]
    assert vs.count_chunks() >= 1

    # Switch to a different model without reindex: unchanged file is skipped,
    # so new-model vectors are absent (old-model vectors stay isolated).
    store2, vs2, sc2 = _scanner(db, FakeEmbeddingEngine(model_id="model-B-384"))
    sc2.index_folder(str(folder))
    assert "model-B-384" not in vs2.stored_model_ids()

    qb = FakeEmbeddingEngine(model_id="model-B-384").embed_text("vector search indexing notes")
    res_b = vs2.search_semantic_chunks(qb, "model-B-384", 384, top_k=5, min_similarity=0.3)
    # Nothing indexed under the new model yet.
    assert res_b == []

    # Explicit reindex converts everything to the new model.
    sc2.index_folder(str(folder), reindex_on_model_change=True)
    assert vs2.stored_model_ids() == ["model-B-384"]
    res_b2 = vs2.search_semantic_chunks(qb, "model-B-384", 384, top_k=5, min_similarity=0.3)
    assert len(res_b2) >= 1

    store.close()
    store2.close()
    vs.close()
    vs2.close()
