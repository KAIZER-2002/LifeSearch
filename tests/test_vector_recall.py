import numpy as np
import pytest

from src.vector.chunker import TextChunk
from src.vector.store import VectorStore


def _chunk(aid, text):
    return TextChunk(chunk_id=f"{aid}_c0", artifact_id=aid, text=text, start_char=0, end_char=len(text), chunk_index=0)


def _build(n=600, seed=1234):
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n):
        v = rng.standard_normal(384).astype(np.float32)
        vectors.append((v / np.linalg.norm(v)).tolist())
    return vectors


def test_ann_recall_at_384_dim(tmp_path):
    vectors = _build(n=600)
    exact = VectorStore(str(tmp_path / "exact.db"), ann_enabled=False)
    ann = VectorStore(str(tmp_path / "ann.db"), ann_enabled=True)

    # Identical corpus in both stores.
    for i, vec in enumerate(vectors):
        exact.save_chunks(i + 1, [_chunk(i + 1, f"doc {i}")], [vec], "recall", 384)
        ann.save_chunks(i + 1, [_chunk(i + 1, f"doc {i}")], [vec], "recall", 384)

    rng = np.random.default_rng(99)
    query_idx = rng.choice(len(vectors), size=50, replace=False)
    recalls = []
    for qi in query_idx:
        q = vectors[qi]
        exact_res = exact.search_semantic_chunks(q, "recall", 384, top_k=5, min_similarity=-1.0)
        ann_res = ann.search_semantic_chunks(q, "recall", 384, top_k=5, min_similarity=-1.0)
        exact_set = {r.artifact_id for r in exact_res}
        ann_set = {r.artifact_id for r in ann_res}
        if exact_set:
            recalls.append(len(exact_set & ann_set) / len(exact_set))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.95, f"Recall@5 = {mean_recall:.3f} < 0.95"

    exact.close()
    ann.close()
