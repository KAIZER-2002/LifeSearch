import os
import tempfile
import pytest

from src.vector.chunker import TextChunk
from src.vector.store import VectorStore


def test_vector_store_roundtrip_and_cosine_search():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        store = VectorStore(db_path)

        chunks = [
            TextChunk(chunk_id="1_c0", artifact_id=1, text="MongoDB server daemon refused connection", start_char=0, end_char=40, chunk_index=0),
            TextChunk(chunk_id="2_c0", artifact_id=2, text="Qdrant vector index guide", start_char=0, end_char=25, chunk_index=0),
        ]
        # 3-dim mock embeddings
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        store.save_chunks(artifact_id=1, chunks=[chunks[0]], embeddings=[embeddings[0]], model_id="mock-v1", dimension=3)
        store.save_chunks(artifact_id=2, chunks=[chunks[1]], embeddings=[embeddings[1]], model_id="mock-v1", dimension=3)

        assert store.count_chunks("mock-v1") == 2

        # Query vector close to [1.0, 0.0, 0.0]
        query_vec = [0.9, 0.1, 0.0]
        matches = store.search_semantic_chunks(query_vec, model_id="mock-v1", dimension=3, top_k=5, min_similarity=0.50)

        assert len(matches) == 1
        assert matches[0].artifact_id == 1
        assert matches[0].chunk_id == "1_c0"
        assert matches[0].similarity > 0.8

        store.close()


def test_vector_store_dimension_validation():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        store = VectorStore(db_path)

        chunks = [TextChunk(chunk_id="1_c0", artifact_id=1, text="Sample text", start_char=0, end_char=10, chunk_index=0)]
        bad_embeddings = [[1.0, 0.0]]  # len 2 vs dim 3

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.save_chunks(artifact_id=1, chunks=chunks, embeddings=bad_embeddings, model_id="mock-v1", dimension=3)

        store.close()


def test_vector_store_model_id_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        store = VectorStore(db_path)

        chunks = [TextChunk(chunk_id="1_c0", artifact_id=1, text="Sample text", start_char=0, end_char=10, chunk_index=0)]
        embeddings = [[1.0, 0.0, 0.0]]

        store.save_chunks(artifact_id=1, chunks=chunks, embeddings=embeddings, model_id="model-A", dimension=3)

        # Searching with model-B returns 0 results
        matches = store.search_semantic_chunks([1.0, 0.0, 0.0], model_id="model-B", dimension=3)
        assert len(matches) == 0

        store.close()


def test_vector_store_atomic_replacement():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        store = VectorStore(db_path)

        c1 = TextChunk(chunk_id="1_c0", artifact_id=1, text="Initial text", start_char=0, end_char=12, chunk_index=0)
        store.save_chunks(artifact_id=1, chunks=[c1], embeddings=[[1.0, 0.0, 0.0]], model_id="mock-v1", dimension=3)
        assert store.count_chunks() == 1

        c2 = TextChunk(chunk_id="1_c0_v2", artifact_id=1, text="Updated text", start_char=0, end_char=12, chunk_index=0)
        store.save_chunks(artifact_id=1, chunks=[c2], embeddings=[[0.0, 1.0, 0.0]], model_id="mock-v1", dimension=3)
        assert store.count_chunks() == 1  # Replaced, not duplicated!

        store.close()
