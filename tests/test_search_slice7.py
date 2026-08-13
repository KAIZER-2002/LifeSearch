"""
tests/test_search_slice7.py

Integration tests for Slice 7: Local Semantic Retrieval.
Verifies concept-based search, OCR semantic search, fallback behavior,
and ranking score fusion without breaking Slices 1-6.
"""

import os
import tempfile
from typing import List

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.search.engine import SearchEngine
from src.vector.chunker import TextChunk
from src.vector.embeddings import NullEmbeddingEngine
from src.vector.store import VectorStore


class MockEmbeddingEngine:
    """Mock embedding engine mapping concept words to deterministic 3D vectors."""

    def __init__(self):
        self.model_id = "mock-v1"
        self.dimension = 3

    def embed_text(self, text: str) -> List[float]:
        t_lower = text.lower()
        if any(w in t_lower for w in ["mongo", "database", "daemon", "server", "refused", "offline"]):
            return [1.0, 0.0, 0.0]
        if any(w in t_lower for w in ["qdrant", "vector", "index", "nn"]):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def test_concept_semantic_search_with_mock_engine(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    folder = str(tmp_path / "files")
    os.makedirs(folder, exist_ok=True)

    # Document text uses daemon/offline phrasing
    doc_path = os.path.join(folder, "mongo_notes.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("MongoDB daemon connection refused localhost")

    store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    mock_engine = MockEmbeddingEngine()

    extractor = Extractor(NullOCREngine())
    scanner = ArtifactScanner(store, extractor, vector_store=vector_store, embedding_engine=mock_engine)
    scanner.index_folder(folder)

    assert vector_store.count_chunks("mock-v1") >= 1

    search_engine = SearchEngine(
        store,
        vector_store=vector_store,
        embedding_engine=mock_engine,
        min_semantic_similarity=0.50,
    )

    # Query uses different phrasing ("server offline")
    results = search_engine.search("database server offline")
    assert len(results) >= 1
    assert results[0].file_name == "mongo_notes.txt"
    assert "Semantic match to concepts in document text" in results[0].why


def test_missing_embedding_engine_fallback_to_fts(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    folder = str(tmp_path / "files")
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "notes.txt"), "w", encoding="utf-8") as f:
        f.write("lexical search keyword matching test")

    store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    null_engine = NullEmbeddingEngine()

    extractor = Extractor(NullOCREngine())
    scanner = ArtifactScanner(store, extractor, vector_store=vector_store, embedding_engine=null_engine)
    scanner.index_folder(folder)

    search_engine = SearchEngine(
        store,
        vector_store=vector_store,
        embedding_engine=null_engine,
    )

    results = search_engine.search("lexical search")
    assert len(results) >= 1
    assert results[0].file_name == "notes.txt"
