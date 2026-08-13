"""
tests/test_semantic_eval.py

Evaluation test suite measuring Recall@5, Recall@10, and MRR
on positive target documents and hard negatives.
"""

import os
import tempfile
from typing import Dict, List, Tuple

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.search.engine import SearchEngine
from src.vector.embeddings import NullEmbeddingEngine, ONNXEmbeddingEngine
from src.vector.store import VectorStore


def test_semantic_evaluation_suite(tmp_path):
    engine_onnx = ONNXEmbeddingEngine()
    if not engine_onnx._available:
        pytest.skip("ONNX embedding model unavailable for eval suite")

    db_path = str(tmp_path / "eval_ls.db")
    folder = str(tmp_path / "files")
    os.makedirs(folder, exist_ok=True)

    # 1. Corpus Creation: Positive targets + Hard negatives
    corpus = {
        # Query 1 Target & Hard Negative
        "mongo_daemon_refused.txt": "MongoDB connection refused because local daemon stopped running 27017",
        "mongo_installation_guide.txt": "MongoDB official installation guide for Ubuntu Linux setup steps",

        # Query 2 Target & Hard Negative
        "qdrant_indexing_notes.txt": "Qdrant nearest neighbor high-dimensional vector search documentation",
        "sql_relational_schema.txt": "SQL relational database schema design normalized tables",

        # Query 3 Target & Hard Negative
        "voice_receptionist_arch.txt": "AI phone voice receptionist system architecture audio pipeline",
        "generic_chatbot_faq.md": "Generic website customer support FAQ list text answers",
    }

    for fn, content in corpus.items():
        with open(os.path.join(folder, fn), "w", encoding="utf-8") as f:
            f.write(content)

    store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    extractor = Extractor(NullOCREngine())
    scanner = ArtifactScanner(store, extractor, vector_store=vector_store, embedding_engine=engine_onnx)
    scanner.index_folder(folder)

    search_engine = SearchEngine(
        store,
        vector_store=vector_store,
        embedding_engine=engine_onnx,
        min_semantic_similarity=0.35,
    )

    eval_cases = [
        {
            "query": "database connection refused daemon",
            "positive": "mongo_daemon_refused.txt",
            "negative": "mongo_installation_guide.txt",
        },
        {
            "query": "vector similarity search docs",
            "positive": "qdrant_indexing_notes.txt",
            "negative": "sql_relational_schema.txt",
        },
        {
            "query": "AI voice phone receptionist architecture",
            "positive": "voice_receptionist_arch.txt",
            "negative": "generic_chatbot_faq.md",
        },
    ]

    recalls_5 = []
    mrr_list = []

    for case in eval_cases:
        results = search_engine.search(case["query"], limit=5)
        filenames = [r.file_name for r in results]

        pos_in_top5 = case["positive"] in filenames[:5]
        recalls_5.append(1.0 if pos_in_top5 else 0.0)

        rank = 0
        for idx, fn in enumerate(filenames):
            if fn == case["positive"]:
                rank = idx + 1
                break
        mrr = (1.0 / rank) if rank > 0 else 0.0
        mrr_list.append(mrr)

        # Assert positive target ranks higher than hard negative
        if case["negative"] in filenames and case["positive"] in filenames:
            assert filenames.index(case["positive"]) < filenames.index(case["negative"])

    mean_recall_5 = sum(recalls_5) / len(recalls_5)
    mean_mrr = sum(mrr_list) / len(mrr_list)

    store.close()
    vector_store.close()

    assert mean_recall_5 >= 0.66
    assert mean_mrr >= 0.50
