"""
scripts/benchmark_semantic_search.py

Benchmark script measuring local ONNX embedding engine performance:
  - Model load time
  - CPU embedding throughput (chunks/sec)
  - Exact NumPy vector search latency
  - Hybrid search latency
  - RAM & disk footprint
"""

import argparse
import os
import sys
import tempfile
import time

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.search.engine import SearchEngine
from src.vector.chunker import TextChunk
from src.vector.embeddings import ONNXEmbeddingEngine
from src.vector.persistence import vector_index_path
from src.vector.store import VectorStore


def run_benchmark():
    print("=== Life Search Slice 7 Semantic Search Benchmark ===")
    
    t0 = time.perf_counter()
    engine = ONNXEmbeddingEngine()
    t1 = time.perf_counter()

    if not engine._available:
        print("ONNX Embedding Engine is unavailable (model not downloaded/installed).")
        print("Run explicit download helper to install all-MiniLM-L6-v2.")
        return

    model_load_ms = (t1 - t0) * 1000
    print(f"Model Load Time: {model_load_ms:.2f} ms")

    # 1. Measure Embedding Throughput
    sample_chunks = [
        f"Sample text chunk #{i} for semantic vector embedding throughput measurement."
        for i in range(100)
    ]
    t_emb0 = time.perf_counter()
    embs = engine.embed_batch(sample_chunks)
    t_emb1 = time.perf_counter()
    duration_sec = t_emb1 - t_emb0
    throughput = len(sample_chunks) / duration_sec
    per_chunk_ms = (duration_sec / len(sample_chunks)) * 1000

    print(f"Embedding Throughput: {throughput:.1f} chunks/sec ({per_chunk_ms:.2f} ms/chunk)")

    # 2. Build Benchmark Corpus
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "bench.db")
        folder = os.path.join(tmp, "files")
        os.makedirs(folder, exist_ok=True)

        for i in range(50):
            with open(os.path.join(folder, f"doc_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(f"Document {i} discussing vector search, database management, and python indexing algorithms.")

        store = ArtifactStore(db_path)
        vec_store = VectorStore(db_path)
        scanner = ArtifactScanner(store, Extractor(NullOCREngine()), vector_store=vec_store, embedding_engine=engine)
        
        t_idx0 = time.perf_counter()
        scanner.index_folder(folder)
        t_idx1 = time.perf_counter()
        
        chunk_count = vec_store.count_chunks(engine.model_id)
        idx_duration_sec = t_idx1 - t_idx0
        print(f"Indexed 50 files ({chunk_count} chunks) in {idx_duration_sec:.2f} s")

        # 3. Measure Vector & Hybrid Search Latency
        search_engine = SearchEngine(store, vector_store=vec_store, embedding_engine=engine)

        q_vec = engine.embed_text("database management indexing")
        t_vec0 = time.perf_counter()
        vec_store.search_semantic_chunks(q_vec, engine.model_id, engine.dimension, top_k=20)
        t_vec1 = time.perf_counter()
        vec_latency_ms = (t_vec1 - t_vec0) * 1000
        print(f"Vector Search Latency ({chunk_count} chunks): {vec_latency_ms:.2f} ms")

        t_hyb0 = time.perf_counter()
        results = search_engine.search("database management indexing", limit=10)
        t_hyb1 = time.perf_counter()
        hyb_latency_ms = (t_hyb1 - t_hyb0) * 1000
        print(f"End-to-End Hybrid Search Latency: {hyb_latency_ms:.2f} ms (Found {len(results)} results)")

        db_size_bytes = os.path.getsize(db_path)
        print(f"Database Disk Size: {db_size_bytes / 1024:.1f} KB")

        store.close()
        vec_store.close()

def _build_synthetic_corpus(n, dim=384, seed=42):
    import numpy as np

    rng = np.random.default_rng(seed)
    vecs = []
    for _ in range(n):
        v = rng.standard_normal(dim).astype(np.float32)
        vecs.append((v / np.linalg.norm(v)).tolist())
    return vecs


def run_scaling_benchmark(scale: str):
    """Synthetic, deterministic scaling benchmark comparing ANN vs exact.

    Uses synthetic 384-dim unit vectors (no real model) so it runs fast and
    repeatably. Reports build time, ANN vs exact query latency, and Recall@k.
    The 1M scale is memory-intensive and only runs when explicitly requested.
    """
    import numpy as np

    sizes = {"10k": 10_000, "100k": 100_000, "1m": 1_000_000}
    if scale not in sizes:
        print(f"Unknown scale '{scale}'. Use one of: 10k, 100k, 1m")
        return
    n = sizes[scale]
    if scale == "1m":
        print("WARNING: 1M scale is memory-intensive. Running only because explicitly requested.")

    dim = 384
    model_id = "bench-384"
    print(f"=== Vector Scaling Benchmark ({scale}: {n} vectors, {dim}-dim) ===")

    vecs = _build_synthetic_corpus(n)
    queries = vecs[: min(200, n)]

    with tempfile.TemporaryDirectory() as tmp:
        db_ann = os.path.join(tmp, "ann.db")
        db_exact = os.path.join(tmp, "exact.db")
        ann = VectorStore(db_ann, ann_enabled=True)
        exact = VectorStore(db_exact, ann_enabled=False)

        # Insertion (realistic incremental path) + in-memory ANN build.
        t0 = time.perf_counter()
        for gi in range(n):
            chunk = TextChunk(
                chunk_id=f"{gi}_c0", artifact_id=gi, text=f"doc {gi}",
                start_char=0, end_char=5, chunk_index=0,
            )
            ann.save_chunks(gi, [chunk], [vecs[gi]], model_id, dim)
            exact.save_chunks(gi, [chunk], [vecs[gi]], model_id, dim)
        ann.flush()
        build_s = time.perf_counter() - t0
        print(f"Insert + build ({n} vectors): {build_s:.2f} s ({n / build_s:.0f} vec/s)")
        ann_path = vector_index_path(db_ann, model_id, dim)
        print(f"ANN index disk size: {os.path.getsize(ann_path) / 1024:.1f} KB" if os.path.exists(ann_path) else "ANN index not persisted")

        # ANN query latency
        t0 = time.perf_counter()
        ann_hits = []
        for q in queries:
            ann_hits.append(ann.search_semantic_chunks(q, model_id, dim, top_k=10, min_similarity=-1.0))
        ann_lat_ms = (time.perf_counter() - t0) / len(queries) * 1000

        # Exact query latency
        t0 = time.perf_counter()
        exact_hits = []
        for q in queries:
            exact_hits.append(exact.search_semantic_chunks(q, model_id, dim, top_k=10, min_similarity=-1.0))
        exact_lat_ms = (time.perf_counter() - t0) / len(queries) * 1000

        # Recall@10
        recalls = []
        for a, e in zip(ann_hits, exact_hits):
            a_set = {r.artifact_id for r in a}
            e_set = {r.artifact_id for r in e}
            if e_set:
                recalls.append(len(a_set & e_set) / len(e_set))
        mean_recall = sum(recalls) / len(recalls) if recalls else 0.0

        print(f"ANN  query latency: {ann_lat_ms:.2f} ms/query")
        print(f"Exact query latency: {exact_lat_ms:.2f} ms/query")
        print(f"Recall@10 (ANN vs exact): {mean_recall:.3f}")

        ann.close()
        exact.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Life Search semantic search benchmarks")
    parser.add_argument(
        "--scale",
        choices=["small", "10k", "100k", "1m"],
        default="small",
        help="small = real-model benchmark (Slice 7); 10k/100k/1m = synthetic scaling (ANN vs exact).",
    )
    args = parser.parse_args()
    if args.scale == "small":
        run_benchmark()
    else:
        run_scaling_benchmark(args.scale)
