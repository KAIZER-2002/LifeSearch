"""
End-to-end search integration tests (C8-4).

These were previously SKIPPED because they imported a non-existent
module-level ``search`` function from ``src.search.engine``. C8-4 converts
them into real, deterministic acceptance coverage that exercises the actual
runtime path:

    real files -> run_ingest -> Events/Episodes/Memories -> SearchEngine.search

No manual manufacturing of EventStore / EpisodeStore / MemoryStore state:
everything is produced by the real orchestrator. No model download; lexical
+ episode retrieval is sufficient for these checks.
"""

import os
import tempfile

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.store import EpisodeStore
from src.events.store import EventStore
from src.memories.store import MemoryStore
from src.search.engine import SearchEngine
from src.vector.embeddings import NullEmbeddingEngine
from src.vector.store import VectorStore

from src.ingest.orchestrator import run_ingest


def _make_scanner(db_path: str) -> ArtifactScanner:
    artifact_store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    return ArtifactScanner(
        artifact_store,
        Extractor(NullOCREngine()),
        vector_store=vector_store,
        embedding_engine=NullEmbeddingEngine(),
    )


def _seed(folder: str, name: str, content: str) -> str:
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


def _build_engine(db_path: str) -> SearchEngine:
    return SearchEngine(
        ArtifactStore(db_path),
        episode_store=EpisodeStore(db_path),
        memory_store=MemoryStore(db_path),
        event_store=EventStore(db_path),
        vector_store=VectorStore(db_path),
        embedding_engine=NullEmbeddingEngine(),
    )


def test_recall_cases_top3():
    """A distinctive query must surface the expected artifact in the top-3.

    This is the deterministic, model-free version of the curated recall
    cases: we verify lexical recall over a real runtime-indexed dataset.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        folder = os.path.join(tmp, "docs")
        os.makedirs(folder)
        _seed(folder, "qdrant_notes.txt", "qdrant vector database notes for retrieval")

        scanner = _make_scanner(db)
        run_ingest(scanner, [folder], db)

        engine = _build_engine(db)
        results = engine.search("qdrant", limit=3)

        assert results, "expected at least one result for a distinctive term"
        top3 = results[:3]
        assert any(
            "qdrant" in (r.file_name + r.snippet + r.path).lower() for r in top3
        ), "expected qdrant artifact in top-3"


def test_evidence_trace_in_results():
    """Search results must include runtime evidence tracing to events/artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ls.db")
        folder = os.path.join(tmp, "docs")
        os.makedirs(folder)
        _seed(folder, "qdrant_notes.txt", "qdrant vector database notes for retrieval")

        scanner = _make_scanner(db)
        run_ingest(scanner, [folder], db)

        engine = _build_engine(db)
        results = engine.search("qdrant", limit=1)
        assert results, "expected at least one result"
        top = results[0]

        assert "evidence" in top and isinstance(top.evidence, list)
        assert top.evidence, "evidence must be non-empty after runtime ingest"

        # Evidence must reference real source artifacts/events. At minimum one
        # item must point at this artifact (event/episode/memory with artifact_id).
        assert any(
            isinstance(ev, dict)
            and ev.get("type") in ("event", "episode", "memory")
            and ev.get("artifact_id") == top.id
            for ev in top.evidence
        ), "evidence must reference the source artifact via event/episode/memory"
