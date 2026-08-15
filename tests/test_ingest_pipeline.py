"""
C8-1 runtime integration tests for the Events -> Episodes -> Memories
ingestion pipeline.

These tests exercise the REAL indexing path (scanner.index_folder +
run_ingest_folder), i.e. they must NOT manually call
EventStore.append_event(), EpisodeEngine.detect_and_persist(), or
MemoryBuilder.build(). Those operations must happen through the
orchestrator that the CLI and HTTP server both use.

This directly closes the false-confidence gap identified in the C8 audit:
previously the episode/memory enrichment was only ever tested with
hand-wired stores, so the runtime wiring was never verified.
"""

import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.client import HTTPConnection

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.events.store import EventStore
from src.episodes.store import EpisodeStore
from src.memories.store import MemoryStore
from src.search.engine import SearchEngine
from src.vector.embeddings import NullEmbeddingEngine
from src.vector.store import VectorStore
from src.ingest.orchestrator import run_ingest, run_ingest_folder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scanner(db_path: str) -> ArtifactScanner:
    """Build a scanner without optional OCR/embedding models (fast, robust).

    This still uses the REAL ArtifactScanner.index_folder and the REAL
    run_ingest_folder / run_ingest orchestration code.
    """
    artifact_store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    extractor = Extractor(NullOCREngine())
    scanner = ArtifactScanner(
        artifact_store,
        extractor,
        vector_store=vector_store,
        embedding_engine=NullEmbeddingEngine(),
    )
    return scanner


def _write(path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def _seed(folder, files: dict) -> None:
    for name, content in files.items():
        _write(os.path.join(folder, name), content)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# 1. Core wiring: real index + orchestrator populates events/episodes/memories
# ---------------------------------------------------------------------------

def test_run_ingest_folder_populates_stores(tmp_path):
    db = str(tmp_path / "ls.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    _seed(folder, {
        "qdrant_notes.txt": "qdrant vector database notes",
        "mongodb_error.png": "placeholder image content",
        "todo.md": "buy milk and eggs",
    })

    scanner = _make_scanner(db)
    totals = run_ingest_folder(scanner, str(folder), db)

    assert totals["processed"] >= 3
    assert totals["events"] >= 3, "each indexed file should yield a creation event"
    assert totals["episodes"] >= 1, "events should be grouped into episodes"
    assert totals["memories"] >= 1, "episodes should yield memories"

    # Verify real persistence via fresh store connections (not the orchestrator's).
    es = EventStore(db)
    assert len(es.query_events()) >= 3
    es.close()

    eps = EpisodeStore(db)
    assert len(eps.get_episodes()) >= 1
    eps.close()

    ms = MemoryStore(db)
    assert len(ms.get_memories()) >= 1
    ms.close()


# ---------------------------------------------------------------------------
# 2. Search enrichment actually contains real episode/memory data
# ---------------------------------------------------------------------------

def test_search_enrichment_after_ingest(tmp_path):
    db = str(tmp_path / "ls.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(os.path.join(folder, "qdrant_notes.txt"), "qdrant vector database notes for retrieval")

    scanner = _make_scanner(db)
    run_ingest_folder(scanner, str(folder), db)

    engine = SearchEngine(
        ArtifactStore(db),
        episode_store=EpisodeStore(db),
        memory_store=MemoryStore(db),
        vector_store=VectorStore(db),
        embedding_engine=NullEmbeddingEngine(),
    )
    results = engine.search("qdrant")
    assert results, "expected at least one search result"

    top = results[0]
    assert len(top.episodes) >= 1, "search result must be enriched with real episodes"
    assert len(top.evidence) >= 1, "search result must include evidence"
    assert any(ev["type"] == "episode" for ev in top.evidence)


# ---------------------------------------------------------------------------
# 3. Idempotency: repeated unchanged index must not duplicate events
# ---------------------------------------------------------------------------

def test_run_ingest_idempotent_repeat(tmp_path):
    db = str(tmp_path / "ls.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    _seed(folder, {"a.txt": "alpha", "b.txt": "beta"})

    scanner = _make_scanner(db)
    t1 = run_ingest_folder(scanner, str(folder), db)

    es = EventStore(db)
    n1 = len(es.query_events())
    es.close()

    t2 = run_ingest_folder(scanner, str(folder), db)

    es = EventStore(db)
    n2 = len(es.query_events())
    es.close()

    assert n2 == n1, "re-indexing an unchanged folder must not duplicate events"
    assert t2["events"] == 0, "second run should generate 0 new events"
    assert t2["episodes"] == t1["episodes"], "episode count must be stable (INSERT OR REPLACE)"


# ---------------------------------------------------------------------------
# 4. Force re-index must not duplicate creation events
# ---------------------------------------------------------------------------

def test_run_ingest_reindex_idempotent(tmp_path):
    db = str(tmp_path / "ls.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    _write(os.path.join(folder, "a.txt"), "alpha")

    scanner = _make_scanner(db)
    run_ingest_folder(scanner, str(folder), db, reindex=False)

    es = EventStore(db)
    n1 = len(es.query_events())
    es.close()

    run_ingest_folder(scanner, str(folder), db, reindex=True)

    es = EventStore(db)
    n2 = len(es.query_events())
    es.close()

    assert n2 == n1, "force re-index must not duplicate creation events"


# ---------------------------------------------------------------------------
# 5. Deleted file: no crash, no duplication, artifact marked missing
#    (NOTE: the existing FilesystemEventSource does not emit FILE_DELETED
#     in this flow because index_folder pre-marks the artifact missing
#     before generate_events runs; that is a pre-existing component
#     limitation, not introduced by C8-1. We assert the safe behavior.)
# ---------------------------------------------------------------------------

def test_run_ingest_deleted_file_no_crash_no_duplication(tmp_path):
    db = str(tmp_path / "ls.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    f = os.path.join(folder, "a.txt")
    _write(f, "alpha")
    _write(os.path.join(folder, "b.txt"), "beta")

    scanner = _make_scanner(db)
    run_ingest_folder(scanner, str(folder), db)

    es = EventStore(db)
    n1 = len(es.query_events())
    es.close()

    os.remove(f)
    # Re-ingest simulates a later indexing pass after deletion.
    run_ingest_folder(scanner, str(folder), db)

    es = EventStore(db)
    n2 = len(es.query_events())
    es.close()
    assert n2 == n1, "deletion must not cause duplicate events"

    store = ArtifactStore(db)
    row = store.get_artifact_by_path(os.path.abspath(f))
    assert row is not None and bool(row["missing"]), "deleted artifact should be marked missing"
    store.close()


# ---------------------------------------------------------------------------
# 6. Multi-folder orchestration (run_ingest convenience wrapper)
# ---------------------------------------------------------------------------

def test_run_ingest_multi_folder(tmp_path):
    db = str(tmp_path / "ls.db")
    f1 = tmp_path / "d1"
    f2 = tmp_path / "d2"
    f1.mkdir()
    f2.mkdir()
    _write(f1 / "x.txt", "xcontent")
    _write(f2 / "y.txt", "ycontent")

    scanner = _make_scanner(db)
    totals = run_ingest(scanner, [str(f1), str(f2)], db)

    assert totals["processed"] >= 2
    assert totals["episodes"] >= 1
    assert totals["memories"] >= 1


# ---------------------------------------------------------------------------
# 7. CLI indexing path wires the pipeline (uses the real build_stack)
# ---------------------------------------------------------------------------

def test_cli_index_wires_pipeline(tmp_path):
    from src.cli.main import main

    db = str(tmp_path / "cli.db")
    folder = tmp_path / "ws"
    folder.mkdir()
    _write(os.path.join(folder, "notes.txt"), "qdrant vector search notes")

    captured = io.StringIO()
    old = sys.stdout
    sys.stdout = captured
    try:
        rc = main(["--db", db, "index", str(folder)])
    finally:
        sys.stdout = old

    assert rc == 0
    assert "episodes" in captured.getvalue()

    es = EventStore(db)
    assert len(es.query_events()) >= 1
    es.close()

    eps = EpisodeStore(db)
    assert len(eps.get_episodes()) >= 1
    eps.close()

    ms = MemoryStore(db)
    assert len(ms.get_memories()) >= 1
    ms.close()


# ---------------------------------------------------------------------------
# 8. HTTP indexing path wires the pipeline (POST /index, then search)
# ---------------------------------------------------------------------------

def test_server_index_wires_pipeline(tmp_path):
    from src.server.app import LifeSearchServer, init_stack, get_search_engine

    db = str(tmp_path / "srv.db")
    folder = tmp_path / "ws"
    folder.mkdir()
    _write(os.path.join(folder, "qdrant_notes.txt"), "qdrant vector database notes")

    port = _find_free_port()
    server = LifeSearchServer(host="127.0.0.1", port=port)
    server.start(db)
    time.sleep(0.2)
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST", "/index",
            json.dumps({"paths": [str(folder)]}),
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        assert resp.status == 202, resp.status
        conn.close()

        deadline = time.time() + 30
        status = None
        while time.time() < deadline:
            conn = HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("GET", "/index/status")
            status = json.loads(conn.getresponse().read().decode())
            conn.close()
            if not status["indexing_in_progress"]:
                break
            time.sleep(0.1)

        assert status is not None
        assert status["indexing_in_progress"] is False
        assert status["episodes"] >= 1, "server indexing should produce episodes"

        engine = get_search_engine()
        results = engine.search("qdrant")
        assert results, "expected search results after server indexing"
        assert len(results[0].episodes) >= 1, "server-indexed results must be enriched"
    finally:
        server.shutdown()
