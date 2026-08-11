"""
tests/test_search_5b.py

End-to-end integration tests for Slice 5B Contextual Retrieval.
Uses a deterministic corpus with fixed absolute timestamps.
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.engine import EpisodeEngine
from src.episodes.store import EpisodeStore
from src.events.model import Event
from src.events.store import EventStore
from src.memories.builder import MemoryBuilder
from src.memories.store import MemoryStore
from src.search.engine import SearchEngine


def _write_file(path: str, content: str = "") -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


@pytest.fixture
def corpus_env(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    folder = str(tmp_path / "files")
    os.makedirs(folder, exist_ok=True)

    # 1. Create Files
    f_qdrant = os.path.join(folder, "qdrant_notes.pdf")
    f_retrieval = os.path.join(folder, "retrieval.txt")
    f_mongo = os.path.join(folder, "mongodb_error.png")
    f_jyotish = os.path.join(folder, "jyotishai_architecture.md")
    f_qlora = os.path.join(folder, "qlora_notes.md")
    f_vacation = os.path.join(folder, "vacation.jpg")

    _write_file(f_qdrant, "Qdrant is a vector database for semantic search. Downloaded May 2026.")
    _write_file(f_retrieval, "# retrieval pipeline\nimport qdrant_client\n# modified after reading qdrant notes")
    _write_file(f_mongo, "")
    _write_file(f_jyotish, "JyotishAI system architecture document. Components: astrology engine, recommendation pipeline.")
    _write_file(f_qlora, "Notes on QLoRA fine-tuning. 4-bit quantization reduces VRAM requirements.")
    _write_file(f_vacation, "")

    # 2. Index Artifacts
    artifact_store = ArtifactStore(db_path)
    scanner = ArtifactScanner(artifact_store, Extractor())
    scanner.index_folder(folder)

    id_qdrant = int(artifact_store.get_artifact_by_path(f_qdrant)["id"])
    id_retrieval = int(artifact_store.get_artifact_by_path(f_retrieval)["id"])
    id_mongo = int(artifact_store.get_artifact_by_path(f_mongo)["id"])
    id_jyotish = int(artifact_store.get_artifact_by_path(f_jyotish)["id"])
    id_qlora = int(artifact_store.get_artifact_by_path(f_qlora)["id"])
    id_vacation = int(artifact_store.get_artifact_by_path(f_vacation)["id"])

    # 3. Create Events
    event_store = EventStore(db_path)
    events = [
        Event("e1", "FILE_DOWNLOADED", "2026-05-14T14:00:00+00:00", "test", "simulated", artifact_id=id_qdrant, payload={"path": f_qdrant}),
        Event("e2", "FILE_OPENED", "2026-05-14T14:05:00+00:00", "test", "simulated", artifact_id=id_qdrant, payload={"path": f_qdrant}),
        Event("e3", "FILE_MODIFIED", "2026-05-14T14:20:00+00:00", "test", "simulated", artifact_id=id_retrieval, payload={"path": f_retrieval}),
        Event("e4", "SCREENSHOT_DISCOVERED", "2026-05-14T14:30:00+00:00", "test", "filesystem", artifact_id=id_mongo, payload={"path": f_mongo}),
        Event("e5", "FILE_MODIFIED", "2026-07-15T10:00:00+00:00", "test", "simulated", artifact_id=id_jyotish, payload={"path": f_jyotish}),
        Event("e6", "FILE_OPENED", "2026-07-15T10:05:00+00:00", "test", "simulated", artifact_id=id_qlora, payload={"path": f_qlora}),
        Event("e7", "SCREENSHOT_DISCOVERED", "2026-08-05T21:30:00+00:00", "test", "filesystem", artifact_id=id_vacation, payload={"path": f_vacation}),
    ]
    for ev in events:
        event_store.append_event(ev)

    # 4. Group Episodes
    episode_engine = EpisodeEngine(max_episode_gap_seconds=1800)
    episode_store = EpisodeStore(db_path)
    episodes = episode_engine.detect_and_persist(
        event_store, episode_store,
        start_ts="2026-05-01T00:00:00+00:00",
        end_ts="2026-08-31T23:59:59+00:00",
    )

    # 5. Build Memories
    memory_store = MemoryStore(str(tmp_path / "memories.db"))
    mem_builder = MemoryBuilder(event_provider=event_store)
    memories = [m for ep in episodes if (m := mem_builder.build(ep)) is not None]
    memory_store.save_memories(
        memories,
        range_start_ts="2026-05-01T00:00:00+00:00",
        range_end_ts="2026-08-31T23:59:59+00:00",
    )

    search_engine = SearchEngine(artifact_store, episode_store, memory_store)

    return {
        "search_engine": search_engine,
        "artifact_store": artifact_store,
        "ref_date": datetime(2026, 8, 12, 12, 0, 0, tzinfo=timezone.utc),
    }


def test_query_qdrant_pdf(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("Qdrant PDF", reference_date=corpus_env["ref_date"])
    assert len(results) >= 1
    assert results[0].file_name == "qdrant_notes.pdf"
    assert "qdrant" in results[0].why.lower()


def test_query_qdrant_pdf_around_may(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("Qdrant PDF around May", reference_date=corpus_env["ref_date"])
    assert len(results) >= 1
    top = results[0]
    assert top.file_name == "qdrant_notes.pdf"
    assert top.score > 0.4
    assert len(top.episodes) >= 1


def test_query_mongodb_screenshot(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("MongoDB screenshot", reference_date=corpus_env["ref_date"])
    assert len(results) >= 1
    assert results[0].file_name == "mongodb_error.png"
    assert "mongodb" in results[0].why.lower() or "image" in results[0].why.lower()


def test_query_activity_on_specific_date(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("What files did I work on 2026-05-14?", reference_date=corpus_env["ref_date"])
    assert len(results) >= 2
    filenames = [r.file_name for r in results]
    assert "qdrant_notes.pdf" in filenames
    assert "retrieval.txt" in filenames


def test_query_jyotishai(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("JyotishAI", reference_date=corpus_env["ref_date"])
    assert len(results) >= 1
    assert results[0].file_name == "jyotishai_architecture.md"


def test_query_qlora_document(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("QLoRA document", reference_date=corpus_env["ref_date"])
    assert len(results) >= 1
    assert results[0].file_name == "qlora_notes.md"


def test_isolation_query_does_not_return_vacation(corpus_env):
    engine = corpus_env["search_engine"]
    results = engine.search("qdrant", reference_date=corpus_env["ref_date"])
    filenames = [r.file_name for r in results]
    assert "vacation.jpg" not in filenames
