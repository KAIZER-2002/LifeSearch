"""
Production-readiness hardening tests for C8-5 (V1 hardening milestone).

These tests prove the durability and orphaned-derived-data guarantees at the
data layer and through the real runtime ingest orchestrator. They are
deterministic: no network, no model downloads, no external services.

Coverage:
  A. WAL + foreign_keys are actually enabled on every store connection.
  B. Orphaned derived data is removed when an artifact goes missing:
       - vector chunks (per artifact)
       - filesystem events (per artifact)
       - dangling episodes (preserving multi-artifact episodes)
       - dangling memories (preserving episode/artifact-backed memories)
  C. Unrelated artifacts are preserved when one artifact is deleted.
  D. Repeated indexing is idempotent (no duplicate events/artifacts).
  E. Partial indexing failure (per-file extraction error) does not abort the run.
  F. API/spec alignment: /status model fields, /feedback size cap (413),
     reserved filters accepted, /index priority accepted-and-ignored.
  G. Recovery: after deletion + reindex, present data remains queryable/intact.
"""

import json
import os
import socket
import sqlite3
import tempfile
import time
from http.client import HTTPConnection
from typing import Callable, Dict, List

import pytest

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.model import Episode
from src.episodes.store import EpisodeStore
from src.events.model import Event
from src.events.store import EventStore
from src.ingest.orchestrator import run_ingest_folder
from src.memories.model import Memory
from src.memories.store import MemoryStore
from src.server.app import LifeSearchServer
from src.vector.backends.sqlite_exact import SQLiteExactBackend
from src.vector.chunker import TextChunk


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


# ---------------------------------------------------------------------------
# A. Durability pragmas (WAL + foreign_keys) are actually enabled
# ---------------------------------------------------------------------------


def _store_factories() -> List[tuple]:
    return [
        ("ArtifactStore", lambda db: ArtifactStore(db)),
        ("EventStore", lambda db: EventStore(db)),
        ("EpisodeStore", lambda db: EpisodeStore(db)),
        ("MemoryStore", lambda db: MemoryStore(db)),
        (
            "SQLiteExactBackend",
            lambda db: SQLiteExactBackend(db, check_same_thread=False),
        ),
    ]


def test_durability_pragmas_enabled_on_all_stores():
    """WAL must be on (and persisted) and foreign_keys must be ON per connection."""
    for name, factory in _store_factories():
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "test.db")
            store = factory(db)
            try:
                # The live connection the store uses must be in WAL + FK mode.
                jm = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
                fk = store.conn.execute("PRAGMA foreign_keys").fetchone()[0]
                assert jm == "wal", f"{name}: journal_mode should be 'wal', got {jm!r}"
                assert fk == 1, f"{name}: foreign_keys should be 1 (ON), got {fk!r}"

                # WAL is persisted in the database header: a brand-new connection
                # to the same file must also report 'wal'.
                second = sqlite3.connect(db)
                try:
                    jm2 = second.execute("PRAGMA journal_mode").fetchone()[0]
                finally:
                    second.close()
                assert jm2 == "wal", f"{name}: persisted journal_mode should be 'wal', got {jm2!r}"
            finally:
                store.conn.close()


# ---------------------------------------------------------------------------
# B. Orphaned derived-data cleanup primitives (unit level)
# ---------------------------------------------------------------------------


def test_vector_chunk_cleanup_removes_only_target_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "vec.db")
        backend = SQLiteExactBackend(db, check_same_thread=False)
        try:
            chunks_a = [
                TextChunk("a_c0", 1, 0, 5, 0, "document_text"),
                TextChunk("a_c1", 1, 5, 10, 1, "document_text"),
            ]
            chunks_b = [TextChunk("b_c0", 2, 0, 5, 0, "document_text")]
            backend.save_chunks(1, chunks_a, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "t", 3)
            backend.save_chunks(2, chunks_b, [[0.7, 0.8, 0.9]], "t", 3)

            assert len(backend.get_labels_for_artifact(1)) == 2
            assert len(backend.get_labels_for_artifact(2)) == 1

            backend.delete_artifact_chunks(1)

            assert backend.get_labels_for_artifact(1) == []
            # Unrelated artifact's chunks survive.
            assert len(backend.get_labels_for_artifact(2)) == 1
        finally:
            backend.close()


def test_event_cleanup_removes_only_target_artifact():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ev.db")
        store = EventStore(db)
        try:
            store.append_event(
                Event(id="e1", type="FILE_CREATED", timestamp="2026-01-01T00:00:00Z",
                      source="test", source_kind="filesystem", artifact_id=1)
            )
            store.append_event(
                Event(id="e2", type="FILE_CREATED", timestamp="2026-01-01T00:00:00Z",
                      source="test", source_kind="filesystem", artifact_id=2)
            )
            assert len(store.get_events_for_artifact(1)) == 1
            assert len(store.get_events_for_artifact(2)) == 1

            store.delete_events_for_artifact(1)

            assert store.get_events_for_artifact(1) == []
            assert len(store.get_events_for_artifact(2)) == 1
        finally:
            store.conn.close()


def test_episode_pruning_preserves_multi_artifact_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ep.db")
        store = EpisodeStore(db)
        try:
            ep_a = Episode(
                id="a", start_ts="2026-01-01T00:00:00Z", end_ts="2026-01-01T00:05:00Z",
                event_ids=[], artifact_ids=[1], grouping_confidence=0.5,
                title="Only artifact 1", evidence=[],
            )
            ep_b = Episode(
                id="b", start_ts="2026-01-01T00:00:00Z", end_ts="2026-01-01T00:05:00Z",
                event_ids=[], artifact_ids=[1, 2], grouping_confidence=0.5,
                title="Shared by 1 and 2", evidence=[],
            )
            store.save_episodes([ep_a, ep_b])

            # Only artifact 2 is still present -> ep_a is fully dangling, ep_b kept.
            removed = store.prune_dangling_episodes(present_artifact_ids={2})
            assert removed == 1

            remaining = {e.id for e in store.get_episodes()}
            assert "b" in remaining
            assert "a" not in remaining
        finally:
            store.conn.close()


def test_memory_pruning_preserves_episode_backed_memories():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "mem.db")
        store = MemoryStore(db)
        try:
            mem_a = Memory(
                id="m_a", episode_ids=[], event_ids=[], artifact_ids=[1],
                start_ts="2026-01-01T00:00:00Z", end_ts="2026-01-01T00:05:00Z",
                title="Only artifact 1", topics=[], confidence=0.5, evidence=[],
            )
            mem_b = Memory(
                id="m_b", episode_ids=["eB"], event_ids=[], artifact_ids=[],
                start_ts="2026-01-01T00:00:00Z", end_ts="2026-01-01T00:05:00Z",
                title="Backed by episode eB", topics=[], confidence=0.5, evidence=[],
            )
            store.save_memories([mem_a, mem_b])

            # Neither artifact 1 nor episode eB is present except eB in present set.
            removed = store.prune_dangling_memories(
                present_artifact_ids=set(), present_episode_ids={"eB"}
            )
            assert removed == 1

            remaining = {m.id for m in store.get_memories()}
            assert "m_b" in remaining
            assert "m_a" not in remaining
        finally:
            store.conn.close()


# ---------------------------------------------------------------------------
# C/D/E/G. End-to-end via the real ingest orchestrator
# ---------------------------------------------------------------------------


class _StubExtractor:
    """Mimics Extractor.extract_text but fails (returns an error tuple) for
    paths containing ``fail_substr``. This exercises the per-file failure
    tolerance path without raising (the supported Extractor contract)."""

    def __init__(self, fail_substr: str = "bad.txt") -> None:
        self.fail_substr = fail_substr

    def extract_text(self, path: str, mime_type: str):
        if self.fail_substr in path:
            return "", "simulated extraction failure"
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(), None


def _inject_chunks(backend: SQLiteExactBackend, artifact_id: int, prefix: str) -> None:
    chunks = [
        TextChunk(f"{prefix}_c0", artifact_id, 0, 5, 0, "document_text"),
        TextChunk(f"{prefix}_c1", artifact_id, 5, 10, 1, "document_text"),
    ]
    embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    backend.save_chunks(artifact_id, chunks, embs, "t", 3)


def test_ingest_cleanup_preserves_unrelated_and_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = os.path.join(tmp, "workspace")
        os.makedirs(workspace, exist_ok=True)
        path_a = os.path.join(workspace, "a.txt")
        path_b = os.path.join(workspace, "b.txt")
        _create_text_file(path_a, "alpha document about onboarding")
        _create_text_file(path_b, "beta document about quarterly planning")

        db_path = os.path.join(tmp, "lifesearch.db")
        backend = SQLiteExactBackend(db_path, check_same_thread=False)
        try:
            scanner_store = ArtifactStore(db_path)
            scanner = ArtifactScanner(
                scanner_store, Extractor(), vector_store=backend
            )

            # ---- Run 1: index both files ----
            totals1 = run_ingest_folder(scanner, workspace, db_path, reindex=False)
            assert totals1["processed"] == 2
            assert totals1["events"] >= 2

            aid_a = scanner_store.get_artifact_by_path(path_a)["id"]
            aid_b = scanner_store.get_artifact_by_path(path_b)["id"]

            # Inject orphan-prone derived data for BOTH artifacts so we can prove
            # that cleanup targets only the missing one.
            _inject_chunks(backend, aid_a, "a")
            _inject_chunks(backend, aid_b, "b")
            assert len(backend.get_labels_for_artifact(aid_a)) == 2
            assert len(backend.get_labels_for_artifact(aid_b)) == 2

            with EventStore(db_path) as ev:
                assert len(ev.get_events_for_artifact(aid_a)) >= 1
                assert len(ev.get_events_for_artifact(aid_b)) >= 1

            # ---- Delete file A, then re-index ----
            os.remove(path_a)
            totals2 = run_ingest_folder(scanner, workspace, db_path, reindex=False)
            # A is now missing; B is unchanged so it is skipped.
            assert totals2["processed"] == 0

            # Vector chunks for the missing artifact are gone; B's survive.
            assert backend.get_labels_for_artifact(aid_a) == []
            assert len(backend.get_labels_for_artifact(aid_b)) == 2

            # Events for the missing artifact are gone; B's survive (preservation).
            with EventStore(db_path) as ev:
                assert ev.get_events_for_artifact(aid_a) == []
                b_events_after = len(ev.get_events_for_artifact(aid_b))
                assert b_events_after >= 1

            # The missing artifact is recorded as missing and excluded from present.
            with ArtifactStore(db_path) as art:
                assert art.list_present_artifact_ids() == [aid_b]

            # ---- Run 3: idempotent on the unchanged (A deleted, B present) state ----
            totals3 = run_ingest_folder(scanner, workspace, db_path, reindex=False)
            assert totals3["processed"] == 0
            assert totals3["events"] == 0  # no duplicate events generated

            # Present data is intact after recovery (no silent re-cleanup damage).
            assert len(backend.get_labels_for_artifact(aid_b)) == 2
            with EventStore(db_path) as ev:
                assert len(ev.get_events_for_artifact(aid_b)) == b_events_after
            with ArtifactStore(db_path) as art:
                assert art.list_present_artifact_ids() == [aid_b]
        finally:
            backend.close()


def test_partial_indexing_failure_does_not_abort_run():
    with tempfile.TemporaryDirectory() as tmp:
        workspace = os.path.join(tmp, "workspace")
        os.makedirs(workspace, exist_ok=True)
        good = os.path.join(workspace, "good.txt")
        bad = os.path.join(workspace, "bad.txt")
        _create_text_file(good, "healthy document content")
        _create_text_file(bad, "corrupt document content")

        db_path = os.path.join(tmp, "lifesearch.db")
        scanner_store = ArtifactStore(db_path)
        # Stub extractor fails on bad.txt but still returns a tuple (no raise).
        scanner = ArtifactScanner(
            scanner_store, _StubExtractor(fail_substr="bad.txt"), vector_store=None
        )

        totals = run_ingest_folder(scanner, workspace, db_path, reindex=False)

        # Both artifacts are still recorded; one merely carries an extract error.
        assert totals["processed"] == 2
        assert totals["errors"] == 0

        with ArtifactStore(db_path) as art:
            good_row = art.get_artifact_by_path(good)
            bad_row = art.get_artifact_by_path(bad)
            assert good_row is not None
            assert bad_row is not None
            # The healthy file's text was extracted; the bad file's was not.
            assert good_row["extracted_text"] == "healthy document content"
            assert bad_row["extracted_text"] in (None, "")


# ---------------------------------------------------------------------------
# F. API / spec alignment
# ---------------------------------------------------------------------------


class TestProductionReadinessAPI:
    """In-process server tests (matches test_server.py lifecycle conventions)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "workspace")
        os.makedirs(self.artifact_dir, exist_ok=True)
        _create_text_file(
            os.path.join(self.artifact_dir, "notes.txt"),
            "searchable content about the onboarding project",
        )
        _create_text_file(
            os.path.join(self.artifact_dir, "story.md"),
            "a markdown story about daily life",
        )

        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.3)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _make_request(self, method: str, path: str, body=None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Content-Type": "application/json"} if body is not None else {}
        try:
            if body is not None:
                conn.request(method, path, json.dumps(body), headers)
            else:
                conn.request(method, path)
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            return response.status, data
        finally:
            conn.close()

    def test_status_includes_model_fields(self):
        status, data = self._make_request("GET", "/status")
        assert status == 200
        resp = json.loads(data)
        # C8-5D: spec requires model_available (bool) and model_id (str|null).
        assert "model_available" in resp
        assert "model_id" in resp
        assert isinstance(resp["model_available"], bool)
        assert resp["model_id"] is None or isinstance(resp["model_id"], str)

    def test_feedback_oversized_request_rejected(self):
        # C8-5D: feedback payloads above 64 KiB must be rejected with 413.
        big = " " * 70000
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(
                "POST", "/feedback", big, {"Content-Type": "application/json"}
            )
            response = conn.getresponse()
            assert response.status == 413
        finally:
            conn.close()

    def test_search_accepts_reserved_filters(self):
        # app_origin / project are accepted-for-compatibility but ignored.
        status, data = self._make_request(
            "POST",
            "/search",
            {"query": "searchable", "k": 5, "filters": {"app_origin": "x", "project": "y"}},
        )
        assert status == 200
        resp = json.loads(data)
        assert isinstance(resp["results"], list)

    def test_index_accepts_priority_ignored(self):
        # C8-5D: priority is accepted by the server but has no behavioral effect.
        index_dir = os.path.join(self.temp_dir.name, "to_index")
        os.makedirs(index_dir, exist_ok=True)
        _create_text_file(os.path.join(index_dir, "a.txt"), "index me please")
        status, data = self._make_request(
            "POST", "/index", {"paths": [index_dir], "priority": True}
        )
        assert status == 202
        resp = json.loads(data)
        assert resp.get("status") in ("scheduled", "accepted")
