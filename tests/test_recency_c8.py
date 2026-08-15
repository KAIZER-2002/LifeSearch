"""
C8-6 — Result Recency & Open Tracking tests.

C8-6A (storage + open capture):
  - last_opened_ms column migration (fresh + existing DB, idempotent)
  - update_last_opened() behavior (epoch ms, repeatable, preserves other ts)
  - CLI `open` records last_opened_ms + emits FILE_OPENED
  - HTTP GET /document/{id} records last_opened_ms + emits FILE_OPENED
  - recency failure never breaks document retrieval/open

C8-6B/C tests are appended in later phases of this same file.
"""

import json
import os
import socket
import sqlite3
import sys
import tempfile
import time
from http.client import HTTPConnection
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.events.store import EventStore
from src.search.engine import SearchEngine
from src.server.app import LifeSearchServer


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# C8-6A: storage migration + update_last_opened
# ---------------------------------------------------------------------------


def test_fresh_db_has_last_opened_column():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "fresh.db")
        store = ArtifactStore(db)
        cols = [r[1] for r in store.conn.execute("PRAGMA table_info(artifacts)")]
        assert "last_opened_ms" in cols
        store.close()


def test_existing_db_migration_compatible():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "old.db")
        # Simulate a pre-C8-6 database that lacks the last_opened_ms column.
        raw = sqlite3.connect(db)
        raw.execute(
            "CREATE TABLE artifacts ("
            "id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, file_name TEXT NOT NULL, "
            "mime_type TEXT NOT NULL, size INTEGER NOT NULL, created_at TEXT NOT NULL, "
            "modified_at TEXT NOT NULL, indexed_at TEXT NOT NULL, content_hash TEXT, "
            "extracted_text TEXT, extract_error TEXT, missing INTEGER NOT NULL DEFAULT 0)"
        )
        raw.execute(
            "INSERT INTO artifacts (path, file_name, mime_type, size, created_at, "
            "modified_at, indexed_at, missing) VALUES (?,?,?,?,?,?,?,0)",
            ("/x/a.txt", "a.txt", "text/plain", 10, "2026-01-01T00:00:00Z",
             "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"),
        )
        raw.commit()
        raw.close()

        # Opening via ArtifactStore must migrate without data loss.
        store = ArtifactStore(db)
        cols = [r[1] for r in store.conn.execute("PRAGMA table_info(artifacts)")]
        assert "last_opened_ms" in cols
        row = store.get_artifact_by_path("/x/a.txt")
        assert row["path"] == "/x/a.txt"
        assert row["file_name"] == "a.txt"
        assert row["last_opened_ms"] is None  # never opened yet
        store.close()


def test_migration_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "idem.db")
        ArtifactStore(db).close()
        # Second open must not error and column must still be present.
        store = ArtifactStore(db)
        cols = [r[1] for r in store.conn.execute("PRAGMA table_info(artifacts)")]
        assert "last_opened_ms" in cols
        store.close()


def test_update_last_opened_sets_epoch_ms():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "u.db")
        store = ArtifactStore(db)
        aid = store.upsert_artifact(
            "/x/a.txt", "a.txt", "text/plain", 10,
            "2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "text", None, None,
        )
        store.update_last_opened(aid)
        row = store.get_artifact(aid)
        assert isinstance(row["last_opened_ms"], int)
        assert row["last_opened_ms"] > 0
        # Within a generous window of "now".
        assert abs(row["last_opened_ms"] - int(time.time() * 1000)) < 3_600_000
        store.close()


def test_repeated_update_last_opened_works():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "r.db")
        store = ArtifactStore(db)
        aid = store.upsert_artifact(
            "/x/a.txt", "a.txt", "text/plain", 10,
            "c", "m", "t", None, None,
        )
        store.update_last_opened(aid)
        first = store.get_artifact(aid)["last_opened_ms"]
        time.sleep(0.01)
        store.update_last_opened(aid)
        second = store.get_artifact(aid)["last_opened_ms"]
        assert second >= first
        store.close()


def test_update_last_opened_preserves_other_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "p.db")
        store = ArtifactStore(db)
        created = "2026-01-01T00:00:00Z"
        modified = "2026-01-02T00:00:00Z"
        aid = store.upsert_artifact(
            "/x/a.txt", "a.txt", "text/plain", 10, created, modified, "t", None, None,
        )
        before = store.get_artifact(aid)
        store.update_last_opened(aid)
        after = store.get_artifact(aid)
        assert after["created_at"] == before["created_at"] == created
        assert after["modified_at"] == before["modified_at"] == modified
        assert after["indexed_at"] == before["indexed_at"]
        assert after["last_opened_ms"] is not None
        store.close()


# ---------------------------------------------------------------------------
# C8-6A: CLI open records recency + FILE_OPENED
# ---------------------------------------------------------------------------


def test_cli_open_records_recency_and_event(monkeypatch):
    from src.cli.main import main

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "cli.db")
        folder = os.path.join(tmp, "ws")
        os.makedirs(folder)
        f = os.path.join(folder, "note.txt")
        with open(f, "w", encoding="utf-8") as h:
            h.write("hello world content")

        assert main(["--db", db, "index", folder]) == 0
        store = ArtifactStore(db)
        aid = store.get_artifact_by_path(os.path.abspath(f))["id"]
        store.close()

        # Avoid spawning an OS file-opener during the test.
        monkeypatch.setenv("LIFESEARCH_DRY_RUN", "1")
        assert main(["--db", db, "open", str(aid)]) == 0

        store = ArtifactStore(db)
        row = store.get_artifact(aid)
        assert row["last_opened_ms"] is not None
        store.close()

        es = EventStore(db)
        try:
            events = es.get_events_for_artifact(aid)
            assert any(e.type == "FILE_OPENED" for e in events), "FILE_OPENED event expected"
        finally:
            es.close()


# ---------------------------------------------------------------------------
# C8-6A: HTTP GET /document/{id} records recency + FILE_OPENED
# ---------------------------------------------------------------------------


class TestHttpOpenRecency:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "srv.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "ws")
        os.makedirs(self.artifact_dir)
        self.path = os.path.join(self.artifact_dir, "note.txt")
        with open(self.path, "w", encoding="utf-8") as h:
            h.write("searchable note content")

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

    def _get(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            return resp.status, data
        finally:
            conn.close()

    def _aid(self) -> int:
        with ArtifactStore(self.db_path) as store:
            return store.get_artifact_by_path(os.path.abspath(self.path))["id"]

    def test_http_document_records_recency_and_event(self):
        aid = self._aid()
        status, _ = self._get(f"/document/{aid}")
        assert status == 200

        with ArtifactStore(self.db_path) as store:
            assert store.get_artifact(aid)["last_opened_ms"] is not None

        es = EventStore(self.db_path)
        try:
            events = es.get_events_for_artifact(aid)
            assert any(e.type == "FILE_OPENED" for e in events)
        finally:
            es.close()

    def test_repeated_opens_remain_safe(self):
        aid = self._aid()
        for _ in range(3):
            status, _ = self._get(f"/document/{aid}")
            assert status == 200
        with ArtifactStore(self.db_path) as store:
            assert store.get_artifact(aid)["last_opened_ms"] is not None

    def test_recency_failure_does_not_break_document(self, monkeypatch):
        aid = self._aid()

        def boom(artifact_id):
            raise RuntimeError("simulated recency write failure")

        monkeypatch.setattr(ArtifactStore, "update_last_opened", boom)
        # Even if recency write fails, the document read must still succeed.
        status, data = self._get(f"/document/{aid}")
        assert status == 200
        import json
        body = json.loads(data)
        assert body["document"]["id"] == aid


# ---------------------------------------------------------------------------
# C8-6B (1): GET /document/{id} exposes last_opened_ms in the RESPONSE BODY
# ---------------------------------------------------------------------------


class TestHttpDocumentLastOpened:
    """Verify the /document response body carries last_opened_ms metadata."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "doc.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "ws")
        os.makedirs(self.artifact_dir)
        self.path = os.path.join(self.artifact_dir, "note.txt")
        with open(self.path, "w", encoding="utf-8") as h:
            h.write("searchable note content")

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

    def _get(self, path: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            return resp.status, data
        finally:
            conn.close()

    def _aid(self) -> int:
        with ArtifactStore(self.db_path) as store:
            return store.get_artifact_by_path(os.path.abspath(self.path))["id"]

    def test_never_opened_is_null_before_any_get(self):
        # No GET has been issued yet in this fresh fixture. The stored value
        # must still be null, proving "never opened" -> null semantics.
        aid = self._aid()
        with ArtifactStore(self.db_path) as store:
            assert store.get_artifact(aid)["last_opened_ms"] is None

    def test_first_open_returns_epoch_ms_in_body(self):
        aid = self._aid()
        status, data = self._get(f"/document/{aid}")
        assert status == 200
        body = json.loads(data)
        doc = body["document"]
        assert "last_opened_ms" in doc
        ts = doc["last_opened_ms"]
        assert isinstance(ts, int)
        assert ts > 0
        # Within a generous window of "now".
        assert abs(ts - int(time.time() * 1000)) < 3_600_000
        # The recorded value must match what the store persisted.
        with ArtifactStore(self.db_path) as store:
            assert store.get_artifact(aid)["last_opened_ms"] == ts

    def test_repeated_opens_update_value_safely(self):
        aid = self._aid()
        seen = []
        for _ in range(3):
            status, data = self._get(f"/document/{aid}")
            assert status == 200
            body = json.loads(data)
            # Document retrieval itself still succeeds on every open.
            assert body["document"]["id"] == aid
            ts = body["document"]["last_opened_ms"]
            assert isinstance(ts, int) and ts > 0
            seen.append(ts)
            time.sleep(0.01)
        # Each successive open must be >= the previous (monotonic, safe).
        assert seen == sorted(seen)


# ---------------------------------------------------------------------------
# C8-6B (2): POST /search exposes last_opened_ms on every ResultCard
# ---------------------------------------------------------------------------


class TestHttpSearchLastOpened:
    """Verify /search ResultCards carry last_opened_ms metadata."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "search.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "ws")
        os.makedirs(self.artifact_dir)
        self.alpha = os.path.join(self.artifact_dir, "alpha.txt")
        self.beta = os.path.join(self.artifact_dir, "beta.txt")
        with open(self.alpha, "w", encoding="utf-8") as h:
            h.write("shared token unique alpha content zebra")
        with open(self.beta, "w", encoding="utf-8") as h:
            h.write("shared token unique beta content yak")

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

    def _post_search(self, query: str):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            body = json.dumps({"query": query, "k": 10})
            conn.request("POST", "/search", body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            return resp.status, data
        finally:
            conn.close()

    def _aid(self, path: str) -> int:
        with ArtifactStore(self.db_path) as store:
            return store.get_artifact_by_path(os.path.abspath(path))["id"]

    def _result_by_id(self, results, doc_id: int):
        for r in results:
            if int(r["document_id"]) == doc_id:
                return r
        return None

    def test_never_opened_result_is_null(self):
        alpha_id = self._aid(self.alpha)
        beta_id = self._aid(self.beta)
        status, data = self._post_search("shared")
        assert status == 200
        body = json.loads(data)
        assert body["results"]
        for doc_id in (alpha_id, beta_id):
            card = self._result_by_id(body["results"], doc_id)
            assert card is not None, "both docs should be in results"
            assert "last_opened_ms" in card
            assert card["last_opened_ms"] is None

    def test_opened_result_contains_epoch_ms(self):
        alpha_id = self._aid(self.alpha)
        beta_id = self._aid(self.beta)
        # Open only alpha via the document endpoint.
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", f"/document/{alpha_id}")
            assert conn.getresponse().status == 200
        finally:
            conn.close()

        status, data = self._post_search("shared")
        assert status == 200
        body = json.loads(data)
        alpha_card = self._result_by_id(body["results"], alpha_id)
        beta_card = self._result_by_id(body["results"], beta_id)
        assert alpha_card is not None and beta_card is not None
        # Opened doc -> integer epoch ms; never-opened doc -> null.
        assert isinstance(alpha_card["last_opened_ms"], int)
        assert alpha_card["last_opened_ms"] > 0
        assert beta_card["last_opened_ms"] is None
        # Recency must NOT mutate any ranking-relevant field.
        for card in body["results"]:
            for field in ("score", "why", "evidence"):
                assert field in card


# ---------------------------------------------------------------------------
# C8-6B (3): ranking invariance - last_opened_ms never changes relevance score
# ---------------------------------------------------------------------------


def test_last_opened_ms_does_not_change_search_scores():
    """End-to-end: identical candidates, only last_opened_ms differs -> same scores/order."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = os.path.join(tmp, "ws")
        os.makedirs(folder)
        pa = os.path.join(folder, "a.txt")
        pb = os.path.join(folder, "b.txt")
        with open(pa, "w", encoding="utf-8") as h:
            h.write("shared token identical content here")
        with open(pb, "w", encoding="utf-8") as h:
            h.write("shared token identical content here")

        db = os.path.join(tmp, "rank.db")
        store = ArtifactStore(db)
        scanner = ArtifactScanner(store, Extractor())
        scanner.index_folder(folder)
        engine = SearchEngine(artifact_store=store)

        before_scores = {r.id: r.score for r in engine.search("shared")}
        before_order = [r.id for r in engine.search("shared")]

        # Record an open with a large epoch value on one document only.
        aid = store.get_artifact_by_path(os.path.abspath(pa))["id"]
        store.update_last_opened(aid)

        after_scores = {r.id: r.score for r in engine.search("shared")}
        after_order = [r.id for r in engine.search("shared")]

        assert before_scores == after_scores, "last_opened_ms must not alter scores"
        assert before_order == after_order, "last_opened_ms must not alter ordering"
        store.close()


def test_rank_candidates_ignores_last_opened_ms():
    """Direct unit check: rank_candidates ignores last_opened_ms entirely."""
    from src.search.query_parser import QueryParser
    from src.search.ranking import rank_candidates

    parsed = QueryParser().parse("shared")
    candidates = [
        {
            "id": 1,
            "file_name": "a.txt",
            "mime_type": "text/plain",
            "modified_at": "2026-01-01T00:00:00Z",
            "rank": 1.0,
            "snippet": "...shared...",
            "fts_matched": True,
            "semantic_score": 0.0,
            "episodes": [],
            "memories": [],
            "last_opened_ms": None,
        },
        {
            "id": 2,
            "file_name": "b.txt",
            "mime_type": "text/plain",
            "modified_at": "2026-01-01T00:00:00Z",
            "rank": 1.0,
            "snippet": "...shared...",
            "fts_matched": True,
            "semantic_score": 0.0,
            "episodes": [],
            "memories": [],
            "last_opened_ms": None,
        },
    ]
    # Version B is identical except last_opened_ms is set to a huge value.
    candidates_b = [
        {**c, "last_opened_ms": 9_999_999_999_999} for c in candidates
    ]

    scores_a = {r[0]["id"]: r[1] for r in rank_candidates(candidates, parsed, None)}
    scores_b = {r[0]["id"]: r[1] for r in rank_candidates(candidates_b, parsed, None)}
    assert scores_a == scores_b


# ---------------------------------------------------------------------------
# C8-6C: browser UI sanity (disk inspection, no browser automation).
# Follows the existing frontend sanity-test style (read served static files).
# ---------------------------------------------------------------------------


def _read_static(name: str) -> str:
    from src.server.app import STATIC_DIR

    with open(os.path.join(STATIC_DIR, name), "r", encoding="utf-8") as fh:
        return fh.read()


class TestFrontendRecencySanity:
    """C8-6C: static UI must render recency metadata from last_opened_ms."""

    def test_app_js_references_last_opened_ms(self):
        js = _read_static("app.js")
        # The field from the API ResultCard must be referenced.
        assert "last_opened_ms" in js

    def test_recency_rendering_exists(self):
        js = _read_static("app.js")
        # A dedicated formatter must exist and be wired into the card render.
        assert "function formatLastOpened" in js
        assert "recencyBadge" in js
        assert "recency-badge" in js

    def test_never_opened_handling_exists(self):
        js = _read_static("app.js")
        # Null/never-opened must render the explicit "Never opened" label.
        assert "Never opened" in js

    def test_optional_recency_sort_control_present(self):
        js = _read_static("app.js")
        html = _read_static("index.html")
        # Client-side "Recently opened" sort is wired without changing server order.
        assert "result-sort" in html
        assert "renderSortedCards" in js
        # Default sort preserves server relevance ranking.
        assert 'value="relevance"' in html
