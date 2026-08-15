"""
Tests for the Life Search HTTP server (Phase A).

Strategy:
- Start the server in-process on an ephemeral port (never permanently 30013).
- Shut the server down deterministically in teardown.
- Do not touch external network services.

Run:
    python3 -m pytest tests/test_server.py -q
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
from typing import Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.search.engine import SearchEngine
from src.search.result import SearchResult
from src.server.app import (
    LifeSearchServer,
    init_stack,
    get_search_engine,
    get_status_info,
    STATIC_DIR,
)
import src.server.app as server_app
from src.server.mapping import (
    map_search_result,
    map_search_results,
    build_search_response,
    build_status_response,
    build_error_response,
)


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


class FakeFailingEngine:
    """Mimics SearchEngine.search() but always raises (for 500 testing)."""

    def search(self, query, limit=20, reference_date=None):
        raise RuntimeError("INTERNAL_SECRET_DETAIL_should_not_leak")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestServer:
    """Integration tests for the HTTP server (in-process, ephemeral port)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "workspace")
        os.makedirs(self.artifact_dir, exist_ok=True)

        create_text_file(os.path.join(self.artifact_dir, "notes.txt"), "searchable content about project")
        create_text_file(os.path.join(self.artifact_dir, "story.md"), "a markdown story about life")
        create_text_file(os.path.join(self.artifact_dir, "todo.txt"), "buy milk and eggs")

        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _make_request(self, method: str, path: str, body: Optional[dict] = None):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
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

    def test_server_creation_startup(self):
        assert self.server.server is not None
        assert self.server._server_thread is not None
        assert self.server._server_thread.is_alive()

    def test_get_status_returns_200(self):
        status, data = self._make_request("GET", "/status")
        assert status == 200

        response = json.loads(data)
        assert "indexed_documents" in response
        assert "indexed_chunks" in response
        assert "model_available" in response
        assert "model_id" in response
        assert response["indexed_documents"] >= 3
        assert isinstance(response["model_available"], bool)

    def test_post_search_returns_200(self):
        status, data = self._make_request("POST", "/search", {"query": "searchable", "k": 10})
        assert status == 200

        response = json.loads(data)
        assert "took_ms" in response
        assert "query_embedding_id" in response
        assert "results" in response
        assert isinstance(response["results"], list)
        assert len(response["results"]) >= 1

        result = response["results"][0]
        for field in (
            "document_id",
            "file_name",
            "path",
            "mime_type",
            "snippet",
            "highlights",
            "score",
            "why",
            "episodes",
            "memories",
            "evidence",
        ):
            assert field in result

    def test_post_search_malformed_json_returns_400(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", "/search", "not valid json", {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()

        assert response.status == 400
        error_response = json.loads(data)
        assert "error" in error_response
        assert error_response["error"]["code"] == 400
        assert "traceback" not in data.lower()
        assert "INTERNAL" not in data  # no exception text leakage

    def test_post_search_missing_query_returns_400(self):
        status, data = self._make_request("POST", "/search", {"k": 10})
        assert status == 400
        assert json.loads(data)["error"]["code"] == 400

    def test_post_search_empty_query_returns_400(self):
        status, data = self._make_request("POST", "/search", {"query": "", "k": 10})
        assert status == 400
        assert json.loads(data)["error"]["code"] == 400

    def test_post_search_invalid_k_returns_400(self):
        for k in (0, 101, "abc"):
            status, _ = self._make_request("POST", "/search", {"query": "test", "k": k})
            assert status == 400, f"k={k!r} should be 400"

    def test_post_search_engine_failure_returns_500(self):
        import src.server.app as app

        # Inject a failing engine into the running server's stack.
        app._search_engine = FakeFailingEngine()
        try:
            status, data = self._make_request("POST", "/search", {"query": "test", "k": 10})
            assert status == 500
            body = json.loads(data)
            assert body["error"]["code"] == 500
            # No internal details must leak to the client.
            assert "INTERNAL_SECRET_DETAIL" not in data
            assert "Traceback" not in data
            assert "traceback" not in data.lower()
        finally:
            # Restore a real engine for any subsequent checks.
            app._search_engine = None

    def test_no_traceback_in_responses(self):
        cases = [
            ("POST", "/search", "invalid json", 400),
            ("POST", "/search", {"query": ""}, 400),
            ("POST", "/search", {"query": "test", "k": 0}, 400),
            ("POST", "/search", {"query": "test", "k": 101}, 400),
            ("GET", "/nonexistent", None, 404),
        ]
        for method, path, body, expected_status in cases:
            conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
            headers = {"Content-Type": "application/json"} if isinstance(body, dict) else {}
            try:
                if isinstance(body, str):
                    conn.request(method, path, body, headers)
                elif body is not None:
                    conn.request(method, path, json.dumps(body), headers)
                else:
                    conn.request(method, path)
                response = conn.getresponse()
                data = response.read().decode("utf-8")
            finally:
                conn.close()

            assert response.status == expected_status, data
            assert "traceback" not in data.lower(), data
            assert "INTERNAL_SECRET" not in data

    def test_localhost_binding(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/status")
        response = conn.getresponse()
        assert response.status == 200
        conn.close()

    def test_rejects_non_localhost_host(self):
        with pytest.raises(ValueError):
            LifeSearchServer(host="0.0.0.0", port=_find_free_port())


class TestMappingLayer:
    """Unit tests for SearchResult -> API ResultCard mapping."""

    def test_map_search_result_basic(self):
        result = SearchResult(
            id=42,
            file_name="test.txt",
            path="/home/user/test.txt",
            mime_type="text/plain",
            size=100,
            modified_at="2024-01-01T00:00:00",
            rank=1.5,
            snippet="This is a test snippet",
            episodes=[],
            memories=[],
            evidence=[],
            result_type="artifact",
            score=0.85,
            why="Matched on test",
        )
        mapped = map_search_result(result)
        assert mapped["document_id"] == "42"
        assert mapped["file_name"] == "test.txt"
        assert mapped["path"] == "/home/user/test.txt"
        assert mapped["mime_type"] == "text/plain"
        assert mapped["snippet"] == "This is a test snippet"
        assert mapped["score"] == 0.85
        assert mapped["why"] == "Matched on test"
        assert mapped["episodes"] == []
        assert mapped["memories"] == []
        assert mapped["evidence"] == []

    def test_map_search_result_with_episodes_memories(self):
        result = SearchResult(
            id=1,
            file_name="doc.pdf",
            path="/doc.pdf",
            mime_type="application/pdf",
            size=1000,
            modified_at="2024-01-01T00:00:00",
            rank=1.0,
            snippet="content",
            episodes=[{"id": "ep1", "title": "Episode 1", "start_ts": "2024-01-01", "end_ts": "2024-01-02"}],
            memories=[{"id": "mem1", "title": "Memory 1", "topics": ["topic1"]}],
            evidence=[{"type": "episode", "id": "ep1", "title": "Episode 1"}],
            result_type="artifact",
            score=0.9,
            why="test",
        )
        mapped = map_search_result(result)
        assert len(mapped["episodes"]) == 1
        assert mapped["episodes"][0]["title"] == "Episode 1"
        assert len(mapped["memories"]) == 1
        assert mapped["memories"][0]["title"] == "Memory 1"
        assert len(mapped["evidence"]) == 1

    def test_map_search_results_list(self):
        results = [
            SearchResult(id=1, file_name="a.txt", path="/a.txt", mime_type="text/plain", size=10,
                         modified_at="", rank=0, snippet="", episodes=[], memories=[], evidence=[],
                         score=0.9, why=""),
            SearchResult(id=2, file_name="b.txt", path="/b.txt", mime_type="text/plain", size=10,
                         modified_at="", rank=0, snippet="", episodes=[], memories=[], evidence=[],
                         score=0.8, why=""),
        ]
        mapped = map_search_results(results)
        assert len(mapped) == 2
        assert mapped[0]["document_id"] == "1"
        assert mapped[1]["document_id"] == "2"

    def test_build_search_response(self):
        results = [
            SearchResult(id=1, file_name="a.txt", path="/a.txt", mime_type="text/plain", size=10,
                         modified_at="", rank=0, snippet="", episodes=[], memories=[], evidence=[],
                         score=0.9, why=""),
        ]
        response = build_search_response(results, took_ms=123, query_embedding_id="emb-123")
        assert response["took_ms"] == 123
        assert response["query_embedding_id"] == "emb-123"
        assert len(response["results"]) == 1

    def test_build_status_response(self):
        response = build_status_response(
            indexed_documents=100,
            indexed_chunks=500,
            model_available=True,
            model_id="all-MiniLM-L6-v2",
        )
        assert response["indexed_documents"] == 100
        assert response["indexed_chunks"] == 500
        assert response["model_available"] is True
        assert response["model_id"] == "all-MiniLM-L6-v2"

    def test_build_error_response(self):
        error = build_error_response(400, "Bad Request", "Query is required")
        assert error["error"]["code"] == 400
        assert error["error"]["message"] == "Bad Request"
        assert error["error"]["detail"] == "Query is required"

        error = build_error_response(500, "Internal Server Error")
        assert error["error"]["code"] == 500
        assert "detail" not in error["error"]


class TestServerModule:
    """Tests for the server module's stack helpers."""

    def test_init_stack_and_get_search_engine(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            with ArtifactStore(db_path) as store:
                scanner = ArtifactScanner(store, Extractor())
                create_text_file(os.path.join(temp_dir, "test.txt"), "hello world")
                scanner.index_folder(temp_dir)

            init_stack(db_path)
            try:
                engine = get_search_engine()
                assert isinstance(engine, SearchEngine)
                assert len(engine.search("hello")) >= 1

                status = get_status_info()
                assert status["indexed_documents"] >= 1
                assert isinstance(status["model_available"], bool)
            finally:
                from src.server.app import _close_current_stack
                _close_current_stack()

    def test_command_serve_signature(self):
        from src.server.app import command_serve
        import inspect

        sig = inspect.signature(command_serve)
        assert "args" in sig.parameters


class TestConcurrentRequests:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "workspace")
        os.makedirs(self.artifact_dir, exist_ok=True)
        create_text_file(os.path.join(self.artifact_dir, "test.txt"), "concurrent test content")

        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def test_concurrent_search_requests(self):
        def make_request():
            conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("POST", "/search", json.dumps({"query": "test", "k": 5}),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            data = response.read().decode("utf-8")
            conn.close()
            return response.status, data

        threads = []
        results = []
        lock = threading.Lock()

        def worker():
            status, data = make_request()
            with lock:
                results.append((status, data))

        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=5)

        for status, data in results:
            assert status == 200, data
            assert "results" in json.loads(data)


class TestCLIBehavior:
    """Verify existing CLI behavior remains intact after Phase A."""

    def test_parser_keeps_existing_and_serve_commands(self):
        from src.cli.main import create_parser

        parser = create_parser()
        # These must not raise (existing commands preserved).
        parser.parse_args(["index", "/some/folder"])
        parser.parse_args(["status"])
        parser.parse_args(["search", "hello"])
        parser.parse_args(["open", "5"])
        parser.parse_args(["serve"])

    def test_serve_defaults_are_localhost(self):
        from src.cli.main import create_parser

        args = create_parser().parse_args(["serve"])
        assert args.host == "127.0.0.1"
        assert args.port == 30013

        args2 = create_parser().parse_args(
            ["serve", "--host", "127.0.0.1", "--port", "9999"]
        )
        assert args2.host == "127.0.0.1"
        assert args2.port == 9999

    def test_existing_search_command_still_works(self, tmp_path):
        from src.cli.main import main

        db = str(tmp_path / "cli.db")
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "notes.txt").write_text("searchable content about project")

        with ArtifactStore(db) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(str(workspace))

        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            rc = main(["--db", db, "search", "searchable"])
        finally:
            sys.stdout = old_stdout

        assert rc == 0
        assert "notes.txt" in captured.getvalue()


class TestStaticServing:
    """GET / and the static UI assets must be served from the whitelist only."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        # Ensure the DB exists; static serving does not require indexed files.
        with ArtifactStore(self.db_path):
            pass
        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8")
            return resp.status, body, dict(resp.getheaders())
        finally:
            conn.close()

    def test_get_root_returns_html(self):
        status, body, _ = self._get("/")
        assert status == 200
        assert "<!DOCTYPE html>" in body or "<html" in body

    def test_get_index_html_returns_html(self):
        status, body, _ = self._get("/index.html")
        assert status == 200
        assert "<!DOCTYPE html>" in body

    def test_static_js_served(self):
        status, body, headers = self._get("/app.js")
        assert status == 200
        ctype = headers.get("Content-Type", "").lower()
        assert "javascript" in ctype
        assert "fetch" in body

    def test_static_css_served(self):
        status, body, headers = self._get("/styles.css")
        assert status == 200
        ctype = headers.get("Content-Type", "").lower()
        assert "text/css" in ctype

    def test_unknown_static_path_404(self):
        status, _, _ = self._get("/randomfile.js")
        assert status == 404

    def test_path_traversal_404_or_400(self):
        for path in ("/../etc/passwd", "/static/../../etc/passwd", "/..%2f..%2fetc%2fpasswd"):
            status, _, _ = self._get(path)
            assert status in (400, 404), path


class TestDocumentEndpoint:
    """GET /document/{id} exposes existing artifact metadata + text only."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "ws")
        os.makedirs(self.artifact_dir)
        create_text_file(os.path.join(self.artifact_dir, "doc.txt"), "alpha beta gamma content")
        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)
            row = store.get_artifact_by_path(os.path.abspath(os.path.join(self.artifact_dir, "doc.txt")))
            self.doc_id = int(row["id"])

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _get(self, path):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read().decode("utf-8")
        finally:
            conn.close()

    def test_document_returns_metadata_and_text(self):
        status, data = self._get("/document/" + str(self.doc_id))
        assert status == 200
        body = json.loads(data)
        assert "document" in body
        doc = body["document"]
        assert doc["file_name"] == "doc.txt"
        assert "extracted_text" in doc
        assert doc["path"]
        assert doc["mime_type"]

    def test_document_unknown_404(self):
        status, _ = self._get("/document/999999999")
        assert status == 404

    def test_document_invalid_id_400(self):
        status, _ = self._get("/document/notanid")
        assert status == 400


class TestFrontendSanity:
    """Lightweight checks that the UI is self-contained and references the API.

    No browser automation; just inspects the served static files on disk.
    """

    def _read(self, name):
        with open(os.path.join(STATIC_DIR, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_index_references_assets(self):
        html = self._read("index.html")
        assert 'href="/styles.css"' in html
        assert 'src="/app.js"' in html

    def test_app_js_references_search_and_document(self):
        js = self._read("app.js")
        assert '"/search"' in js or "'/search'" in js
        assert "/document/" in js

    def test_required_ui_elements_exist(self):
        html = self._read("index.html")
        for needed in (
            'id="search-input"',
            'id="search-button"',
            'id="results"',
            'id="empty-state"',
            'id="loading-state"',
            'id="error-state"',
        ):
            assert needed in html, needed

    def test_no_external_network_urls(self):
        for name in ("index.html", "app.js", "styles.css"):
            content = self._read(name)
            assert "http://" not in content
            assert "https://" not in content
            assert "cdn." not in content
            assert 'src="http' not in content
            assert 'src="//' not in content
            assert 'url(http' not in content


class TestIndexingUISanity:
    """Phase C UI: the browser front-end must drive the C2 indexing API.

    No browser automation; inspects the served static files on disk.
    """

    def _read(self, name):
        with open(os.path.join(STATIC_DIR, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_index_html_has_indexing_panel(self):
        html = self._read("index.html")
        for needed in (
            'id="index-section"',
            'id="index-form"',
            'id="index-path"',
            'id="index-reindex"',
            'id="index-button"',
            'id="index-status"',
            'id="index-progress-bar"',
            'id="index-processed"',
            'id="index-skipped"',
            'id="index-errors"',
            'id="index-current-folder"',
            'id="index-error"',
        ):
            assert needed in html, needed

    def test_index_html_no_external_network_urls(self):
        html = self._read("index.html")
        assert "http://" not in html
        assert "https://" not in html
        assert "cdn." not in html

    def test_app_js_references_index_endpoints(self):
        js = self._read("app.js")
        assert '"/index"' in js or "'/index'" in js
        assert '"/index/status"' in js or "'/index/status'" in js

    def test_app_js_implements_indexing_flow(self):
        js = self._read("app.js")
        # Drives the API: schedules indexing, polls status, renders progress.
        assert "indexing_in_progress" in js
        assert "progress_percent" in js
        assert "last_indexed_ms" in js
        # Re-index flag is forwarded to the backend.
        assert "reindex" in js
        # Friendly error handling on failure.
        assert "indexError" in js or "showIndexError" in js


# ---------------------------------------------------------------------------
# Phase C4: Search feedback capture & ranking signal endpoint tests
# ---------------------------------------------------------------------------
from src.feedback.store import FeedbackStore


class _FakeFailingFeedbackStore:
    """FeedbackStore whose record() always raises (secret-free message)."""

    def record(self, query, document_id, action):
        raise RuntimeError("SECRET_DB_PATH_/tmp/leak details should not surface")

    def close(self):
        pass


class TestFeedbackStore:
    """Unit tests for the isolated FeedbackStore (no server)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "feedback.db")
        self.store = FeedbackStore(self.db_path)

    def teardown_method(self):
        try:
            self.store.close()
        except Exception:
            pass
        self.temp_dir.cleanup()

    def test_record_and_read_click(self):
        self.store.record("find qdrant pdf", "42", "click")
        rows = self.store.get_feedback(document_id="42")
        assert len(rows) == 1
        row = rows[0]
        assert row["action"] == "click"
        assert row["query"] == "find qdrant pdf"
        assert row["document_id"] == "42"
        assert isinstance(row["timestamp"], int)

    def test_record_ignore(self):
        self.store.record("q", "1", "ignore")
        rows = self.store.get_feedback(document_id="1")
        assert len(rows) == 1
        assert rows[0]["action"] == "ignore"

    def test_record_pin(self):
        self.store.record("q", "2", "pin")
        rows = self.store.get_feedback(document_id="2")
        assert len(rows) == 1
        assert rows[0]["action"] == "pin"

    def test_read_back_multiple(self):
        for i in range(3):
            self.store.record("q", str(i), "click")
        assert len(self.store.get_feedback()) == 3

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError):
            self.store.record("q", "1", "explode")

    def test_get_counts(self):
        self.store.record("q", "1", "click")
        self.store.record("q", "1", "pin")
        counts = self.store.get_counts(document_id="1")
        assert counts.get("click") == 1
        assert counts.get("pin") == 1


class TestFeedbackAPI:
    """Integration tests for POST /feedback (Phase C4)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "workspace")
        os.makedirs(self.artifact_dir, exist_ok=True)

        create_text_file(os.path.join(self.artifact_dir, "notes.txt"), "searchable content about project")
        create_text_file(os.path.join(self.artifact_dir, "story.md"), "a markdown story about life")
        create_text_file(os.path.join(self.artifact_dir, "todo.txt"), "buy milk and eggs")

        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)
        # Capture the real feedback store so tests can inject a fake safely.
        self.real_feedback_store = server_app._feedback_store

    def teardown_method(self):
        if hasattr(self, "server"):
            try:
                self.server.shutdown()
            except Exception:
                pass
        server_app._feedback_store = getattr(self, "real_feedback_store", None)
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _make_request(self, method, path, body=None):
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

    def _post_feedback(self, payload):
        return self._make_request("POST", "/feedback", payload)

    def test_valid_click_returns_200(self):
        status, data = self._post_feedback({"query": "x", "document_id": "5", "action": "click"})
        assert status == 200
        assert json.loads(data)["ok"] is True

    def test_valid_ignore_returns_200(self):
        status, data = self._post_feedback({"query": "x", "document_id": "5", "action": "ignore"})
        assert status == 200
        assert json.loads(data)["ok"] is True

    def test_valid_pin_returns_200(self):
        status, data = self._post_feedback({"query": "x", "document_id": "5", "action": "pin"})
        assert status == 200
        assert json.loads(data)["ok"] is True

    def test_missing_query_returns_400(self):
        status, _ = self._post_feedback({"document_id": "5", "action": "click"})
        assert status == 400

    def test_missing_document_id_returns_400(self):
        status, _ = self._post_feedback({"query": "x", "action": "click"})
        assert status == 400

    def test_missing_action_returns_400(self):
        status, _ = self._post_feedback({"query": "x", "document_id": "5"})
        assert status == 400

    def test_invalid_action_returns_400(self):
        status, _ = self._post_feedback({"query": "x", "document_id": "5", "action": "explode"})
        assert status == 400

    def test_malformed_json_returns_400(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/feedback", "not valid json", {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        assert response.status == 400
        assert "traceback" not in data.lower()

    def test_wrong_field_types_returns_400(self):
        # query must be a string
        status, _ = self._post_feedback({"query": 123, "document_id": "5", "action": "click"})
        assert status == 400
        # action must be a string
        status, _ = self._post_feedback({"query": "x", "document_id": "5", "action": 7})
        assert status == 400

    def test_internal_failure_safe_500(self):
        server_app._feedback_store = _FakeFailingFeedbackStore()
        try:
            status, data = self._post_feedback({"query": "x", "document_id": "5", "action": "click"})
            assert status == 500
            body = json.loads(data)
            assert body["error"]["code"] == 500
            # No internal detail leakage.
            assert "SECRET_DB_PATH" not in data
            assert "traceback" not in data.lower()
            assert self.db_path not in data
        finally:
            server_app._feedback_store = self.real_feedback_store

    def test_concurrent_feedback_no_avoidable_500(self):
        errors = []
        lock = threading.Lock()

        def worker():
            status, _ = self._post_feedback({"query": "q", "document_id": "9", "action": "click"})
            with lock:
                errors.append(status)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors, "no requests were made"
        assert all(s == 200 for s in errors), errors

    def test_regression_other_endpoints_still_work(self):
        # /search
        status, data = self._make_request("POST", "/search", {"query": "searchable", "k": 10})
        assert status == 200
        assert len(json.loads(data)["results"]) >= 1
        # /status
        status, _ = self._make_request("GET", "/status")
        assert status == 200
        # /index/status (no job running)
        status, data = self._make_request("GET", "/index/status")
        assert status == 200
        assert json.loads(data)["indexing_in_progress"] is False
        # /document
        engine = server_app.get_search_engine()
        store = getattr(engine, "artifact_store", None)
        row = store.get_artifact_by_path(
            os.path.abspath(os.path.join(self.artifact_dir, "notes.txt"))
        )
        doc_id = int(row["id"])
        status, _ = self._make_request("GET", "/document/" + str(doc_id))
        assert status == 200


class TestFeedbackUISanity:
    """Phase C4 UI: front-end must drive the /feedback endpoint.

    No browser automation; inspects the served static files on disk.
    """

    def _read(self, name):
        with open(os.path.join(STATIC_DIR, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_index_html_has_feedback_status(self):
        html = self._read("index.html")
        assert 'id="feedback-status"' in html

    def test_app_js_references_feedback_endpoint(self):
        js = self._read("app.js")
        assert '"/feedback"' in js or "'/feedback'" in js

    def test_app_js_feedback_actions_represented(self):
        js = self._read("app.js")
        assert "pin" in js
        assert "ignore" in js
        assert "click" in js

    def test_app_js_feedback_includes_active_query(self):
        js = self._read("app.js")
        # The feedback request must carry the active query and document id.
        assert "query: query" in js
        assert "document_id:" in js
        assert "lastQuery" in js


# ---------------------------------------------------------------------------
# C5: Feedback re-ranking server integration tests
# ---------------------------------------------------------------------------
class _RaisingFeedbackStore:
    def get_feedback_for_documents(self, document_ids):
        raise RuntimeError('feedback store exploded')


class TestFeedbackRerankServerIntegration:
    '''C5 integration: feedback changes /search ranking without breaking it.'''

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, 'lifesearch.db')
        self.artifact_dir = os.path.join(self.temp_dir.name, 'workspace')
        os.makedirs(self.artifact_dir, exist_ok=True)
        create_text_file(os.path.join(self.artifact_dir, 'notes.txt'), 'searchable content about project alpha')
        create_text_file(os.path.join(self.artifact_dir, 'story.md'), 'a markdown story')
        create_text_file(os.path.join(self.artifact_dir, 'todo.txt'), 'buy milk')
        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)
        self.port = _find_free_port()
        self.server = LifeSearchServer(host='127.0.0.1', port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)
        self.real_feedback_store = server_app._feedback_store

    def teardown_method(self):
        if hasattr(self, 'server'):
            try:
                self.server.shutdown()
            except Exception:
                pass
        server_app._feedback_store = getattr(self, 'real_feedback_store', None)
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def _search(self, query):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            conn.request('POST', '/search', json.dumps({'query': query, 'k': 10}), {'Content-Type': 'application/json'})
            resp = conn.getresponse()
            data = resp.read().decode('utf-8')
            return resp.status, json.loads(data)
        finally:
            conn.close()

    def _record(self, doc_id, action, query):
        conn = HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            conn.request('POST', '/feedback', json.dumps({'query': query, 'document_id': str(doc_id), 'action': action}), {'Content-Type': 'application/json'})
            return conn.getresponse().status
        finally:
            conn.close()

    def _doc_id(self, filename):
        engine = server_app.get_search_engine()
        store = engine.artifact_store
        row = store.get_artifact_by_path(os.path.abspath(os.path.join(self.artifact_dir, filename)))
        return int(row['id'])

    def test_feedback_failure_does_not_500(self):
        server_app._feedback_store = _RaisingFeedbackStore()
        try:
            status, _ = self._search('searchable content about project alpha')
            assert status == 200
        finally:
            server_app._feedback_store = self.real_feedback_store

    def test_matching_feedback_boosts_score(self):
        doc_id = self._doc_id('notes.txt')
        q = 'searchable content about project alpha'
        _, data1 = self._search(q)
        base = {r['document_id']: r['score'] for r in data1['results']}[str(doc_id)]
        assert self._record(doc_id, 'pin', q) == 200
        _, data2 = self._search(q)
        after = {r['document_id']: r['score'] for r in data2['results']}[str(doc_id)]
        assert after > base

    def test_unrelated_feedback_no_change(self):
        doc_id = self._doc_id('notes.txt')
        q = 'searchable content about project alpha'
        _, data1 = self._search(q)
        base = {r['document_id']: r['score'] for r in data1['results']}[str(doc_id)]
        assert self._record(doc_id, 'pin', 'zzz unrelated query words here') == 200
        _, data2 = self._search(q)
        after = {r['document_id']: r['score'] for r in data2['results']}[str(doc_id)]
        assert abs(after - base) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


# ---------------------------------------------------------------------------
# Phase C: HTTP indexing worker + API tests
# ---------------------------------------------------------------------------
class _FakeSlowScanner:
    """Scanner whose index_folder sleeps, used to hold a job in-progress."""

    def index_folder(self, folder_path, force_reindex=False, progress_callback=None):
        time.sleep(0.5)
        if progress_callback is not None:
            progress_callback({
                "processed": 1,
                "skipped": 0,
                "errors": 0,
                "current": folder_path,
                "total": None,
            })
        return {"processed": 1, "skipped": 0, "errors": 0}


class _FakeFailingScanner:
    """Scanner whose index_folder always raises (secret-free message)."""

    def index_folder(self, folder_path, force_reindex=False, progress_callback=None):
        raise RuntimeError("SECRET_DB_PATH_/tmp/leak details should not surface")


class TestIndexingAPI:
    """Tests for POST /index and GET /index/status (Phase C)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.source_dir = os.path.join(self.temp_dir.name, "source")
        os.makedirs(self.source_dir, exist_ok=True)
        for i in range(3):
            create_text_file(
                os.path.join(self.source_dir, f"doc{i}.txt"),
                f"unique-term-{i} content about project life",
            )
        # A plain file (not a directory) for validation tests.
        self.a_file = os.path.join(self.temp_dir.name, "note.txt")
        create_text_file(self.a_file, "plain file, not a directory")

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)
        # Capture the real scanner so tests can restore it after injecting fakes.
        self.real_scanner = server_app._artifact_scanner

    def teardown_method(self):
        if hasattr(self, "server"):
            try:
                self.server.shutdown()
            except Exception:
                pass
        # Always restore the real scanner so other tests are unaffected.
        server_app._artifact_scanner = getattr(self, "real_scanner", None)
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _make_request(self, method, path, body=None):
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

    def _wait_for_index_done(self, timeout=30.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, data = self._make_request("GET", "/index/status")
            if status != 200:
                continue
            body = json.loads(data)
            if not body["indexing_in_progress"]:
                return body
            time.sleep(0.05)
        raise AssertionError("indexing did not finish within timeout")

    # 1. valid folder -> 202
    def test_post_index_valid_folder_returns_202(self):
        status, data = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        assert json.loads(data)["status"] == "scheduled"
        self._wait_for_index_done()

    # 2. malformed JSON -> 400
    def test_post_index_malformed_json_returns_400(self):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("POST", "/index", "not valid json", {"Content-Type": "application/json"})
        response = conn.getresponse()
        data = response.read().decode("utf-8")
        conn.close()
        assert response.status == 400
        assert "traceback" not in data.lower()

    # 3. missing paths -> 400
    def test_post_index_missing_paths_returns_400(self):
        status, _ = self._make_request("POST", "/index", {"reindex": True})
        assert status == 400

    # 4. relative path -> 400
    def test_post_index_relative_path_returns_400(self):
        status, _ = self._make_request("POST", "/index", {"paths": ["relative/path"]})
        assert status == 400

    # 5. nonexistent folder -> 400/404
    def test_post_index_nonexistent_folder_returns_404(self):
        status, _ = self._make_request("POST", "/index", {"paths": ["/no/such/dir/xyz123"]})
        assert status in (400, 404)

    # 6. file instead of directory -> 400
    def test_post_index_file_instead_of_directory_returns_400(self):
        status, _ = self._make_request("POST", "/index", {"paths": [self.a_file]})
        assert status == 400

    # 7. second concurrent POST -> 409
    def test_post_index_second_concurrent_returns_409(self):
        server_app._artifact_scanner = _FakeSlowScanner()
        try:
            s1, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
            assert s1 == 202
            s2, data = self._make_request("POST", "/index", {"paths": [self.source_dir]})
            assert s2 == 409
            body = json.loads(data)
            assert body["error"]["code"] == 409
        finally:
            server_app._artifact_scanner = self.real_scanner
        self._wait_for_index_done()

    # 8. status before any indexing
    def test_get_index_status_before_indexing(self):
        status, data = self._make_request("GET", "/index/status")
        assert status == 200
        body = json.loads(data)
        assert body["indexing_in_progress"] is False
        assert body["current_folder"] is None
        assert body["last_indexed_ms"] is None
        assert body["processed"] == 0
        assert body["errors"] == 0

    # 9. status during indexing
    def test_get_index_status_during_indexing(self):
        server_app._artifact_scanner = _FakeSlowScanner()
        try:
            status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
            assert status == 202
            # Query quickly, while the slow job is still running.
            st, data = self._make_request("GET", "/index/status")
            assert st == 200
            body = json.loads(data)
            assert body["indexing_in_progress"] is True
            assert body["current_folder"] == self.source_dir
        finally:
            server_app._artifact_scanner = self.real_scanner
        self._wait_for_index_done()

    # 10. status after completion
    def test_get_index_status_after_completion(self):
        status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        body = self._wait_for_index_done()
        assert body["indexing_in_progress"] is False
        assert body["current_folder"] is None

    # 11. progress / counts update
    def test_index_progress_and_counts_update(self):
        status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        body = self._wait_for_index_done()
        assert body["progress_percent"] == 100
        assert body["processed"] == 3
        assert body["skipped"] == 0
        assert body["errors"] == 0

    # 12. successful completion sets last_indexed_ms
    def test_index_success_sets_last_indexed_ms(self):
        status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        body = self._wait_for_index_done()
        assert isinstance(body["last_indexed_ms"], int)
        assert body["last_indexed_ms"] > 0
        assert body["last_error"] is None

    # 13. indexing failure is captured safely
    def test_index_failure_captured_safely(self):
        server_app._artifact_scanner = _FakeFailingScanner()
        try:
            status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
            assert status == 202
            body = self._wait_for_index_done()
            assert body["indexing_in_progress"] is False
            assert body["last_error"] is not None
            # No secret / traceback leakage.
            assert "SECRET_DB_PATH" not in body["last_error"]
            assert "traceback" not in body["last_error"].lower()
            # Failure must not update last_indexed_ms.
            assert body["last_indexed_ms"] is None
        finally:
            server_app._artifact_scanner = self.real_scanner

    # 14. server remains usable after indexing failure
    def test_server_usable_after_index_failure(self):
        server_app._artifact_scanner = _FakeFailingScanner()
        try:
            self._make_request("POST", "/index", {"paths": [self.source_dir]})
            self._wait_for_index_done()
        finally:
            server_app._artifact_scanner = self.real_scanner
        # A subsequent real job is accepted and works.
        status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        body = self._wait_for_index_done()
        assert body["indexing_in_progress"] is False
        # Status endpoint still responds.
        st, _ = self._make_request("GET", "/index/status")
        assert st == 200

    # 15. existing /search continues working after indexing
    def test_search_works_after_indexing(self):
        status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
        assert status == 202
        self._wait_for_index_done()
        st, data = self._make_request("POST", "/search", {"query": "unique-term-1", "k": 10})
        assert st == 200
        results = json.loads(data)["results"]
        assert any("doc0" in r.get("file_name", "") for r in results)

    # 16. server shutdown handles the worker correctly
    def test_shutdown_handles_worker(self):
        server_app._artifact_scanner = _FakeSlowScanner()
        try:
            status, _ = self._make_request("POST", "/index", {"paths": [self.source_dir]})
            assert status == 202
            # Shutdown must join the still-running worker instead of killing it.
            self.server.shutdown()
            assert not self.server._server_thread.is_alive()
        finally:
            server_app._artifact_scanner = self.real_scanner
        # Worker finished (joined), not forcibly killed.
        time.sleep(0.6)
        assert server_app._index_thread is not None
        assert not server_app._index_thread.is_alive()


# ---------------------------------------------------------------------------
# C7b: Structured search filters via POST /search
# ---------------------------------------------------------------------------
class TestSearchFiltersAPI:
    """POST /search with structured filters (mime_types, date_from, date_to)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "lifesearch.db")
        self.artifact_dir = os.path.join(self.temp_dir.name, "workspace")
        os.makedirs(self.artifact_dir, exist_ok=True)

        # Two text types so MIME filtering is meaningful. The query term
        # "project" appears in each FILENAME so an unfiltered search returns
        # both with a positive score (the engine drops score-0 results).
        create_text_file(
            os.path.join(self.artifact_dir, "notes project.txt"),
            "searchable content about alpha",
        )
        create_text_file(
            os.path.join(self.artifact_dir, "story project.md"),
            "a markdown story about beta",
        )

        with ArtifactStore(self.db_path) as store:
            scanner = ArtifactScanner(store, Extractor())
            scanner.index_folder(self.artifact_dir)

        self.port = _find_free_port()
        self.server = LifeSearchServer(host="127.0.0.1", port=self.port)
        self.server.start(self.db_path)
        time.sleep(0.2)

    def teardown_method(self):
        if hasattr(self, "server"):
            self.server.shutdown()
        if hasattr(self, "temp_dir"):
            self.temp_dir.cleanup()

    def _search(self, body):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", "/search", json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode("utf-8")
            return resp.status, (json.loads(data) if data else None)
        finally:
            conn.close()

    def _mimes(self, data):
        return [r["mime_type"] for r in data["results"]]

    def _doc_id(self, filename):
        engine = server_app.get_search_engine()
        store = engine.artifact_store
        row = store.get_artifact_by_path(
            os.path.abspath(os.path.join(self.artifact_dir, filename))
        )
        return int(row["id"])

    def _record(self, doc_id, action, query):
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request(
                "POST",
                "/feedback",
                json.dumps({"query": query, "document_id": str(doc_id), "action": action}),
                {"Content-Type": "application/json"},
            )
            return conn.getresponse().status
        finally:
            conn.close()

    # --- no filter / regression -------------------------------------------
    def test_no_filters_returns_both_types(self):
        status, data = self._search({"query": "project"})
        assert status == 200
        mimes = self._mimes(data)
        assert "text/plain" in mimes
        assert "text/markdown" in mimes

    # --- MIME --------------------------------------------------------------
    def test_mime_filter_text_plain(self):
        status, data = self._search(
            {"query": "project", "filters": {"mime_types": ["text/plain"]}}
        )
        assert status == 200
        assert self._mimes(data) == ["text/plain"]

    def test_mime_filter_text_markdown(self):
        status, data = self._search(
            {"query": "project", "filters": {"mime_types": ["text/markdown"]}}
        )
        assert status == 200
        assert self._mimes(data) == ["text/markdown"]

    def test_mime_filter_major_type_text(self):
        status, data = self._search(
            {"query": "project", "filters": {"mime_types": ["text"]}}
        )
        assert status == 200
        assert set(self._mimes(data)) == {"text/plain", "text/markdown"}

    # --- dates -------------------------------------------------------------
    def test_date_from_includes_recent(self):
        now = int(time.time() * 1000)
        status, data = self._search(
            {"query": "project", "filters": {"date_from": now - 3_600_000}}
        )
        assert status == 200
        assert len(data["results"]) >= 1

    def test_date_to_future_window(self):
        now = int(time.time() * 1000)
        status, data = self._search(
            {"query": "project", "filters": {"date_to": now + 3_600_000}}
        )
        assert status == 200
        assert len(data["results"]) >= 1

    def test_both_dates_window(self):
        now = int(time.time() * 1000)
        status, data = self._search(
            {
                "query": "project",
                "filters": {"date_from": now - 3_600_000, "date_to": now + 3_600_000},
            }
        )
        assert status == 200
        assert len(data["results"]) >= 1

    def test_date_out_of_range_excludes_all(self):
        future = int((time.time() + 10 * 365 * 24 * 3600) * 1000)
        status, data = self._search(
            {"query": "project", "filters": {"date_from": future}}
        )
        assert status == 200
        assert data["results"] == []

    # --- validation ---------------------------------------------------------
    def test_invalid_mime_types_not_array(self):
        status, _ = self._search(
            {"query": "project", "filters": {"mime_types": "text/plain"}}
        )
        assert status == 400

    def test_invalid_mime_types_non_string_element(self):
        status, _ = self._search(
            {"query": "project", "filters": {"mime_types": ["text/plain", 5]}}
        )
        assert status == 400

    def test_invalid_date_from_bool(self):
        status, _ = self._search(
            {"query": "project", "filters": {"date_from": True}}
        )
        assert status == 400

    def test_invalid_date_from_string(self):
        status, _ = self._search(
            {"query": "project", "filters": {"date_from": "2024-01-01"}}
        )
        assert status == 400

    def test_no_traceback_on_invalid_filter(self):
        status, data = self._search(
            {"query": "project", "filters": {"date_from": "bad"}}
        )
        assert status == 400
        assert "traceback" not in json.dumps(data).lower()

    # --- app_origin / project ignored --------------------------------------
    def test_app_origin_accepted_ignored(self):
        status, data = self._search(
            {
                "query": "project",
                "filters": {"app_origin": "browser", "mime_types": ["text/plain"]},
            }
        )
        assert status == 200
        assert self._mimes(data) == ["text/plain"]

    def test_project_accepted_ignored(self):
        status, data = self._search(
            {
                "query": "project",
                "filters": {"project": "life", "mime_types": ["text/markdown"]},
            }
        )
        assert status == 200
        assert self._mimes(data) == ["text/markdown"]

    # --- C5 reranking still works -----------------------------------------
    def test_reranking_still_works(self):
        doc_id = self._doc_id("notes project.txt")
        q = "project"
        _, data1 = self._search({"query": q})
        base = {r["document_id"]: r["score"] for r in data1["results"]}[str(doc_id)]
        assert self._record(doc_id, "pin", q) == 200
        _, data2 = self._search({"query": q})
        after = {r["document_id"]: r["score"] for r in data2["results"]}[str(doc_id)]
        assert after > base
