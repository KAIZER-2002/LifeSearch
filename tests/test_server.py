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
)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
