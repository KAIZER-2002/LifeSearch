"""
Localhost HTTP Search Service for Life Search.

Endpoints (Phase A):
- POST /search
- GET  /status

Security posture (Phase A):
- Binds ONLY to 127.0.0.1 (never 0.0.0.0 by default).
- No external network calls.
- No arbitrary file-reading endpoints.
- No request parameter is ever interpreted as a filesystem path.
- Errors never leak tracebacks, exception messages, secrets, env vars,
  or filesystem details to the client.
"""

import json
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

from src.search.engine import SearchEngine
from src.server.mapping import (
    build_search_response,
    build_status_response,
    build_error_response,
)


# The single source of truth for the running stack. init_stack() populates it
# by reusing the EXISTING build_stack() from src.cli.main (no duplicated
# construction of ArtifactStore / VectorStore / ONNXEmbeddingEngine /
# Extractor / ArtifactScanner / EpisodeStore / MemoryStore / SearchEngine).
_search_engine: Optional[SearchEngine] = None


def init_stack(db_path: Optional[str] = None) -> None:
    """Initialize the search stack by reusing the existing build_stack().

    This intentionally does NOT construct the stores / engines directly; it
    delegates to ``build_stack`` so the server shares the exact same wiring
    as the CLI commands.
    """
    global _search_engine

    # Release any previously-initialized handles to avoid leaking file
    # descriptors between in-process server restarts (e.g. across tests).
    _close_current_stack()

    # Imported lazily to avoid a circular import at module-load time:
    # src.cli.main imports this module for command_serve.
    from src.cli.main import build_stack

    _, _, _search_engine = build_stack(db_path)


def _close_current_stack() -> None:
    engine = _search_engine
    if engine is None:
        return
    for store in (getattr(engine, "artifact_store", None),
                  getattr(engine, "vector_store", None)):
        try:
            if store is not None:
                store.close()
        except Exception:
            pass


def get_search_engine() -> SearchEngine:
    if _search_engine is None:
        raise RuntimeError("Search engine not initialized. Call init_stack() first.")
    return _search_engine


def get_status_info() -> Dict[str, Any]:
    """Build status info purely from the already-initialized stack.

    Only fields that can be determined reliably from the existing
    architecture are exposed. Fields that cannot be determined are omitted
    rather than invented.
    """
    engine = _search_engine
    if engine is None:
        return {
            "indexed_documents": 0,
            "indexed_chunks": 0,
            "model_available": False,
            "model_id": None,
        }

    artifact_store = getattr(engine, "artifact_store", None)
    vector_store = getattr(engine, "vector_store", None)
    embedding_engine = getattr(engine, "embedding_engine", None)

    indexed_documents = 0
    if artifact_store is not None:
        try:
            indexed_documents = artifact_store.artifact_count(include_missing=False)
        except Exception:
            indexed_documents = 0

    indexed_chunks = 0
    if vector_store is not None:
        try:
            indexed_chunks = vector_store.count_chunks()
        except Exception:
            indexed_chunks = 0

    model_available = False
    model_id: Optional[str] = None
    if embedding_engine is not None:
        available = getattr(embedding_engine, "_available", None)
        if available is None:
            # Fallback signal: a positive dimension implies a real engine.
            available = getattr(embedding_engine, "dimension", 0) > 0
        model_available = bool(available)
        if model_available:
            model_id = getattr(embedding_engine, "model_id", None)

    return {
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "model_available": model_available,
        "model_id": model_id,
    }


class SearchRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for search endpoints (Phase A)."""

    # ---- low-level helpers ----------------------------------------------

    def _send_json(self, status_code: int, data: Dict[str, Any]) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, status_code: int, message: str, detail: Optional[str] = None) -> None:
        # Never include raw exception text / tracebacks here.
        self._send_json(status_code, build_error_response(status_code, message, detail))

    def _read_json_body(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Read + parse the request body. Returns (data, error_message)."""
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return None, "Missing Content-Length header"
        try:
            length = int(content_length)
        except ValueError:
            return None, "Invalid Content-Length header"

        if length <= 0:
            return None, "Empty request body"
        if length > 1024 * 1024:  # 1 MiB cap
            return None, "Request body too large"

        try:
            raw = self.rfile.read(length).decode("utf-8")
        except UnicodeDecodeError:
            return None, "Request body must be UTF-8 encoded"

        if not raw.strip():
            return None, "Empty request body"

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, "Request body must be valid JSON"

        if not isinstance(data, dict):
            return None, "Request body must be a JSON object"

        return data, None

    def _validate_search_request(
        self, data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """Validate a /search request. Returns (query, k, error_message)."""
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            return None, None, "query must be a non-empty string"

        # 'k' controls the number of results.
        k = data.get("k", 10)
        if not isinstance(k, int) or isinstance(k, bool):
            try:
                k = int(k)
            except (ValueError, TypeError):
                return None, None, "k must be an integer"
        if k < 1 or k > 100:
            return None, None, "k must be between 1 and 100"

        # 'filters' is accepted for schema compatibility but not yet applied.
        filters = data.get("filters")
        if filters is not None and not isinstance(filters, dict):
            return None, None, "filters must be an object"

        return query.strip(), k, None

    # ---- routing ---------------------------------------------------------

    def do_POST(self) -> None:
        parsed = self.path.split("?", 1)[0]
        if parsed == "/search":
            self._handle_search()
        else:
            self._send_error(404, "Not Found", f"POST {parsed} not found")

    def do_GET(self) -> None:
        parsed = self.path.split("?", 1)[0]
        if parsed == "/status":
            self._handle_status()
        else:
            self._send_error(404, "Not Found", f"GET {parsed} not found")

    # ---- handlers --------------------------------------------------------

    def _handle_search(self) -> None:
        start = time.time()

        data, error = self._read_json_body()
        if error:
            self._send_error(400, "Bad Request", error)
            return

        query, k, error = self._validate_search_request(data)
        if error:
            self._send_error(400, "Bad Request", error)
            return

        try:
            engine = get_search_engine()
            results = engine.search(query, limit=k)
            took_ms = int((time.time() - start) * 1000)
            response = build_search_response(results, took_ms)
            self._send_json(200, response)
        except Exception as exc:  # SearchEngine/infra failure -> 500
            # Log server-side only; never expose internals to the client.
            self.log_error("Search failed: %s", type(exc).__name__)
            self._send_error(500, "Internal Server Error", "Search failed")

    def _handle_status(self) -> None:
        try:
            status_info = get_status_info()
            response = build_status_response(**status_info)
            self._send_json(200, response)
        except Exception:
            self._send_error(500, "Internal Server Error", "Status unavailable")

    def log_message(self, fmt: str, *args: Any) -> None:
        # Quiet by default; errors are logged via log_error instead.
        return


class LifeSearchServer:
    """Life Search HTTP server with deterministic shutdown."""

    def __init__(self, host: str = "127.0.0.1", port: int = 30013):
        if host != "127.0.0.1":
            # Phase A: localhost only. Reject any broader bind.
            raise ValueError("Phase A server is localhost-only; host must be 127.0.0.1")
        self.host = host
        self.port = port
        self.server: Optional[ThreadingHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

    def start(self, db_path: Optional[str] = None) -> None:
        """Initialize the stack and start serving on an ephemeral/real port."""
        init_stack(db_path)
        self.server = ThreadingHTTPServer((self.host, self.port), SearchRequestHandler)
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        # Brief yield so the socket is accepting before requests arrive.
        time.sleep(0.1)

    def _run_server(self) -> None:
        if self.server is not None:
            self.server.serve_forever()

    def shutdown(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)
        self._shutdown_event.set()

    def wait_for_shutdown(self) -> None:
        self._shutdown_event.wait()


def serve(host: str = "127.0.0.1", port: int = 30013, db_path: Optional[str] = None) -> None:
    """Run the Life Search HTTP server (blocking until interrupted)."""
    server = LifeSearchServer(host, port)
    try:
        server.start(db_path)
        print(f"Life Search server running on http://{host}:{port}")
        print("Endpoints: POST /search, GET /status")
        server.wait_for_shutdown()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.shutdown()
        _close_current_stack()


def command_serve(args) -> int:
    """CLI entry point for ``lifesearch serve``."""
    serve(args.host, args.port, args.db)
    return 0
