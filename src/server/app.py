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
import os
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional, Tuple

from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.feedback.store import FeedbackStore
from src.feedback.rerank import apply_feedback_rerank
from src.search.engine import SearchEngine
from src.server.mapping import (
    build_search_response,
    build_status_response,
    build_error_response,
)


# ---------------------------------------------------------------------------
# Minimal reader/writer lock
# ---------------------------------------------------------------------------
# Indexing mutates the shared VectorStore/HNSW + ArtifactStore state, while
# search reads it. We serialize the two at the server coordination layer with
# a reader/writer lock: many concurrent searches (readers) are allowed, but an
# indexing job (writer) is exclusive. This keeps search responsive while
# guaranteeing the HNSW index is never mutated mid-search.
class _ReadWriteLock:
    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def acquire_read(self, timeout: Optional[float] = None) -> bool:
        with self._cond:
            deadline = None if timeout is None else time.time() + timeout
            while self._writer:
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining == 0:
                    return False
                if not self._cond.wait(remaining):
                    return False
            self._readers += 1
        return True

    def release_read(self) -> None:
        with self._cond:
            self._readers = max(0, self._readers - 1)
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self, timeout: Optional[float] = None) -> bool:
        with self._cond:
            deadline = None if timeout is None else time.time() + timeout
            while self._writer or self._readers > 0:
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining == 0:
                    return False
                if not self._cond.wait(remaining):
                    return False
            self._writer = True
        return True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Indexing worker coordination (Phase C)
# ---------------------------------------------------------------------------
# Exactly ONE indexing worker thread may exist at a time. The HTTP handler
# validates input and schedules the job; the worker runs index_folder() off
# the request thread so the handler returns immediately (HTTP 202).
_index_state: Dict[str, Any] = {
    "indexing_in_progress": False,
    "progress_percent": 0,
    "processed": 0,
    "skipped": 0,
    "errors": 0,
    "current_folder": None,
    "last_indexed_ms": None,
    "last_error": None,
}
_index_thread: Optional[threading.Thread] = None
# Guards _index_state and the "is a job running?" transition. Smallest safe
# scope: protects status reads/writes and duplicate-job detection only.
_index_job_lock = threading.Lock()
# Search/index serialization (see _ReadWriteLock above).
_rw_lock = _ReadWriteLock()
# The scanner from build_stack (shares the same stores as the SearchEngine).
_artifact_scanner: Optional[ArtifactScanner] = None

# Feedback signal store (Phase C4). Isolated from the search/artifact stores;
# opens its own connection to the same database file.
_feedback_store: Optional[FeedbackStore] = None


def _close_feedback_store() -> None:
    """Close the feedback store connection cleanly (idempotent)."""
    global _feedback_store
    store = _feedback_store
    _feedback_store = None
    if store is not None:
        try:
            store.close()
        except Exception:
            pass


def _safe_index_error(exc: Exception) -> str:
    """Return a concise, client-safe error message.

    Never includes file paths, tracebacks, database paths, secrets, or file
    contents -- only the exception type, which is safe to surface.
    """
    return f"Indexing failed ({type(exc).__name__})."


def _join_index_worker(timeout: Optional[float] = None) -> None:
    """Wait for the indexing worker to finish; never kill the thread."""
    thread = _index_thread
    if thread is not None and thread.is_alive():
        thread.join(timeout)


def _reset_index_state_locked() -> None:
    _index_state.update(
        {
            "indexing_in_progress": True,
            "progress_percent": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "current_folder": None,
            "last_error": None,
        }
    )


def _reset_index_state_full() -> None:
    """Reset status to a fresh, idle session (called when the stack starts).

    Clears last_indexed_ms too, so a freshly initialized server reports a
    null last-indexed time until its first successful job.
    """
    with _index_job_lock:
        _index_state.update(
            {
                "indexing_in_progress": False,
                "progress_percent": 0,
                "processed": 0,
                "skipped": 0,
                "errors": 0,
                "current_folder": None,
                "last_indexed_ms": None,
                "last_error": None,
            }
        )


# ---------------------------------------------------------------------------
# Static UI asset serving (Phase B)
# ---------------------------------------------------------------------------
# Project-root "static/" directory, resolved relative to this module so the
# server works regardless of the current working directory.
STATIC_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "static")
)

# Whitelist: URL path -> (filename inside STATIC_DIR, Content-Type).
# The URL is mapped to a FIXED filename; it is never used to build a path,
# so directory traversal is impossible by construction.
_STATIC_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}


def _resolve_static_asset(url_path: str):
    """Return (filename, content_type) for a whitelisted asset, else None."""
    name = url_path
    if name.startswith("/static/"):
        name = name[len("/static/"):]
    elif name.startswith("/"):
        name = name[1:]
    if name in ("", "index.html"):
        name = "index.html"
    return _STATIC_ASSETS.get("/" + name)


def _static_file_path(filename: str):
    """Resolve a whitelisted filename to an absolute path inside STATIC_DIR.

    Returns None if the resolved path would escape STATIC_DIR (defense in
    depth); the whitelist already prevents this.
    """
    candidate = os.path.abspath(os.path.join(STATIC_DIR, filename))
    if candidate != STATIC_DIR and not candidate.startswith(STATIC_DIR + os.sep):
        return None
    return candidate


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
    global _search_engine, _artifact_scanner, _index_thread, _feedback_store

    # Release any previously-initialized handles to avoid leaking file
    # descriptors between in-process server restarts (e.g. across tests).
    _close_current_stack()

    # Drop any stale finished indexing worker so duplicate-job detection does
    # not mistakenly treat a dead thread as still running across restarts.
    if _index_thread is not None and not _index_thread.is_alive():
        _index_thread = None

    # Start each server session with a clean indexing status.
    _reset_index_state_full()

    # Imported lazily to avoid a circular import at module-load time:
    # src.cli.main imports this module for command_serve.
    from src.cli.main import build_stack

    # build_stack returns (store, scanner, search_engine); reuse the very same
    # scanner that shares the SearchEngine's stores so indexing populates
    # exactly what /search reads.
    _, _artifact_scanner, _search_engine = build_stack(db_path)

    # The feedback store shares the same on-disk database as the rest of the
    # stack (same resolution rule as build_stack) but owns its own connection.
    resolved = db_path or ArtifactStore.default_db_path()
    _feedback_store = FeedbackStore(resolved)


def _close_current_stack() -> None:
    global _search_engine, _artifact_scanner, _feedback_store
    _artifact_scanner = None
    _close_feedback_store()
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


def get_feedback_store() -> FeedbackStore:
    if _feedback_store is None:
        raise RuntimeError("Feedback store not initialized. Call init_stack() first.")
    return _feedback_store


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
        elif parsed == "/index":
            self._handle_index()
        elif parsed == "/feedback":
            self._handle_feedback()
        else:
            self._send_error(404, "Not Found", f"POST {parsed} not found")

    def do_GET(self) -> None:
        parsed = self.path.split("?", 1)[0]
        if parsed == "/status":
            self._handle_status()
        elif parsed == "/index/status":
            self._handle_index_status()
        elif parsed.startswith("/document/"):
            self._handle_document(parsed)
        else:
            self._handle_static(parsed)

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

        # Coordinate with indexing: a running index job holds the write lock,
        # so search waits for it to finish rather than reading a half-built
        # HNSW index. Multiple searches share the read lock concurrently.
        if not _rw_lock.acquire_read(timeout=30.0):
            self._send_error(503, "Service Unavailable", "Search temporarily unavailable")
            return

        try:
            engine = get_search_engine()
            results = engine.search(query, limit=k)
            # Optional, fail-safe personalization: feedback re-ranking never
            # breaks search (any failure falls back to the original results).
            if _feedback_store is not None:
                try:
                    results = apply_feedback_rerank(results, _feedback_store, query)
                except Exception as fb_exc:
                    self.log_error("Feedback re-ranking skipped: %s", type(fb_exc).__name__)
            took_ms = int((time.time() - start) * 1000)
            response = build_search_response(results, took_ms)
            self._send_json(200, response)
        except Exception as exc:  # SearchEngine/infra failure -> 500
            # Log server-side only; never expose internals to the client.
            self.log_error("Search failed: %s", type(exc).__name__)
            self._send_error(500, "Internal Server Error", "Search failed")
        finally:
            _rw_lock.release_read()

    def _handle_status(self) -> None:
        try:
            status_info = get_status_info()
            response = build_status_response(**status_info)
            self._send_json(200, response)
        except Exception:
            self._send_error(500, "Internal Server Error", "Status unavailable")

    # ---- indexing (Phase C) ------------------------------------------------

    def _handle_index(self) -> None:
        """POST /index: validate input and schedule an indexing job (202)."""
        data, error = self._read_json_body()
        if error:
            self._send_error(400, "Bad Request", error)
            return

        paths = data.get("paths")
        if not isinstance(paths, list) or not paths:
            self._send_error(400, "Bad Request",
                             "paths must be a non-empty JSON array")
            return

        reindex = data.get("reindex", False)
        if not isinstance(reindex, bool):
            self._send_error(400, "Bad Request", "reindex must be a boolean")
            return

        validated: list = []
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.strip():
                self._send_error(400, "Bad Request",
                                 "each path must be a non-empty string")
                return
            if not os.path.isabs(raw_path):
                self._send_error(400, "Bad Request",
                                 "path must be absolute")
                return
            abs_path = os.path.abspath(raw_path)
            if not os.path.exists(abs_path):
                self._send_error(404, "Not Found",
                                 "path does not exist")
                return
            if not os.path.isdir(abs_path):
                self._send_error(400, "Bad Request",
                                 "path is not a directory")
                return
            validated.append(abs_path)

        global _index_thread
        with _index_job_lock:
            if _index_thread is not None and _index_thread.is_alive():
                self._send_error(409, "Conflict",
                                 "an indexing job is already running")
                return
            _reset_index_state_locked()
            thread = threading.Thread(
                target=self._run_index_job,
                args=(validated, reindex),
                daemon=True,
            )
            _index_thread = thread
            thread.start()

        self._send_json(202, {"status": "scheduled"})

    def _run_index_job(self, folders: list, reindex: bool) -> None:
        """Worker: run index_folder() off the request thread, exactly one at a time."""
        scanner = _artifact_scanner
        if scanner is None:
            with _index_job_lock:
                _index_state["indexing_in_progress"] = False
                _index_state["last_error"] = "Indexing unavailable."
            return

        running = {"processed": 0, "skipped": 0, "errors": 0}
        last_error: Optional[str] = None
        total = len(folders)

        def _live_update(payload: Dict[str, Any]) -> None:
            # Live, accumulating progress. Exposes only counts, never paths
            # beyond the folder being indexed or file contents.
            with _index_job_lock:
                _index_state["processed"] = running["processed"] + payload["processed"]
                _index_state["skipped"] = running["skipped"] + payload["skipped"]
                _index_state["errors"] = running["errors"] + payload["errors"]

        for done, folder in enumerate(folders, start=1):
            with _index_job_lock:
                _index_state["current_folder"] = folder
            # Serialize against concurrent search reads of the HNSW index.
            if not _rw_lock.acquire_write(timeout=30.0):
                last_error = "Indexing timed out waiting for readers."
                break
            try:
                result = scanner.index_folder(
                    folder,
                    force_reindex=reindex,
                    progress_callback=_live_update,
                )
                running["processed"] += result.get("processed", 0)
                running["skipped"] += result.get("skipped", 0)
                running["errors"] += result.get("errors", 0)
            except Exception as exc:
                # Capture safely; never crash the HTTP server.
                last_error = _safe_index_error(exc)
            finally:
                _rw_lock.release_write()
            with _index_job_lock:
                _index_state["progress_percent"] = int((done / total) * 100) if total else 100

        with _index_job_lock:
            _index_state["processed"] = running["processed"]
            _index_state["skipped"] = running["skipped"]
            _index_state["errors"] = running["errors"]
            _index_state["current_folder"] = None
            _index_state["indexing_in_progress"] = False
            _index_state["progress_percent"] = 100
            if last_error is None:
                _index_state["last_indexed_ms"] = int(time.time() * 1000)
                _index_state["last_error"] = None
            else:
                _index_state["last_error"] = last_error

    def _handle_index_status(self) -> None:
        with _index_job_lock:
            status = dict(_index_state)
        self._send_json(200, status)

    # ---- feedback (Phase C4) -----------------------------------------------

    def _handle_feedback(self) -> None:
        """POST /feedback: capture a user ranking signal (click/ignore/pin)."""
        data, error = self._read_json_body()
        if error:
            self._send_error(400, "Bad Request", error)
            return

        query = data.get("query")
        document_id = data.get("document_id")
        action = data.get("action")

        # Type/shape validation -> 400 (never 500 for bad client input).
        if not isinstance(query, str) or not query.strip():
            self._send_error(400, "Bad Request", "query must be a non-empty string")
            return
        if not isinstance(document_id, str) or not document_id.strip():
            self._send_error(400, "Bad Request", "document_id must be a non-empty string")
            return
        if not isinstance(action, str):
            self._send_error(400, "Bad Request", "action must be a string")
            return

        try:
            store = get_feedback_store()
            store.record(query.strip(), document_id.strip(), action)
        except ValueError:
            # Invalid action value (e.g. not in click/ignore/pin).
            self._send_error(400, "Bad Request", "action must be one of: click, ignore, pin")
            return
        except Exception as exc:
            # Log server-side only; never expose internals to the client.
            self.log_error("Feedback recording failed: %s", type(exc).__name__)
            self._send_error(500, "Internal Server Error", "Feedback could not be recorded")
            return

        self._send_json(200, {"ok": True})

    def _handle_static(self, parsed_path: str) -> None:
        asset = _resolve_static_asset(parsed_path)
        if asset is None:
            self._send_error(404, "Not Found", f"GET {parsed_path} not found")
            return
        filename, content_type = asset
        file_path = _static_file_path(filename)
        if file_path is None or not os.path.isfile(file_path):
            self._send_error(404, "Not Found", "File not found")
            return
        try:
            with open(file_path, "rb") as fh:
                data = fh.read()
        except OSError:
            self._send_error(404, "Not Found", "File not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_document(self, parsed_path: str) -> None:
        prefix = "/document/"
        id_part = parsed_path[len(prefix):].split("/", 1)[0]
        if not id_part.isdigit():
            self._send_error(400, "Bad Request", "Document id must be an integer")
            return
        doc_id = int(id_part)
        try:
            engine = get_search_engine()
            store = getattr(engine, "artifact_store", None)
            if store is None:
                self._send_error(404, "Not Found", "Document not found")
                return
            row = store.get_artifact(doc_id)
            if row is None:
                self._send_error(404, "Not Found", "Document not found")
                return
            # Expose only existing, user-safe artifact fields. The id is
            # validated as an integer and used for a DB lookup only; it is
            # never interpreted as a filesystem path.
            document = {
                "id": row["id"],
                "file_name": row["file_name"],
                "path": row["path"],
                "mime_type": row["mime_type"],
                "size": row["size"],
                "created_at": row["created_at"],
                "modified_at": row["modified_at"],
                "indexed_at": row["indexed_at"],
                "content_hash": row["content_hash"],
                "extracted_text": row["extracted_text"],
                "missing": bool(row["missing"]),
            }
            self._send_json(200, {"document": document})
        except Exception:
            self.log_error("Document retrieval failed: %s", type(Exception).__name__)
            self._send_error(500, "Internal Server Error", "Unable to retrieve document")

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
        # Wait for the indexing worker to finish cleanly; we never forcibly
        # kill the thread. If it is still running, join it (bounded wait).
        _join_index_worker(timeout=30.0)
        # Close the feedback store connection cleanly on shutdown.
        _close_feedback_store()
        self._shutdown_event.set()

    def wait_for_shutdown(self) -> None:
        self._shutdown_event.wait()


def serve(host: str = "127.0.0.1", port: int = 30013, db_path: Optional[str] = None) -> None:
    """Run the Life Search HTTP server (blocking until interrupted)."""
    server = LifeSearchServer(host, port)
    try:
        server.start(db_path)
        print(f"Life Search server running on http://{host}:{port}")
        print("Endpoints: POST /search, POST /feedback, GET /status")
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
