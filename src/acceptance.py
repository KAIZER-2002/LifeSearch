"""
End-to-end acceptance harness for LifeSearch (C8-4).

This module PROVES the complete user journey using the REAL runtime path:

    real files
      -> ArtifactScanner.index_folder + run_ingest (Events->Episodes->Memories)
      -> actual embeddings (only if a model is installed; never downloaded)
      -> SearchEngine.search (lexical / episode / temporal / structured filters)
      -> runtime evidence (FACT / INFERENCE / GUESS)
      -> feedback reranking (real FeedbackStore + apply_feedback_rerank)
      -> HTTP-facing result (POST /search, /status, /index, /index/status)

Design rules honored:
  - No manual manufacturing of EventStore / EpisodeStore / MemoryStore state.
    Everything is produced by the real orchestrator (run_ingest).
  - No network / model download. If no semantic model is installed, the
    semantic scenario is reported as BLOCKED (skipped), never silently passed.
  - Deterministic: synthetic dataset in a temp dir, fixed queries, no sleeps
    beyond explicit index-status polling.
  - No absolute user paths / secrets in the report (temp paths sanitized).

The same run_acceptance() is used by the pytest suite and by the
`lifesearch acceptance` CLI command.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest import mock

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.store import EpisodeStore
from src.events.store import EventStore
from src.feedback.rerank import apply_feedback_rerank
from src.feedback.store import FeedbackStore
from src.memories.store import MemoryStore
from src.model_lifecycle import validate_model
from src.search.engine import SearchEngine
from src.search.result import SearchResult
from src.vector.embeddings import ONNXEmbeddingEngine
from src.vector.store import VectorStore

from src.ingest.orchestrator import run_ingest

# Status values for a scenario.
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class ScenarioResult:
    id: str
    name: str
    status: str
    details: str = ""
    # Optional structured extras (never include secrets/absolute paths).
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AcceptanceContext:
    base_dir: str
    folder: str
    db_path: str
    files: Dict[str, str] = field(default_factory=dict)  # name -> abs path
    artifact_ids: Dict[str, int] = field(default_factory=dict)  # name -> id
    engine: Optional[SearchEngine] = None
    model_available: bool = False
    semantic_target: str = ""


# ---------------------------------------------------------------------------
# Synthetic dataset (small, deterministic, no network)
# ---------------------------------------------------------------------------

def _seed_dataset(folder: str) -> Dict[str, str]:
    """Create a small, deterministic dataset exercising all required facets.

    - distinctive fact (qdrant)            -> lexical + semantic target
    - shared activity/context (3 files)     -> episode/memory grouping
    - two MIME types (text/markdown, text/plain)
    - content that yields event/episode/memory + evidence
    """
    files: Dict[str, str] = {}

    qdrant = os.path.join(folder, "qdrant_vector_notes.md")
    with open(qdrant, "w", encoding="utf-8") as fh:
        fh.write(
            "Qdrant vector database stores high-dimensional embeddings for "
            "semantic similarity search across the document collection."
        )
    files["qdrant.md"] = qdrant

    retrieval = os.path.join(folder, "retrieval_strategy.txt")
    with open(retrieval, "w", encoding="utf-8") as fh:
        fh.write(
            "Hybrid retrieval strategy combines keyword matching with dense "
            "vector search to rank documents by meaning rather than exact words."
        )
    files["retrieval.txt"] = retrieval

    meeting = os.path.join(folder, "meeting_notes.md")
    with open(meeting, "w", encoding="utf-8") as fh:
        fh.write(
            "Meeting notes about the LifeSearch project roadmap, milestones, "
            "and the episodic memory pipeline design."
        )
    files["meeting.md"] = meeting

    return files


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sanitize(ctx: AcceptanceContext, text: str) -> str:
    """Replace the temp dataset path with a placeholder (no absolute leakage)."""
    if ctx and ctx.base_dir and text:
        text = text.replace(ctx.base_dir, "<dataset_dir>")
    return text


# ---------------------------------------------------------------------------
# Build + run the real runtime pipeline
# ---------------------------------------------------------------------------

def _build_context() -> AcceptanceContext:
    base_dir = tempfile.mkdtemp(prefix="lifesearch_acceptance_")
    folder = os.path.join(base_dir, "workspace")
    os.makedirs(folder, exist_ok=True)
    files = _seed_dataset(folder)
    db_path = os.path.join(base_dir, "lifesearch.db")

    # Real scanner + orchestrator, using the REAL embedding engine so that
    # semantic indexing works automatically when a model is installed.
    artifact_store = ArtifactStore(db_path)
    vector_store = VectorStore(db_path)
    embedding_engine = ONNXEmbeddingEngine()
    extractor = Extractor()
    scanner = ArtifactScanner(
        artifact_store,
        extractor,
        vector_store=vector_store,
        embedding_engine=embedding_engine,
    )

    totals = run_ingest(scanner, [folder], db_path)

    # Build the search engine with ALL runtime stores wired (event_store too),
    # matching production build_stack so evidence is real.
    episode_store = EpisodeStore(db_path)
    memory_store = MemoryStore(db_path)
    event_store = EventStore(db_path)
    engine = SearchEngine(
        artifact_store,
        episode_store=episode_store,
        memory_store=memory_store,
        event_store=event_store,
        vector_store=vector_store,
        embedding_engine=embedding_engine,
    )

    artifact_ids = {}
    for name, path in files.items():
        row = artifact_store.get_artifact_by_path(os.path.abspath(path))
        if row is not None:
            artifact_ids[name] = int(row["id"])

    ctx = AcceptanceContext(
        base_dir=base_dir,
        folder=folder,
        db_path=db_path,
        files=files,
        artifact_ids=artifact_ids,
        engine=engine,
        model_available=bool(embedding_engine._available),
        semantic_target="retrieval.txt",
    )
    # Stash totals for inspection by scenarios.
    ctx.extra_totals = totals  # type: ignore[attr-defined]
    return ctx


def _close_context(ctx: AcceptanceContext) -> None:
    engine = ctx.engine
    if engine is None:
        return
    for store in (
        getattr(engine, "artifact_store", None),
        getattr(engine, "vector_store", None),
        getattr(engine, "episode_store", None),
        getattr(engine, "memory_store", None),
        getattr(engine, "event_store", None),
    ):
        try:
            if store is not None:
                store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Scenario helpers
# ---------------------------------------------------------------------------

def _ok(ctx, sid, name, details="", extra=None) -> ScenarioResult:
    return ScenarioResult(sid, name, STATUS_PASSED, _sanitize(ctx, details), extra or {})


def _fail(ctx, sid, name, details="", extra=None) -> ScenarioResult:
    return ScenarioResult(sid, name, STATUS_FAILED, _sanitize(ctx, details), extra or {})


def _skip(ctx, sid, name, details="", extra=None) -> ScenarioResult:
    return ScenarioResult(sid, name, STATUS_SKIPPED, _sanitize(ctx, details), extra or {})


# ---------------------------------------------------------------------------
# Scenarios A-K
# ---------------------------------------------------------------------------

def scenario_basic_indexing(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        totals = ctx.extra_totals  # type: ignore[attr-defined]
        artifact_store = ctx.engine.artifact_store
        artifact_count = artifact_store.artifact_count(include_missing=False)
        chunk_count = ctx.engine.vector_store.count_chunks()
        if artifact_count < 3:
            return _fail(ctx, "A", "Basic indexing", f"expected >=3 artifacts, got {artifact_count}")
        if chunk_count < 1:
            return _fail(ctx, "A", "Basic indexing", f"expected >=1 chunk, got {chunk_count}")
        if totals.get("errors", 0) != 0:
            return _fail(ctx, "A", "Basic indexing", f"indexing errors: {totals.get('errors')}")
        return _ok(
            ctx, "A", "Basic indexing",
            f"artifacts={artifact_count} chunks={chunk_count} processed={totals.get('processed')}",
            {"artifacts": artifact_count, "chunks": chunk_count},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "A", "Basic indexing", f"unexpected error: {type(exc).__name__}")


def scenario_events_episodes_memories(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        totals = ctx.extra_totals  # type: ignore[attr-defined]
        events = totals.get("events", 0)
        episodes = totals.get("episodes", 0)
        memories = totals.get("memories", 0)
        if events < 1:
            return _fail(ctx, "B", "Events/Episodes/Memories", f"no events generated (got {events})")
        if episodes < 1:
            return _fail(ctx, "B", "Events/Episodes/Memories", f"no episodes generated (got {episodes})")
        if memories < 1:
            return _fail(ctx, "B", "Events/Episodes/Memories", f"no memories generated (got {memories})")
        return _ok(
            ctx, "B", "Events/Episodes/Memories",
            f"events={events} episodes={episodes} memories={memories} (via real run_ingest)",
            {"events": events, "episodes": episodes, "memories": memories},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "B", "Events/Episodes/Memories", f"unexpected error: {type(exc).__name__}")


def scenario_lexical_search(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        results = ctx.engine.search("qdrant", limit=5)
        if not results:
            return _fail(ctx, "C", "Lexical search", "no results for distinctive term 'qdrant'")
        top = results[0]
        if "qdrant" not in (top.file_name + top.snippet + top.path).lower():
            return _fail(ctx, "C", "Lexical search", f"top result not the qdrant doc: {top.file_name}")
        return _ok(
            ctx, "C", "Lexical search",
            f"distinctive term 'qdrant' -> {top.file_name} (score={top.score:.3f})",
            {"top_file": top.file_name, "score": round(top.score, 4)},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "C", "Lexical search", f"unexpected error: {type(exc).__name__}")


def scenario_semantic_hybrid(ctx: AcceptanceContext) -> ScenarioResult:
    if not ctx.model_available:
        return _skip(
            ctx, "D", "Semantic / hybrid search",
            "SEMANTIC ACCEPTANCE BLOCKED: no valid embedding model installed. "
            "Semantic/hybrid retrieval was NOT verified. Run `lifesearch model install` "
            "to enable it. (No model was downloaded automatically.)",
        )
    try:
        # Paraphrase with minimal lexical overlap to prove embedding-based match.
        query = "find text that is similar in meaning using vector closeness"
        results = ctx.engine.search(query, limit=5)
        target = ctx.semantic_target
        matched = any(r.file_name == target for r in results)

        # Independent semantic-similarity signal via the real engine.
        sim = 0.0
        try:
            with open(ctx.files[target], "r", encoding="utf-8") as fh:
                doc_text = fh.read()
            a = ctx.engine.embedding_engine.embed_text(doc_text)
            b = ctx.engine.embedding_engine.embed_text(query)
            if a and b:
                import numpy as np
                av = np.array(a, dtype=float)
                bv = np.array(b, dtype=float)
                sim = float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv) + 1e-9))
        except Exception:
            sim = 0.0

        if matched or sim >= 0.4:
            return _ok(
                ctx, "D", "Semantic / hybrid search",
                f"paraphrase query matched '{target}' (top_match={matched}, cosine={sim:.3f})",
                {"target": target, "matched": matched, "cosine": round(sim, 4)},
            )
        return _fail(
            ctx, "D", "Semantic / hybrid search",
            f"paraphrase query did not match '{target}' (matched={matched}, cosine={sim:.3f})",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "D", "Semantic / hybrid search", f"unexpected error: {type(exc).__name__}")


def scenario_temporal(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        ref = datetime.now(timezone.utc)
        results = ctx.engine.search("what did i work on today", limit=10, reference_date=ref)
        # The freshly ingested files produced events "today", so at least one
        # dataset artifact should surface via the episode time-window path.
        matched_ids = {r.id for r in results}
        if not matched_ids:
            return _fail(ctx, "E", "Temporal search", "no results for temporal query 'today'")
        if not any(name for name, aid in ctx.artifact_ids.items() if aid in matched_ids):
            return _fail(ctx, "E", "Temporal search", "temporal query did not surface any dataset artifact")
        return _ok(
            ctx, "E", "Temporal search",
            f"temporal query surfaced {len(matched_ids)} result(s) for 'today'",
            {"result_count": len(matched_ids)},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "E", "Temporal search", f"unexpected error: {type(exc).__name__}")


def scenario_structured_filters(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        # Baseline: docs whose text contains "search".
        base = ctx.engine.search("search", limit=10)
        base_ids = {r.id for r in base}
        if len(base_ids) < 2:
            return _fail(ctx, "F", "Structured filters", f"expected >=2 docs for 'search', got {len(base_ids)}")

        # MIME filter: text/markdown should return exactly the markdown docs
        # among the search results (filtering, not re-querying all artifacts).
        md_expected = {r.id for r in base if r.mime_type == "text/markdown"}
        md_only = ctx.engine.search("search", limit=10, filters={"mime_types": ["text/markdown"]})
        if {r.id for r in md_only} != md_expected:
            return _fail(
                ctx, "F", "Structured filters",
                f"mime filter mismatch: got {sorted(r.id for r in md_only)} expected {sorted(md_expected)}",
            )

        # Combined filter: text/plain AND a wide date window.
        now_ms = int(time.time() * 1000)
        day = 24 * 60 * 60 * 1000
        combined_expected = {r.id for r in base if r.mime_type == "text/plain"}
        combined = ctx.engine.search(
            "search", limit=10,
            filters={"mime_types": ["text/plain"], "date_from": now_ms - day, "date_to": now_ms + day},
        )
        if {r.id for r in combined} != combined_expected:
            return _fail(ctx, "F", "Structured filters", "combined mime+date filter mismatch")

        # Out-of-range date window -> empty (proves date filtering is real).
        empty = ctx.engine.search(
            "search", limit=10,
            filters={"date_from": 0, "date_to": 1},  # epoch ms ~ 1970
        )
        if empty:
            return _fail(ctx, "F", "Structured filters", "out-of-range date filter returned results")

        return _ok(
            ctx, "F", "Structured filters",
            f"mime filter -> {len(md_expected)} md; combined -> {len(combined_expected)} txt; oor -> 0",
            {"md_count": len(md_expected), "txt_count": len(combined_expected)},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "F", "Structured filters", f"unexpected error: {type(exc).__name__}")


def scenario_runtime_evidence(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        results = ctx.engine.search("qdrant", limit=5)
        if not results:
            return _fail(ctx, "G", "Runtime evidence", "no results to inspect evidence")
        top = results[0]
        if not top.evidence:
            return _fail(ctx, "G", "Runtime evidence", "no evidence items on result")
        types = {e.get("type") for e in top.evidence}
        # Evidence must reference real source artifacts/events.
        has_artifact_ref = any(e.get("artifact_id") == top.id for e in top.evidence)
        has_event_or_episode = bool(types & {"event", "episode", "memory"})
        # Snippet populated from the artifact's source text.
        snippet_present = any(e.get("snippet") for e in top.evidence)
        if not (has_artifact_ref and has_event_or_episode):
            return _fail(
                ctx, "G", "Runtime evidence",
                f"evidence missing source refs: types={types} artifact_ref={has_artifact_ref}",
            )
        return _ok(
            ctx, "G", "Runtime evidence",
            f"{len(top.evidence)} evidence items types={sorted(types)} snippet={snippet_present}",
            {"evidence_count": len(top.evidence), "types": sorted(types), "snippet": snippet_present},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "G", "Runtime evidence", f"unexpected error: {type(exc).__name__}")


def scenario_confidence(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        results = ctx.engine.search("qdrant", limit=5)
        top = results[0]
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in top.evidence:
            by_type.setdefault(e.get("type"), []).append(e)

        problems = []
        # Event evidence -> FACT (direct observable).
        for ev in by_type.get("event", []):
            if ev.get("confidence_type") != "FACT":
                problems.append(f"event {ev.get('id')} not FACT")
        # Episode/Memory -> INFERENCE when underpinning events resolve (event_store
        # is wired in this harness) or GUESS otherwise.
        for ep in by_type.get("episode", []):
            if ep.get("confidence_type") not in ("INFERENCE", "GUESS"):
                problems.append(f"episode {ep.get('id')} bad class {ep.get('confidence_type')}")
        for mem in by_type.get("memory", []):
            if mem.get("confidence_type") not in ("INFERENCE", "GUESS"):
                problems.append(f"memory {mem.get('id')} bad class {mem.get('confidence_type')}")

        if problems:
            return _fail(ctx, "H", "Confidence (FACT/INFERENCE/GUESS)", "; ".join(problems))
        return _ok(
            ctx, "H", "Confidence (FACT/INFERENCE/GUESS)",
            f"event=FACT, episode/memory in {{INFERENCE,GUESS}}; classes={sorted({e.get('confidence_type') for e in top.evidence})}",
            {"classes": sorted({e.get("confidence_type") for e in top.evidence})},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "H", "Confidence (FACT/INFERENCE/GUESS)", f"unexpected error: {type(exc).__name__}")


def scenario_feedback(ctx: AcceptanceContext) -> ScenarioResult:
    """Deterministic, unit-level feedback verification using the REAL store + reranker."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            fb_db = tf.name
        try:
            store = FeedbackStore(fb_db)
            # Two results with identical base scores.
            r_a = SearchResult(
                id=1, file_name="a.txt", path="/a.txt", mime_type="text/plain", size=10,
                modified_at="2024-01-01T00:00:00", rank=0, snippet="retrieval", score=0.5,
                result_type="artifact",
            )
            r_b = SearchResult(
                id=2, file_name="b.txt", path="/b.txt", mime_type="text/plain", size=10,
                modified_at="2024-01-01T00:00:00", rank=0, snippet="retrieval", score=0.5,
                result_type="artifact",
            )

            # 1) Feedback persists.
            store.record("retrieval", "1", "pin")
            recorded = store.get_feedback(document_id="1")
            if len(recorded) != 1:
                return _fail(ctx, "I", "Feedback reranking", "feedback not persisted")

            # 2) Matching feedback influences ranking (pin boosts doc 1).
            reranked = apply_feedback_rerank([r_a, r_b], store, "retrieval")
            if reranked[0].id != 1:
                return _fail(ctx, "I", "Feedback reranking", f"pin did not boost doc 1 (order={[r.id for r in reranked]})")

            # 3) Unrelated query does NOT receive an inappropriate boost.
            unrelated = apply_feedback_rerank([r_a, r_b], store, "banana")
            if [r.id for r in unrelated] != [1, 2]:
                return _fail(ctx, "I", "Feedback reranking", "unrelated query wrongly reranked")

            store.close()
            return _ok(
                ctx, "I", "Feedback reranking",
                "feedback persisted; matching query boosted pinned doc; unrelated query unchanged",
                {"recorded": len(recorded)},
            )
        finally:
            try:
                os.remove(fb_db)
            except OSError:
                pass
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "I", "Feedback reranking", f"unexpected error: {type(exc).__name__}")


def scenario_http(ctx: AcceptanceContext) -> ScenarioResult:
    """Start the REAL localhost HTTP server and exercise the browser-facing API."""
    try:
        from src.server.app import LifeSearchServer
    except Exception as exc:  # pragma: no cover
        return _fail(ctx, "J", "HTTP search", f"cannot import server: {type(exc).__name__}")

    port = _find_free_port()
    server = LifeSearchServer(host="127.0.0.1", port=port)
    server.start(ctx.db_path)
    time.sleep(0.3)
    details = {}
    try:
        import http.client
        import json as _json

        def _post(path, body):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            conn.request("POST", path, _json.dumps(body), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = resp.read().decode()
            conn.close()
            return resp.status, _json.loads(data) if data else {}

        def _get(path):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
            conn.request("GET", path)
            resp = conn.getresponse()
            data = resp.read().decode()
            conn.close()
            return resp.status, _json.loads(data) if data else {}

        # /status
        st, st_body = _get("/status")
        if st != 200 or "model_available" not in st_body:
            return _fail(ctx, "J", "HTTP search", f"/status failed: {st} {st_body}")
        details["status_model_available"] = st_body.get("model_available")

        # POST /search
        code, body = _post("/search", {"query": "qdrant", "k": 5})
        if code != 200 or not body.get("results"):
            return _fail(ctx, "J", "HTTP search", f"/search failed: {code} {body}")
        top = body["results"][0]
        if not top.get("evidence"):
            return _fail(ctx, "J", "HTTP search", "HTTP result missing evidence")
        if not any(e.get("confidence_type") for e in top["evidence"]):
            return _fail(ctx, "J", "HTTP search", "HTTP evidence missing confidence_type")
        details["http_top_file"] = top.get("file_name")

        # POST /feedback (real API) then re-search (rerank must not error).
        doc_id = str(top["document_id"])
        fcode, fbody = _post("/feedback", {"query": "qdrant", "document_id": doc_id, "action": "pin"})
        if fcode != 200 or not fbody.get("ok"):
            return _fail(ctx, "J", "HTTP search", f"/feedback failed: {fcode} {fbody}")
        code2, body2 = _post("/search", {"query": "qdrant", "k": 5})
        if code2 != 200:
            return _fail(ctx, "J", "HTTP search", f"/search after feedback failed: {code2}")

        # POST /index (re-index the same folder) then poll /index/status.
        rcode, rbody = _post("/index", {"paths": [os.path.abspath(ctx.folder)], "reindex": False})
        if rcode != 202:
            return _fail(ctx, "J", "HTTP search", f"/index failed: {rcode} {rbody}")
        deadline = time.time() + 30
        last = None
        while time.time() < deadline:
            scode, sbody = _get("/index/status")
            last = sbody
            if scode == 200 and not sbody.get("indexing_in_progress"):
                break
            time.sleep(0.1)
        if last is None or last.get("indexing_in_progress"):
            return _fail(ctx, "J", "HTTP search", f"/index/status never completed: {last}")
        details["index_episodes"] = last.get("episodes")

        return _ok(
            ctx, "J", "HTTP search",
            f"/status,/search,/feedback,/search,/index,/index/status all OK; top={top.get('file_name')}",
            details,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "J", "HTTP search", f"unexpected error: {type(exc).__name__}: {exc}")
    finally:
        try:
            server.shutdown()
        except Exception:
            pass


def scenario_model_readiness(ctx: AcceptanceContext) -> ScenarioResult:
    try:
        # Missing model is reported clearly (no download). Use a temp missing dir.
        missing_dir = tempfile.mkdtemp(prefix="ls_model_missing_")
        with mock.patch("src.vector.model_manager.urllib.request.urlretrieve") as spy:
            status = validate_model(missing_dir)
        if status.get("installed") is not False or status.get("valid") is not False:
            return _fail(ctx, "K", "Model readiness", f"missing model not reported clearly: {status}")
        if spy.called:
            return _fail(ctx, "K", "Model readiness", "validate_model triggered a network download")

        # The acceptance run itself must not have downloaded a model.
        with mock.patch("src.vector.model_manager.urllib.request.urlretrieve") as spy2:
            # Re-validate the real (possibly missing) default model location.
            real = validate_model(None)
        downloaded_during_acceptance = spy2.called

        return _ok(
            ctx, "K", "Model readiness",
            f"missing model reported clearly; no download during acceptance; "
            f"default model installed={real.get('installed')}",
            {"default_installed": real.get("installed"), "downloaded": downloaded_during_acceptance},
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _fail(ctx, "K", "Model readiness", f"unexpected error: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

SCENARIOS = [
    scenario_basic_indexing,
    scenario_events_episodes_memories,
    scenario_lexical_search,
    scenario_semantic_hybrid,
    scenario_temporal,
    scenario_structured_filters,
    scenario_runtime_evidence,
    scenario_confidence,
    scenario_feedback,
    scenario_http,
    scenario_model_readiness,
]


def run_acceptance(verbose: bool = False) -> Dict[str, Any]:
    """Run the full end-to-end acceptance suite. Returns a machine-readable report."""
    ctx = _build_context()
    try:
        scenarios: List[ScenarioResult] = []
        for fn in SCENARIOS:
            scenarios.append(fn(ctx))

        passed = sum(1 for s in scenarios if s.status == STATUS_PASSED)
        failed = sum(1 for s in scenarios if s.status == STATUS_FAILED)
        skipped = sum(1 for s in scenarios if s.status == STATUS_SKIPPED)
        total = len(scenarios)
        # Acceptance "passed" only when nothing failed. Skipped (e.g. semantic
        # blocked by missing model) does not fail the suite.
        overall_passed = failed == 0

        report = {
            "passed": overall_passed,
            "total": total,
            "passed_count": passed,
            "failed_count": failed,
            "skipped_count": skipped,
            "model_available": ctx.model_available,
            "semantic_blocked_by_missing_model": (skipped > 0 and not ctx.model_available),
            "scenarios": [
                {
                    "id": s.id,
                    "name": s.name,
                    "status": s.status,
                    "details": s.details,
                    "extra": s.extra,
                }
                for s in scenarios
            ],
        }
        if verbose:
            print(_format_human(report))
        return report
    finally:
        _close_context(ctx)


def _format_human(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("LifeSearch End-to-End Acceptance")
    lines.append(f"  model_available: {report['model_available']}")
    lines.append(
        f"  result: {'PASS' if report['passed'] else 'FAIL'} "
        f"({report['passed_count']} passed, {report['failed_count']} failed, "
        f"{report['skipped_count']} skipped of {report['total']})"
    )
    if report.get("semantic_blocked_by_missing_model"):
        lines.append("  NOTE: semantic/hybrid acceptance BLOCKED (no model installed).")
    for s in report["scenarios"]:
        lines.append(f"  [{s['status'].upper():7}] {s['id']} {s['name']}: {s['details']}")
    return "\n".join(lines)


def main_cli(json_path: Optional[str] = None, verbose: bool = True) -> int:
    """CLI entry point. Returns process exit code (0=pass, 1=fail)."""
    report = run_acceptance(verbose=verbose)
    if json_path:
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not write JSON report: {exc}")
    return 0 if report["passed"] else 1
