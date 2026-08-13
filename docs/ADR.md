# Life Search V2 — Architecture Decision Records (ADRs)

Below are concise ADRs for key architecture choices in Life Search V2. Each entry includes: Title, Decision, Status, Context, Consequences, and When to revisit.

---

## ADR 001 — Local-first
Decision: Make the client experience work fully offline with primary data stored locally and optional sync.
Status: Accepted
Context: Users value privacy, resilience, and low-latency access to personal data across devices.
Consequences:
- + Fast, private access; works without network.
- + Simpler privacy model for personal data.
- - Sync and multi-device consistency become optional, more complex features.
When to revisit: If product needs shift to cloud-first collaboration or real-time multi-user sync.

---

## ADR 002 — Modular monolith
Decision: Implement a single deployable app split into well-defined modules (UI, ingestion, storage, search, models) rather than microservices.
Status: Accepted
Context: Project maintained by a small team/single developer; operational overhead of services is high.
Consequences:
- + Easier development, testing, and deployment; lower ops burden.
- + Modules can be refactored into services later if needed.
- - Limits independent scaling of components.
When to revisit: If team grows or performance/scale demands require separate services.

---

## ADR 003 — Events → Episodes → Memories as primary model
Decision: Model user data as a hierarchy: raw events → grouped episodes → indexed memories for retrieval.
Status: Accepted
Context: Natural human timeline and narrative structure map well to retrieval and summarization workflows.
Consequences:
- + Intuitive UX for journaling, search, and summarization.
- + Enables multi-granularity retrieval and episodic summarization.
- - Requires robust episode-detection and retention policies.
When to revisit: If evidence shows a different primary model (e.g., graph-first) yields substantially better retrieval.

---

## ADR 004 — SQLite + FTS5 for metadata
Decision: Use embedded SQLite with FTS5 for metadata indexing and text search.
Status: Accepted
Context: Need for a lightweight, local, reliable metadata store with full-text search and wide platform support.
Consequences:
- + Local, zero-install storage with mature tooling and transactions.
- + FTS5 provides performant local full-text search for metadata and notes.
- - Complex vector search and embeddings must be stored or referenced separately.
When to revisit: If search scale or feature needs exceed SQLite performance or cross-device sync requires server-side indexing.

---

## ADR 005 — Local vector store (FAISS / hnswlib) adapter
Decision: Provide an adapter interface to local vector stores (FAISS, hnswlib) with a pluggable implementation.
Status: Accepted (implemented: hnswlib primary, SQLite exact fallback)
Context: Semantic search and retrieval rely on vector similarity; user devices vary in capability.
Consequences:
- + High-performance local semantic search; choice of backend per platform.
- + Adapter allows switching backends without changing higher-level logic.
- - Native bindings (FAISS/hnswlib) may complicate cross-platform packaging; fallback/simple JS implementations may be needed.
When to revisit: When more portable or managed vector options become preferable, or packaging issues arise.

### Implementation notes (Vector Production Hardening)
- `VectorIndexBackend` (src/vector/backends/base.py) is the minimal adapter abstraction (add_items, search, mark_deleted, save, load, rebuild_from, count, close).
- **Primary backend: hnswlib** (`src/vector/backends/hnsw.py`, cosine space, 384-dim MiniLM works, configurable M / ef_construction / ef_search).
- **SQLite exact cosine search** (`src/vector/backends/sqlite_exact.py`) is the permanent, always-available fallback and the canonical source of truth (it holds the embedding BLOBs + chunk text).
- `VectorStore` (src/vector/store.py) is a **facade** with the unchanged public contract (save_chunks / delete_artifact_chunks / search_semantic_chunks / count_chunks / close). It writes canonical data to SQLite, keeps HNSW synchronized, queries HNSW first, and **transparently falls back to exact SQLite on any ANN failure** (missing / corrupt / incompatible dimension / model mismatch / init or query error). SearchEngine and ArtifactScanner remain backend-agnostic.
- HNSW indexes are a **derived, rebuildable cache**, persisted under `~/.lifesearch/vector_index/<db-profile>/<model_id>_<dimension>.hnsw` and keyed by (model_id, dimension) so different embedding models never cross-query. SQLite remains the recovery source for rebuilds.
- FAISS was not added; the adapter leaves room for it as a future secondary backend.

---

## ADR 006 — Heuristic-first episode detection
Decision: Use deterministic, heuristic-first episode detection (time gaps, activity changes, calendar/context cues), with optional ML refinement later.
Status: Accepted
Context: Quick, explainable grouping is needed now; ML models add cost and unpredictability.
Consequences:
- + Predictable, debuggable grouping; good initial UX without heavy models.
- + Easier to tune and expose to users as settings.
- - May miss subtle patterns that ML could capture; might require later migration.
When to revisit: After collecting usage data that shows heuristics are insufficient or ML cost falls.

---

## ADR 007 — Model adapter abstraction
Decision: Introduce an abstraction layer for ML models (embedding, completion) so implementations (local, hosted) can be swapped.
Status: Accepted
Context: Users/devices may use local models, cloud providers, or a mix; need to avoid tight coupling.
Consequences:
- + Flexibility to add/remove providers and local runtimes without changing core logic.
- + Facilitates experimentation and offline-first options.
- - Adds an abstraction cost and initial integration work.
When to revisit: If a dominant provider or runtime standardizes interfaces (e.g., ONNX runtime API) making the adapter redundant.

---

## ADR 008 — Privacy default: opt-out cloud
Decision: Default to keeping user data local and disabled cloud upload; cloud features are opt-in and explicitly consented.
Status: Accepted
Context: Privacy-sensitive personal data requires conservative defaults to build trust.
Consequences:
- + Strong privacy posture and trust with users.
- + Regulatory alignment and lower legal risk.
- - Slower adoption of cloud-only collaboration features; extra UX friction for users who want cloud services.
When to revisit: If business or user needs make cloud-first defaults necessary and legal/privacy mitigations are available.

---

## ADR 009 — Single-developer constraints
Decision: Design features, infrastructure, and deployment choices to be manageable by a single developer (low ops, simple releases, minimal infra).
Status: Accepted
Context: Project resources are limited; maintainability and developer productivity are critical.
Consequences:
- + Faster iteration and lower maintenance burden.
- + Prioritizes simplicity and automation over complex, brittle architectures.
- - May limit rapid scaling or complex distributed features until team grows.
When to revisit: When team size or funding increases, or operational demands exceed a single maintainer's capacity.

---

End of ADR list.
