Life Search — Final Architecture Review (Solo Developer Edition)

Purpose
This document is the final V2 architecture review focused on a single highly capable developer building the product. It halts all implementation activity until the architecture is accepted. It answers the 12 requested items and clarifies what to build now, what to postpone, and which choices are reversible.

Executive summary
- Keep a modular monolith local-first architecture.
- Core shift: Events → Episodes → Memories are the first-class user-facing concepts; Artifacts are evidence.
- Start V0 as a minimal, usable, local prototype that proves episode-based recall and evidence-driven explanations.
- Avoid infrastructure complexity: embedded DB (SQLite), on-disk vector index optional, Tesseract for OCR, deterministic algorithms for parsing/dates, and a small model adapter for optional local embeddings.

1) Minimum domain model required for V0
Only include the minimal entities needed to prove the product value (capture activity and retrieve with evidence).
- Artifact (minimal fields): id, path, mime_type, created_at, modified_at, content_hash
- Event (minimal): id, type, timestamp, source, artifact_id (nullable), metadata (short), raw_payload (optional)
- Episode (minimal): id, start_ts, end_ts, event_ids (ordered), confidence
- Memory (minimal): id, summary_text (deterministic template), linked_episode_ids, evidence_refs (event ids), confidence (FACT/INFERENCE classification stored)
- Entity: lightweight token/label (string) extracted by simple rules / keyword match
- Relationship & Evidence: implicit pointers (event->artifact) and event list used as evidence (no separate Relationship table required initially)

Rationale: These allow you to capture file downloads/opens/edits/screenshots and group them into Episodes sufficiently to answer "what was I doing" queries and provide event-level evidence.

2) Minimum domain model required for V1
Add fields and objects to enable better UX and hybrid retrieval.
- Artifact: add extract_path (cached text), thumbnail_path, size
- Event: add app, confidence, linked_entities (list), tags
- Episode: add dominant_entities (list), dominant_apps, inferred_title, tags, evidence_summary (short)
- Memory: id, summary_text, linked_episode_ids, evidence_refs, confidence (template-based summaries)
- Entity: normalized entity records with type (person, tech, error_code)
- Relationship: small adjacency list support for episode→artifact and entity→artifact linking (for faster traversal)
- Indexing artifacts: FTS index for text extracts; optional embeddings for semantic lookup (adapter)

Rationale: V1 delivers better retrieval quality, explainability, and limited semantic matching while remaining local-first.

3) Long-term domain model
A richer graph that supports personalization, multi-day episodes, hierarchical memories, and model-driven inference.
- Artifact: full metadata, multiple versions (history), content fingerprints
- Event: richer types (meeting transcript, audio), user annotations, privacy flags
- Episode: hierarchical episodes (parent/child), lifecycle state (draft/confirmed), user-edits
- Memory: versioned memories (user-edited), relevance signals, personal metrics
- Entity: canonicalized entity graph (aliases, disambiguation), frequency/time series
- Relationship: typed, evidence-backed edges with weights & timestamps
- Additional stores: user preferences, ranking models, training artifacts (opt-in), encrypted sync metadata

Rationale: This model supports advanced features like proactive reminders, personalized rerankers, long-term memory consolidation, and safe sync.

4) Clear definitions (concise)
- Artifact: A persisted object on disk or accessible store used as evidence (file, image, email, bookmark). Artifacts are immutable references to content (via content_hash) and metadata. They are evidence, not the memory itself.

- Event: An immutable, normalized observation: something happened at a point in time (FileDownloaded, FileOpened, ScreenshotCaptured, EditorSave, BrowserVisited). Events link to artifacts when relevant and carry minimal metadata and optional raw_payload for provenance.

- Episode: A bounded, ordered collection of Events representing a single continuous user activity or task. Episodes have start/end timestamps, member event ids, a confidence score, and derived metadata (dominant apps/entities) for retrieval.

- Memory: A human-oriented, higher-level summary derived from Episode(s). A Memory includes a concise summary_text, links to evidence (Episode ids and Event ids), entities, and a confidence metric. Memories are optionally materialized and editable by the user.

- Entity: A named concept extracted from artifacts or events (e.g., 'Qdrant', 'MongoDB', 'JyotishAI', 'Error 500'). Entities are strings with optional normalized type and frequency counts.

- Relationship: A typed linkage between domain objects (Event→Artifact, Episode→Event, Memory→Episode, Entity→Artifact). Relationships must carry evidence (the event ids proving the link) and a confidence score.

- Evidence: A minimal verifiable pointer (Event id or Artifact id plus snippet & timestamp) that justifies an inference. Evidence items must always be presented with (a) factual confidence — whether the evidence itself is a direct FACT (for example, a download event or file system timestamp) and (b) retrieval confidence — how strongly the system ranked the result. These two scores must be kept distinct: retrieval score is about ranking quality; factual confidence is about whether the evidence supports the assertion. The UI and APIs must never conflate retrieval score with factual certainty. Classify evidence provenance as FACT (direct observable), INFERENCE (derived via rules/models), or GUESS (low-confidence hypothesis), and surface both the provenance classification and a separate retrieval score.

5) Minimum components required for V0
Only build what proves the product's claim with minimal engineering overhead.
- Capture: local filesystem watcher for selected folders (Downloads, Desktop, Documents, Screenshots) and a simple screenshot detector.
- Event Store: append-only event store implemented in SQLite (events table) that records normalized Events and supports append/query.
- Artifact Extractor: simple text extraction for PDF/DOCX/TXT and Tesseract OCR for screenshots (store cached extracts in app data folder).
- Episode Detector: heuristic-first module that groups Events into Episodes (time-window + shared artifact/entity and app continuity). Must be deterministic and explainable.
- Memory generator (V0 deterministic): create minimal Memory records from Episodes using template-based summaries and link evidence (episode ids and key event ids).
- Lexical Search: FTS (SQLite FTS5) over text extracts and event metadata for initial retrieval.
- Evidence API: trace helper that given an artifact/event returns the Episode and list of nearby Events as evidence; must always separate retrieval score from factual confidence.
- UI: provide both a minimal CLI and a tiny local web UI (single-page, static HTML + minimal backend) so the product is usable from keyboard and browser; both present evidence traces and clear FACT/INFERENCE/GUESS labels.

Why these? They let you answer the core user queries (find PDF downloaded around May; show screenshot with error; what was I doing last Tuesday) and demonstrate episode-based retrieval with provenance.

6) Components explicitly postponed until later
- Cloud sync, zero-knowledge or otherwise.
- Federated or centralized model training, telemetry pipelines, and remote inference.
- Advanced local LLM summarizers beyond template-based memory generation (may add small local models later, opt-in).
- Full-featured vector search as primary retrieval (embedding layer optional and later).
- Multi-user or shared memories, team collaboration features.
- Large-scale analytics, heavy model training infra, or microservices.

7) Exact vertical-slice development order (single-developer friendly)
Build vertical slices that each deliver a working feature end-to-end. Each slice should be runnable and testable locally.

Slice 1 — V0 vertical slice (proof of concept)
- Implement: Capture (fs watcher) -> Event Store (append-only) -> Artifact Extractor (text+OCR) -> Episode Detector (heuristic) -> Lexical search over extracts -> minimal CLI that runs a query and prints artifact + evidence.
- Acceptance: Given a simulated session (browse -> download -> edit -> screenshot), the system forms an Episode and returns the downloaded PDF with download event evidence for the query "Find the Qdrant PDF I downloaded around May."

Slice 2 — Improve UX and retrieval (move toward V1)
- Add: richer Event metadata (app, tags), cached extract thumbnails, episode metadata (dominant_entities), Memory template generator (deterministic), FTS tuning, and basic ranking heuristics (recency, access frequency).
- Acceptance: episode queries return episode summary with top evidence and list of artifacts; search CLI UI displays why.

Slice 3 — Optional semantic augment (if needed)
- Add: optional local embedding model (adapter), vector index for fuzzy matching, and hybrid ranking combining lexical + semantic signals. Keep embedding optional and disabled by default (user opt-in).
- Acceptance: queries with fuzzy phrasing ("find doc about QLoRA") return relevant artifacts even if lexical match absent.

8) What can be built independently
- Artifact Extractor and OCR: can be built and validated independently by running on sample files.
- Event Store: independent module for append/query operations and persistence.
- Episode Detector: can be developed and tested offline using sample event streams (simulate sessions).
- Lexical Search: indexing and search over cached extracts can be run independently.
- UI: the minimal CLI/web UI can be developed in parallel and consume the Search/Evidence APIs.

These modules should have small, well-defined interfaces so you can develop them in isolation and glue later.

9) What should NOT be built yet
- Any cloud components (sync, remote inference, telemetry ingestion).
- Heavy ML infrastructure: training pipelines, dataset servers, distributed compute clusters.
- Multi-user features and role-based access controls.
- Complex microservices or distributed storage systems.
- Extensive plugin architecture; start with a small connector set and add a plugin API only when needed.

10) Which architectural decisions are reversible vs difficult to change later
Reversible decisions (easy to change):
- Choice of embedding model (adapter pattern planned): easy to swap.
- Minor schema additions to SQLite: reversible with migrations.
- Using Tesseract vs better OCR later: reversible by adapter.
- UI technology (CLI -> web -> Electron): reversible, though migration costs UI work.

Difficult-to-change decisions (plan carefully):
- Event immutability and provenance model: once deployed, changing semantics (mutating events, deleting raw payload implicitly) complicates provenance and user trust.
- Primary storage choice (embedded SQLite vs custom store): SQLite is practical, but migrating to a different store with large datasets and vector indices may be harder. Still feasible with good abstraction.
- Data privacy defaults: switching from local-only to cloud-by-default breaks trust; keep opt-in if added.
- Making Episodes first-class mutable objects with user edits & merging: if you support open editing and merging early, the code complexity increases; design episode representation carefully.

11) Maximum reasonable complexity for a solo developer
- Keep the number of moving parts small: a single process (or two: UI + background worker) with 5–8 modules (capture, enrichment, storage, episode engine, search, UI, model adapter, background worker). Each module should be small (~200–1000 lines to start).
- Favor embedded, single-node components: SQLite, on-disk vector indices (optional), file-based cache.
- Limit the number of connectors initially (Files + Screenshots + optional Browser history). Each additional connector increases permissions and testing burden.
- Limit the number of model types initially (Tesseract + optional small embedding model). Full ML training and large models should be postponed.

12) Realistic V0 the solo developer can complete and use personally
V0 scope (single vertical slice):
- Features:
  - Capture: local FS watcher for Documents/Downloads/Desktop/Screenshots.
  - Event store: append-only events persisted locally (SQLite or simple file-based JSON log).
  - Artifact extractor: PDF/DOCX/TXT text extraction, Tesseract OCR for images; cached extracts.
  - Episode detector: heuristic grouping (time window + shared artifact/entity + app continuity).
  - Lexical search: SQLite FTS5 over cached extracts & event metadata.
  - Evidence trace: present Event ids and snippets that justify results; classification of evidence as FACT or INFERENCE.
  - Minimal UI: CLI or paging web UI for quick queries and evidence display.
- Success criteria:
  - From a realistic local session (browse -> download PDF -> open file -> edit -> screenshot), the system constructs an Episode and answers queries like "Find the Qdrant PDF I downloaded around May" with correct artifact in top-3 and shows download event as evidence.
  - The system runs wholly offline and respects folder selection and simple privacy controls (preview which folders are indexed).
- Time estimate (solo dev, rough): 3–8 weeks depending on familiarity with libraries (faster if prototyping in Python).

Operational recommendations (solo dev hygiene)
- Keep modules small and well-documented with clear interfaces.
- Start with prototype artifacts and a few simulated sessions to drive iteration — avoid indexing entire disk early.
- Add user-facing privacy prompts and a simple settings page early to build confidence with testers.
- Keep model adapters and vector adapters as thin layers to avoid rework later.

Final checklist before implementation
- Confirm acceptance of this architecture review.
- Decide the exact tech choices for V0 (language: Python/Go/Rust; UI: CLI/Flask/Electron; DB: SQLite; OCR: Tesseract).
- Confirm the single MVP vertical slice and acceptance tests (use the curated recall tasks in docs/ as evaluation cases).

If you accept this architecture, next (and final) step before coding is to produce a lean implementation plan for the V0 vertical slice that lists the small number of files to create, function signatures, and dev run commands; I will not produce that until you confirm acceptance of this architecture.

End of ARCHITECTURE_REVIEW.md
