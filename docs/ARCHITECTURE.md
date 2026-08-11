Life Search — V2 Architecture

Overview

The architecture is intentionally a modular monolith that runs locally on the user's machine. It consists of the following high-level components:

- Capture layer: OS watchers and lightweight app connectors that create normalized Events.
- Enrichment layer: extractors (PDF, DOCX, text), OCR, entity extraction, lightweight NER and topic tags.
- Storage layer: embedded databases for events, episodes, memories, metadata, and vector indexes for embeddings.
- Episode & Memory engine: heuristic-first episode detection, memory generation, relationship graph builder.
- Search & Query layer: hybrid retrieval across events/episodes/memories and files, re-ranking and explainability.
- UI layer: global search UI, timeline, memory explorer, and minimal settings.
- Background worker: indexing, enrichment, model inference, and maintenance tasks.

High-level architecture diagram (text)

  [Capture] -> [Enrichment] -> [Storage]
       |                           |
       v                           v
   Events DB  <-> Episode Engine <-> Memory Store
       |                           |
       v                           v
   Search API <-- Re-ranker / Hybrid Retrieval <-- Vector Store
       |
       v
     UI (global hotkey / tray)

Design constraints
- Single-developer operability: keep the codebase small, prefer high-productivity languages for prototypes (Python/TypeScript) with later rewrite options.
- Local-first: embedded SQLite / LMDB and a local vector index (FAISS, Annoy, or HNSW-based minimal implementation).
- Minimal external dependencies: Tesseract for OCR, sentence-transformers for embeddings (optional CPU models), no mandatory remote APIs.

Data flows
- Capture layer captures raw system signals and normalizes them into Event objects.
- Enrichment augments Events (text extract, OCR, entities) and writes enriched Event records to Events DB.
- Episode Engine consumes Events (live or batch) and creates Episode objects based on heuristics.
- Memory Generator summarizes Episodes into Memories when appropriate (manually or on schedule).
- Indexer produces indexes for search: lexical FTS (SQLite FTS5) and vector index for embeddings.
- Search API answers queries by combining lexical matches, vector matches, temporal filters, and episode/memory relevance signals.

Components (implementation notes)

1) Capture Layer
- Filesystem watcher: watches user-selected folders (Documents, Downloads, Desktop, Screenshots). Emits FileCreated, FileModified, FileDeleted, FileOpened events.
- Screenshot detector: watch screenshot folders or listen for OS screenshot events where available.
- Application usage listener: lightweight polling or OS-provided app events (app opened, foreground change).
- Browser connector (optional): local browser history ingestion (reading user profile) for Chrome/Edge/Firefox on user consent.
- Git watcher (optional): watch repos for commits.

2) Enrichment Layer
- Text extraction: PDFs, DOCX, Markdown using robust existing libraries.
- OCR: Tesseract tuned for screenshots.
- Entity extraction: small local NER model or rule-based heuristics (regex for error messages, code tokens, email addresses).
- Topic tags: simple keyword mapping and TF-IDF / LSA on document text; later replaced by topic model.

3) Storage
- Events DB: append-only store with indexing for time, source, and linked artifact id. (SQLite recommended)
- Episodes store: Episode records with references to member event ids.
- Memories store: human-readable synthesized memories with pointers and confidence.
- Artifact store: metadata for files/screenshots (path, hash, mime, size, first_seen, last_seen).
- Vector store: local FAISS or an embeddable HNSW index for chunk/document embeddings.

4) Episode & Memory engine
- Runs heuristics that group events into episodes; stores episode metadata (start, end, dominant entities, apps, confidence).
- Memory generator produces short natural-language summaries (optionally via a small local LLM or deterministic template).

5) Search & Query layer
- Exposes a local HTTP/IPC API for the UI.
- Implements hybrid retrieval orchestration: lexical FTS + ANN + episode scoring + temporal reasoning.
- Re-ranker combines multiple signals and returns results with provenance and confidence.

6) UI
- Global hotkey to open a small search window.
- Result cards with artifact preview, episode links, and "Why" explanation showing evidence and confidence.
- Memory explorer and timeline view (minimal for MVP).

Architecture Decisions (ADR summary)
- ADR 001: Local-first, modular monolith (reason: single-developer scope, privacy requirement).
- ADR 002: Events→Episodes→Memories are primary abstractions.
- ADR 003: SQLite + embedded vector index for storage (reason: embeddable, low ops overhead).
- ADR 004: Heuristic-first episode detection with ML migration path.
- ADR 005: Provider-independent AI layer — abstract model inference behind an adapter.

What can be extracted later
- If the product needs to scale or move to cloud, the modular monolith can break into services: Capture Service, Index Service, Episode Service, Search API.

End of ARCHITECTURE.md
