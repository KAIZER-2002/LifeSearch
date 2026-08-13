Development Guide — Life Search (single-developer edition)

Purpose
- Provide practical guidance to build and maintain Life Search as a single developer. Include repo structure, development workflow, testing, and operational hooks.

Recommended stack (one-developer friendly)
- Language: Python for prototype/iterative work OR a split stack: TypeScript (Electron) for UI + Rust/Go for indexer if performance-critical later.
- Database: SQLite (FTS5) for metadata and event store.
- Vector index: FAISS or hnswlib (bindings available for Python); keep adapter layer.
- OCR: Tesseract.
- Embeddings: sentence-transformers (small CPU models) for MVP.

Repository structure (recommended)
- src/
  - capture/            # OS watchers and connectors
  - enrichment/         # extractors, OCR wrappers, entity extractors
  - storage/            # DB models, migrations, storage adapter
  - episode/            # episode detection engine
  - memory/             # memory generation and summarization
  - indexer/            # embedding and indexing logic (vector adapter)
  - search/             # query parsing, retrieval orchestration, reranking
  - ui/                 # frontend (Electron or local web UI)
  - cli/                # developer CLI utilities
  - tests/              # unit and integration tests
- docs/
- scripts/               # packaging, installers, model downloaders
- data/sample_corpus/    # sample files for local dev tests

Development workflow
- Use a virtualenv and pin dependencies in requirements.txt.
- Work on small vertical slices: capture -> enrichment -> store -> search.
- Write automated tests for each layer; integration tests run on sample corpus.

Testing strategy
- Unit tests: parsing, chunking heuristics, event normalization.
- Integration tests: ingest a known sample session and verify episode and memory outputs.
- Performance tests: index X files and measure time and resource consumption.
- Human tests: scripted recall tasks executed locally.

Developer tools
- lifecmd (CLI) with commands: index, status, reindex, compact-index, export-sample
- dev-mode: start with a sample corpus and an in-memory DB for fast iteration.

Packaging & release
- Build native installers: Electron-builder (if using Electron) or platform-specific packaging.
- Code signing for Windows & macOS recommended before distributing to testers.

Documentation and README
- Keep README minimal: how to run locally, where models are stored, how to enable indexing for folders.
- Add troubleshooting and model download steps.

Maintenance notes
- Keep the model adapter and vector adapter isolated; replacing backends should be small changes.
- Keep retention and deletion UI reachable; privacy concerns are primary.

End of DEVELOPMENT.md