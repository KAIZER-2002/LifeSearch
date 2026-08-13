Roadmap — V0 → V1 → V2 → V3

Overview
This roadmap prioritizes achievable milestones for a single developer while preserving a path to advanced capabilities.

V0 — Technical prototype (2–4 weeks)
- Goals: validate event capture, enrichment, episode grouping heuristics, and retrieval pipeline on a small corpus.
- Features:
  - Filesystem watcher for selected folders
  - Event storage (append-only)
  - Simple text extraction (PDF/TXT/DOCX) and Tesseract OCR for screenshots
  - SQLite for events and artifacts, simple FTS for text
  - Heuristic episode grouping by time-window and app co-occurrence
  - Minimal CLI to query events and find artifacts by keyword
- Success criteria:
  - Can ingest a curated sample session and reconstruct an Episode that contains the download, edit, and screenshot events.
  - Query like "Qdrant PDF downloaded in May" returns the correct artifact with event evidence.
- Not included: embeddings, vector index, UX polish, cloud

V1 — Usable personal product (8–12 weeks)
- Goals: usable product for early adopters, local GUI, hybrid search.
- Features:
  - Background indexer
  - Embeddings + FAISS HNSW ANN for semantic retrieval (opt-in for users without model support)
  - Local search UI: global hotkey, result cards with "why" evidence, timeline view
  - Episode engine made robust with heuristics
  - Feedback loop for ranking (clicks/pins)
  - Packaging and signed installers for Windows/macOS
- Success criteria:
  - Private beta with 10-50 users; median time-to-find for curated tasks < 60s.
- Not included: cloud sync, team features, heavy LLM summarization

V2 — Intelligent memory system (3–6 months)
- Goals: produce high-quality Memories and strong contextual retrieval across episodes.
- Features:
  - Memory generator with polished summaries (template+optional small local LLM)
  - Better episode detection: entity co-occurrence & project inference
  - Improved ranking using relationship graph signals
  - Local model adapter, optional local reranker
  - Memory explorer UI and export/import
- Success criteria:
  - Users report being able to recall sessions ("what was I doing last Tuesday") reliably; user satisfaction improves vs V1.
- Not included: cloud training, multi-user collaboration

V3 — Advanced personal AI memory (12+ months)
- Goals: long-term differentiation, advanced personalization, optional secure cloud features for power users.
- Features:
  - On-device personalization (fine-tuning of reranker, episode segmentation) with explicit opt-in
  - Zero-knowledge sync option for encrypted index across devices
  - Offline semantic agents: proactive reminders and surfacing
  - Research pipelines for model improvements using opt-in datasets
- Success criteria:
  - High trust among users for privacy, unique retrieval capability, and decreased time-to-find vs competitors.

Notes on pace and focus
- Prioritize features that make the product demonstrably better than file search: temporal recall, screenshot recall, and episode-based queries.
- Delay cloud/training until a stable user base and explicit consent model exists.

End of ROADMAP.md