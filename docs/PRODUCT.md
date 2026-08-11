Life Search — V2 Product Blueprint

Summary

This V2 blueprint re-centers Life Search around a memory-first architecture: EVENTS → EPISODES → MEMORIES. It preserves core ideas from the earlier plan (semantic search, embeddings, OCR) but transforms the conceptual center to a continuous activity model built from events and episodes where files/screenshots are evidence, not the product.

Intended outcome
- A simple, local-first personal memory product that helps you find what you once did or learned, not merely where a file sits.
- A coherent, single-developer-friendly roadmap that leads to a genuinely differentiated memory layer.

Product vision (short)
- Make the user feel: "I don’t need to remember where I put anything. I can just ask what I did and why." Life Search is the private, local memory layer that reconstructs and surfaces the user’s activities and artifacts as Memories.

Product principles
- Local-first and private by default.
- Build for one developer: modular monolith, minimal runtime dependencies, clear interfaces.
- Evidence-led: every inference shows provenance and confidence.
- Heuristic-first, ML-augment later: start simple; add models where they clearly improve outcomes.
- Incremental UX: early wins that demonstrate the difference from file search.

Primary user problems
- Fragmented discovery: users remember concepts and context but not file locations.
- Context dissociation: files divorced from the sessions that produced them.
- Temporal fuzziness: users recall relative times ("last Tuesday") more frequently than exact timestamps.
- Multimodal evidence: critical content often lives in screenshots, browser history, or ephemeral notes, not searchable by filename.

What Life Search is
- A personal local service that: records normalized Events from the device, groups them into Episodes, extracts Memories, indexes evidence, and provides hybrid retrieval across files, events, episodes, memories, entities, and projects.

What Life Search is NOT
- Not a cloud-first service by default. Not a large distributed system in early phases. Not an oracle that fabricates plausible events without evidence.

Core deliverable (first proof)
- Reconstruct small activity sessions (Episode) from events, index the evidence, and answer queries like:
  - "Find the Qdrant PDF I downloaded around May." (returns PDF + download event evidence)
  - "Show the screenshot where I had the MongoDB error." (returns screenshot + OCR text evidence)
  - "What was I working on last Tuesday?" (returns Episode summary + list of artifacts)

Change log vs previous plan
- Moved from file/chunk/embedding-first to event/episode/memory-first.
- Elevated activity and relationships to first-class data rather than secondary signals.
- Reduced infrastructure complexity: single-process modular monolith instead of microservices.
- Reduced early reliance on cloud LLMs; designed for provider-independence.

What was deliberately removed
- Early cloud sync and remote LLM dependency.
- Complex microservices and distributed databases.
- Team collaboration and multi-tenant features.

What was postponed
- Cloud sync, multi-user sharing, large-scale model training, proactive cloud-based summarization.

First implementation milestone (explicit)
- Implement event capture (filesystem events), an event store implemented in SQLite, local extraction (text/OCR), heuristic episode detection, deterministic minimal Memory generation (template-based) in V0, and lexical search (SQLite FTS5). Embeddings and semantic components are optional and disabled by default.
- Provide both a minimal CLI and a tiny local web UI (single-page) for queries and evidence display so the product is usable via keyboard and browser.
- Acceptance test: From an activity trace containing browsing, download, file edits, and screenshot, Life Search should form an Episode and a V0 Memory and correctly answer representative queries (e.g., "Find the Qdrant PDF I downloaded around May") with artifact or episode results and show explicit evidence. The UI must present evidence provenance with a factual confidence label (FACT/INFERENCE/GUESS) separate from any retrieval/ranking score.

Next steps
- Read the companion docs in /docs for architecture, event and memory models, search design, privacy, AI strategy, and development roadmap.

End of PRODUCT.md
