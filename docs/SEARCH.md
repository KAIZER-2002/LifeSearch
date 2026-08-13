Search Design — Hybrid Retrieval across Events, Episodes, Memories

Goals
- Provide a single query endpoint capable of searching across files, events, episodes, memories, entities, and projects.
- Combine lexical precision, semantic recall, temporal reasoning, and relationship signals.
- Always show provenance and confidence for results.

Search components
1) Query Understanding
2) Candidate Retrieval
3) Re-ranking and scoring
4) Evidence collection and explainability

1) Query Understanding
- Parse the user query into structured signals:
  - Temporal hints ("last Tuesday", "around May")
  - Type hints (PDF, screenshot)
  - Activity hints (editing, researching, debugging)
  - Entity hints (Qdrant, MongoDB, JyotishAI)
  - Implicit references ("this" or "that screenshot") — resolved using current context or UI selection
- Use deterministic parsing and rule-based heuristics first (chrono for dates, regex for file types, simple NER). Optionally use local lightweight models for entity recognition.

2) Candidate Retrieval (hybrid)
- Lexical retrieval: SQLite FTS5 over extracted text, event metadata, and memory text.
- Semantic retrieval: ANN over embeddings for chunks, events summaries, and memories. Use embeddings for fuzzy matches (e.g., "QLoRA" matches notes mentioning that term even if named differently).
- Episode-aware retrieval: score episodes by (coverage of query entities + recency + duration + activity density) and surface episodes as first-class results.
- Event retrieval: support queries like "what did I do before this file" by locating the event and returning previous events within X minutes.

3) Re-ranking and scoring
- Combine signals into final score: S = w_lex * lex_score + w_sem * emb_score + w_temp * time_boost + w_rel * relation_score + w_access * access_boost + w_proj * project_affinity
- Defaults (tuneable): w_lex=0.35, w_sem=0.35, w_temp=0.15, w_rel=0.10, w_access=0.05
- Relation_score boosts results that are connected to high-confidence episodes or memories.
- Time boosting: apply recency when the query implies recency or when user preference is set.

4) Evidence & Explainability
- For each result return:
  - type (file/event/episode/memory)
  - main snippet or summary
  - why (list of evidence items: event id, artifact id, text snippet, OCR text)
  - confidence score with classification: FACT (direct evidence), INFERENCE (derived via heuristics/models), GUESS (low confidence)
- Provide a trace view where the user can open the chain: Query → Candidate → Episode → Event → Artifact.

Special queries
- Temporal queries ("last Tuesday"): map to local timezone; search episodes whose time window intersects that day and rank by relevance.
- "Before/after" queries: given a reference artifact or selection, find events in the time window before/after and optionally expand into the episode.
- "What was I doing" style queries: find candidate episodes overlapping the requested time and surface a Memory summary.

Partial-match & clarification
- If top results have low confidence or multiple clusters exist, ask lightweight clarifying questions ("Do you mean the PDF in Downloads or in Documents?"). Keep clarifications minimal.

Feedback loop
- Record clicks/pins/ignores to re-rank future results locally. Use small-scale on-device learning (decayed weights), not centralized training.

Performance targets (local desktop typical)
- Target: Lexical FTS query <50ms for 100k documents (SQLite WAL, tuned indexes)
- Target: ANN retrieval ~50–200ms for top-k on 100k chunks using HNSW with efSearch tuned
- Re-ranker + explainability: <100ms
- End-to-end: median <= 300ms; acceptable P99 <= 2s for complex queries (episode retrieval)

End of SEARCH.md