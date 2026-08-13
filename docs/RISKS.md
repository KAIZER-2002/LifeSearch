Risks and Mitigations — Life Search V2

Overview
A concise list of primary risks for Life Search V2. Each entry includes a short description, impact, likelihood, and a mitigation that one developer can act on.

1) Technical risk: Indexing cost and resource usage
- Description: OCR, embeddings, and indexing can overwhelm CPU, memory, and disk I/O on typical user machines.
- Impact: High — slow or failed indexing causes poor UX and churn.
- Likelihood: High
- Mitigation (one-dev tasks): Implement an incremental indexer that processes files in small batches; add a user-configurable CPU cap and AC-power-only toggle; add progress + retry logic and telemetry (index time, failures).

2) Product risk: Low precision / low recall
- Description: Semantic results return irrelevant or miss key documents, eroding user trust.
- Impact: High — users revert to other tools.
- Likelihood: Medium
- Mitigation (one-dev tasks): Implement hybrid retrieval (BM25 + semantic rerank) with conservative default weights; add result evidence snippets and a simple feedback button that logs incorrect/important results for later tuning.

3) Privacy risk: User data leakage (perceived or real)
- Description: Users may be unsure whether content leaves their device or is stored unencrypted.
- Impact: High — legal/regulatory and user trust consequences.
- Likelihood: Medium
- Mitigation (one-dev tasks): Make default mode local-only, add an explicit opt-in cloud sync toggle, surface a clear onboarding modal explaining what is indexed, and implement an "index preview" UI showing candidate files. Log telemetry only after opt-in and document retention rules.

4) Performance risk: Search latency at scale
- Description: ANN/FTS queries and large indexes increase query latency for large corpora.
- Impact: Medium — slow search degrades utility.
- Likelihood: Medium
- Mitigation (one-dev tasks): Add LRU result cache for recent queries, tune ANN parameters (e.g., efSearch) exposed as a diagnostic option, support partial/lightweight indexes (metadata-only) and paginated query execution with progress indicators.

5) Model risk: OCR / extraction quality
- Description: Poor OCR or entity extraction on images/screenshots reduces recall and structured search accuracy.
- Impact: Medium — breaks screenshot and entity queries.
- Likelihood: Medium
- Mitigation (one-dev tasks): Add a re-OCR button per document, integrate a fallback lexical-only extraction path, add unit tests for OCR pipeline and a small validation dataset; expose a setting to choose higher-accuracy models for power users.

6) UX risk: Complexity of controls and explanations
- Description: Excessive options for privacy, indexing, and model selection overwhelm non-technical users.
- Impact: Medium — feature discoverability and adoption suffer.
- Likelihood: Medium
- Mitigation (one-dev tasks): Implement progressive disclosure: sensible defaults on first run, an "Advanced settings" panel, and concise inline help tooltips. Add a short guided tour that highlights indexing privacy and feedback controls.

7) Scope risk: Feature creep and overreach
- Description: Expanding scope (agents, home timeline, multi-source sync) risks delaying core search quality work.
- Impact: High — slower delivery of core value.
- Likelihood: High
- Mitigation (one-dev tasks): Create a strict V2 scope checklist and a DO-NOT-IMPLEMENT short list; gate new features behind milestone acceptance criteria; require a one-sentence UX justification and time estimate before adding work to the board.

Action notes for the developer
- Prioritize: technical, product, and privacy mitigations first (they have highest combined impact).
- Make each mitigation small and measurable (add one feature, one metric, one test).
- Track progress with a short TODO list: incremental indexer, hybrid retrieval POC, local-only default + onboarding, result cache, re-OCR button, guided tour, scope checklist.

End of RISKS.md