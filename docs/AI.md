# AI for Life Search — Architecture & Guidance

This document describes the AI architecture for Life Search. It is written for a single developer implementing an on-device, provider-independent, offline-first AI stack. It covers the model adapter pattern, recommended local model choices for embeddings / OCR / NER, where to prefer deterministic algorithms, model lifecycle (download, cache, verification), offline-first inference UX, and how to later support proprietary training without sending user data off-device.

Goals
- Provider-independent: keep higher-level code agnostic to model vendors and runtimes.
- Local-first & privacy-preserving: run inference on-device, by default no cloud network calls for user data.
- Deterministic when appropriate: use classical algorithms for exactness and performance.
- Incremental: ship an MVP using CPU-friendly open-source models, with a clear migration path to stronger models and quantized runtimes.

Overview: where AI adds value
- Embeddings: convert documents / events / screenshots into vectors for semantic search and retrieval. High value for fuzzy matching and relevance scoring.
- NER / entity extraction: extract names, error codes, libraries, function names, locations. ML improves recall over brittle regexes.
- OCR: image -> text (screenshots, photos). Deterministic OCR (Tesseract) is sufficient for many cases; neural OCR improves messy inputs.
- Re-ranking / reranker: apply learned scoring after a fast candidate retrieval (hybrid search: lexical + embedding).
- Summarization / condensation: optional, for creating short memory summaries or preview text.
- Query understanding / intent classification: lightweight classifiers for routing queries to the right search pipeline.

Where deterministic algorithms are better
- Date/time parsing, path and filesystem operations, exact string matches, regex-based code-pattern extraction, content-hash deduplication, deterministic ranking signals (timestamps, explicit metadata). Use deterministic logic for anything that must be exact or explainable.

Provider-independent model adapter
Create a single small interface (ModelAdapter) that your app uses everywhere. Examples of minimal methods:
- embed(text: str) -> List[float]
- ner(text: str) -> List[Entity]
- ocr(image: bytes) -> str
- summarize(texts: List[str]) -> str  (optional)
- rank(query: str, candidates: List[str]) -> List[float]  (optional)

Adapter responsibilities
- Resolve which backend/runtime to use (PyTorch, ONNX, TensorFlow, ggml/llama.cpp, or external service) based on config and available hardware.
- Abstract model file paths and expose model metadata (id, version, checksum).
- Perform input preprocessing and output normalization (consistent entity formats, vector dims).
- Provide a null/fallback adapter that implements the same API using deterministic heuristics (e.g., TF-IDF or rule-based NER) when a model is unavailable.

Local model choices (recommended for MVP — CPU friendly)
These are pragmatic, fast-to-integrate options for an MVP on typical developer machines (no GPU required):
- Embeddings
  - sentence-transformers/all-MiniLM-L6-v2 (Hugging Face) — very small, fast on CPU, 384-dim vectors. Good for semantic search and clustering.
  - Alternative: sentence-transformers/distilbert-base-nli-stsb-mean-tokens (larger, slower).
- NER / extraction
  - spaCy en_core_web_sm — small statistical model; extend with rule-based matchers (spaCy's Matcher) for domain tokens (error codes, file paths).
  - For higher quality later: spaCy transformer models or Hugging Face token-classification models.
- OCR
  - Tesseract OCR (open-source native binary) — robust and lightweight for screenshots and well-formatted text.
  - Optional: easyocr (PyTorch) or Kraken for harder layouts; note these increase dependency and CPU load.
- Summarization / small LLMs (optional)
  - Skip for MVP. If needed for local-only summarization, evaluate tiny quantized models (ggml versions) on capable machines; otherwise rely on deterministic templates.

Memory & compute notes (MVP)
- all-MiniLM-L6-v2 embedding model: tens of MBs on disk; fast inference on CPU (ms–100s ms per call depending on hardware and batching).
- spaCy en_core_web_sm: ~50MB disk; low CPU.
- Tesseract: native binary (system dependency); OCR cost depends on image size and tessdata language packs.

Model lifecycle: download, cache, verify, and evict
- Model download manager responsibilities:
  - Discover model by id and version (e.g., models.json manifest hosted in releases).
  - Download to a temporary file, verify checksum/signature, atomically move into cache.
  - Store models under an app-controlled cache dir (e.g., %LOCALAPPDATA%/life-search/models or ./cache/models).
  - Maintain metadata: model id, version, source URL, checksum, downloaded_at, backend (onnx/pt/ggml), expected input dim.
  - Implement simple eviction (LRU) with configurable total cache size.
- On startup, the adapter should check for required model(s), load them if present, and fall back to the null adapter otherwise.
- Offer an in-app prompt to download optional larger models (GPU users) — explicit opt-in.

Offline-first inference & UX
- Default: attempt local inference with cached models.
- If model missing, fall back to deterministic behavior and surface a single non-blocking UI prompt: "Download model for improved accuracy" with size and privacy info.
- Allow users to pre-download needed models for offline use.
- Log model load/availability to telemetry (local only) for diagnostics — do not send user data.

How to support proprietary model training later without sending raw user data off-device
- Local fine-tuning and adapter-based updates
  - Use adapter-friendly fine-tuning patterns like LoRA or adapter modules that are small (few MB) compared to full models.
  - Allow users to fine-tune locally on their device; produce only the adapter files (e.g., LoRA weights) that are additive and do not contain raw text logs.
  - If a user explicitly opts in to share their adapter to a central service, document that the adapter contains learned weights derived from their data and obtain explicit consent.
- Differential / aggregated updates (future)
  - Consider secure aggregation protocols where devices send encrypted, gradient-level summaries to a coordinator that aggregates without inspecting individual contributions.
  - Prefer on-device personalization as the first-class approach: keep models private and store adapters locally or as user-managed exports.
- Export and audit
  - Provide an export tool that lets users review which examples were used for fine-tuning (anonymized / redacted) before sharing any derived artifact.

Migration path to better models
- Keep the adapter API stable — this makes swapping backends trivial.
- Short-term upgrades
  - Move from all-MiniLM to larger sentence-transformers models (e.g., all-mpnet-base-v2) for better embeddings.
  - Replace spaCy-sm with spaCy-transformer or HF token-classification for NER.
- Medium-term: quantization & ggml runtimes
  - Use quantized formats (ONNX int8, ggml) to run larger LLMs locally with reasonable RAM/CPU. Tools like llama.cpp enable small LLMs on CPU.
- Long-term: GPU & server-side optional workflows
  - For power-users or optional cloud features, allow configuration to point adapters to a local GPU runtime or a trusted remote inference endpoint.

Implementation notes for one developer (practical steps)
1. Implement ModelAdapter (small module) with the methods above and a fallback adapter.
2. Prototype with Python:
   - sentence-transformers to produce embeddings
   - spaCy for NER
   - pytesseract or subprocess call to tesseract for OCR
3. Build a tiny model manager that downloads models.json and provides a CLI / UI button to fetch models.
4. Add unit tests for adapter method contracts: consistent vector length, entity schema, OCR return type.
5. Measure latency & memory on a low-end dev machine and document acceptable timeouts and batching strategies.

Security, privacy checklist
- Do not transmit user text/images to any remote service by default.
- Verify model checksums and use HTTPS for model downloads.
- Make any cloud/upload or training features explicit, opt-in, and auditable.
- Document the data flow for each feature in PRIVACY.md (link to existing [PRIVACY.md](/d:/project/New folder/docs/PRIVACY.md)).

Recommended starting list (MVP)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- NER: spaCy en_core_web_sm + spaCy Matcher rules for domain tokens
- OCR: Tesseract (system package) with language packs as needed

Closing notes
Start small: implement the adapter + a single embedding-backed search pipeline and Tesseract OCR for screenshots. Ship deterministic fallbacks so the product works without models and progressively improve accuracy by adding optional model downloads. Keep privacy and explicit user consent as first-class constraints — prefer local personalization via adapters over centralized training.

If you'd like, next steps I can generate a starter ModelAdapter example (Python) and a model manifest schema for the download manager.