# Life Search — Indexer Design (MVP)

This document specifies the indexer design for the Life Search MVP. It covers preprocessing, chunking, embeddings, vector store config, deduplication, scheduling, and operational controls.

## Goals
- Produce repeatable, incremental indexing of user-selected folders.
- Create chunk-level embeddings and a small metadata DB to support fast ANN + metadata re-rank queries.
- Keep the design local-first and resource-conscious.

## High-level pipeline
1. File discovery (crawler / FS watcher)
2. Format-specific extraction (PDF, DOCX, TXT/MD)
3. Image OCR (screenshots) where applicable
4. Chunking & canonicalization
5. Embedding computation
6. Index insertion: vector store (chunk embedding) + SQLite metadata (documents, chunks, fts_chunks)
7. Background scheduling and incremental updates

## Tools & components (MVP choices)
- Language/runtime: prototype in Python for speed; production native (Rust/Go) later.
- PDF extraction: pdfminer.six or poppler-based (pdftotext) for robust text + layout hints.
- DOCX: python-docx or Mammoth.
- OCR: Tesseract (tuned psm 6 for screenshots). Use whitelist and ASCII normalization for short error lines.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (CPU-friendly). Provide optional GPU path for larger models.
- Vector store: FAISS local HNSW + PQ (on-disk mmap). Optionally ship with local Qdrant if user installs it.
- Metadata DB: SQLite + FTS5.

## Chunking strategy
- Goal: balance retrieval precision vs. index size.
- Default: target chunk_size_tokens = 512 tokens (approx 3–4 KB), overlap = 128 tokens.
- For short files (<512 tokens): single chunk.
- For code files: prefer logical block chunking (split by blank lines, function defs) where feasible.
- For images: OCR text produces short "virtual document" which should create 1–3 chunks depending on length.
- Save primary snippet of first ~400 chars as chunks.text_snippet.

Rationale: 512-token chunks provide enough context for semantic models without exploding chunk count for long docs.

## Embeddings
- Model: all-MiniLM-L6-v2 (embedding_dim=384) for MVP.
- Compute chunk-level embedding and a document-level aggregate embedding (mean-pool of chunk embeddings).
- Store embeddings on disk in vector store; maintain embeddings_map table linking embedding_id -> chunk_id.
- Persist embeddings as float16 to reduce disk space; enable PQ quantization for large corpora.

## Vector store config (FAISS HNSW + PQ)
- Index type: HNSW for nearest neighbour with PQ for disk efficiency.
- HNSW parameters (M=32, efConstruction=200) for build quality.
- efSearch=128 default at query time.
- Keep index memory-mapped for read performance.
- Batch insertion during indexing using long-lived index builder for speed.

## Deduplication & canonicalization
- Compute SHA256 of file bytes -> content_hash.
- If identical content_hash exists, append path to documents.metadata_json.paths and do not re-store duplicate embeddings.
- For near-duplicates (e.g., same PDF with different metadata), compute doc-level embedding similarity; if > 0.95, mark as near-duplicate and create cross-reference.

## Metadata and FTS
- Store minimal needed metadata in documents table for ranking (modified_at, indexed_at, app_origin, device).
- Store chunk text in small cached files; use fts_chunks for exact/fuzzy keyword matches to aid snippet highlighting.

## Scheduling & resource controls
- First-run: full crawl in prioritized order (small files first), run in background with a visible progress bar.
- Realtime: FS watcher triggers incremental reindex for changed/new files.
- Heavy ops (OCR, embeddings) only when:
  - device is on AC power OR user forced immediate indexing
  - CPU utilization < threshold (configurable)
- Rate limiting: indexer runs at N threads (default 2) and uses bounded queue + backpressure for embedding requests.
- Batch embeddings into groups of 32–128 chunks to amortize model costs.

## Failure & retry
- On transient embedding/model failure, re-enqueue chunk with exponential backoff (max 3 tries) and log detailed error in local logs.
- On corrupted file extraction, flag document with status='extract_error' and surface in UI for manual reattempt.

## Incremental updates
- For each file change event:
  - Compute content_hash; if unchanged, skip.
  - If changed: remove old chunks for document_id (or compute diff if implemented later), then re-chunk, re-embed and upsert vectors and metadata.
- Provide a compaction job to remove stale embeddings and reclaim disk space.

## Access signals & feedback
- Log user actions (open, click, pin) into accesses and re_rank_feedback tables.
- Periodically compute access frequency decayed over time for recencyBoost and accessFreqBoost.

## CLI examples
- lifecmd index --paths "C:\Users\Me\Documents" --priority
- lifecmd status
- lifecmd reindex --document-id <id>
- lifecmd compact-index

## Acceptance criteria (indexer)
- For a 10k-token document, chunking produces roughly ceil(10k / ~2048) expected chunks according to token estimate rules and stored chunks link to parent document.
- Searching a sentence from a document returns that document within top-3 results in a local test harness.
- OCRed screenshot containing an error string returns the screenshot when querying the exact error string.

## Performance expectations (MVP)
- Embedding speed (CPU): ~100–300 chunks/sec depending on CPU and batch size; GPU optional for higher throughput.
- ANN query latency (on-disk FAISS HNSW): median < 200ms for corpora up to ~100k chunks on modern laptops.
- Indexing: initial 10k-file corpus should begin returning results in minutes as incremental chunks get indexed; full indexing time depends on OCR and CPU.

## Security & privacy considerations
- Default local-only: no network calls for content.
- Cache and DB stored in app data folder; optional encryption with OS keystore.
- Provide user controls to include/exclude folders and to preview before indexing.

## Next improvements (post-MVP)
- Add audio transcription pipeline (Whisper or local ASR) and index transcripts.
- Improve chunking with semantic boundary detection.
- Optional local LLM summarizer for higher-quality snippets.
- Optional zero-knowledge encrypted cloud sync of index blobs.

## Implementation tips
- Start with a small prototype in Python to validate chunking, embeddings, and FAISS integration quickly.
- Keep embedding/FAISS components isolated: design an adapter so vector store can be swapped (local FAISS -> local Qdrant -> cloud vector DB) without schema changes.
- Use SQLite PRAGMAs for performance (journal_mode=WAL, synchronous=NORMAL) and test with expected corpus sizes.

## Example dev workflow
1. Download sample embedding model and place under app cache.
2. Run lifecmd index --paths "sample_corpus/" --priority
3. Use the local dev UI to run test queries and verify retrieval.

-- End of indexer_design.md --
