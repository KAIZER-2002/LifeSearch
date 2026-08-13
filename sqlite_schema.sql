-- SQLite schema for Life Search MVP
PRAGMA foreign_keys = ON;

-- documents: one row per distinct file (deduplicated by content_hash)
CREATE TABLE IF NOT EXISTS documents (
  id TEXT PRIMARY KEY,             -- UUID
  path TEXT NOT NULL,              -- absolute path
  content_hash TEXT NOT NULL,      -- SHA256 of bytes
  file_name TEXT,
  mime_type TEXT,
  size INTEGER,
  device TEXT,
  app_origin TEXT,                 -- e.g., "Chrome", "VSCode" if known
  created_at INTEGER,              -- epoch ms from file metadata
  modified_at INTEGER,
  indexed_at INTEGER,             -- epoch ms when indexed
  thumbnail_path TEXT,             -- optional generated thumbnail
  doc_embedding_id TEXT,           -- pointer to aggregate embedding if stored separately
  metadata_json TEXT               -- reserved JSON for extra fields
);

-- ensure dedupe on content
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);

-- chunks: semantically coherent chunks of documents
CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,       -- UUID
  document_id TEXT NOT NULL, -- FK to documents.id
  chunk_index INTEGER NOT NULL,
  text_snippet TEXT,         -- short snippet for display
  text_path TEXT,            -- path to cached full chunk text if needed
  start_offset INTEGER,      -- char offset in document
  end_offset INTEGER,
  tokens_est INTEGER,
  embedding_id TEXT,         -- pointer to vector store id
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

-- Full-text search on chunk text for exact / fuzzy matches
-- Uses FTS5 for fast snippet extraction
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(text, document_id UNINDEXED, tokenize = 'porter');

-- accesses: logs for recency and frequency signals
CREATE TABLE IF NOT EXISTS accesses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id TEXT,
  access_type TEXT,      -- 'open','click','pin','preview'
  timestamp INTEGER
);
CREATE INDEX IF NOT EXISTS idx_accesses_doc ON accesses(document_id);
CREATE INDEX IF NOT EXISTS idx_accesses_time ON accesses(timestamp);

-- projects: optional project nodes for quick grouping
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY,
  name TEXT,
  document_ids TEXT      -- JSON array for small quick lookups (expandable)
);

-- re_rank_feedback: store user interactions to tune ranking
CREATE TABLE IF NOT EXISTS re_rank_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_text TEXT,
  document_id TEXT,
  action TEXT, -- 'click','ignore','pin'
  timestamp INTEGER
);

-- simple metadata table for embedding id -> chunk mapping (if using external vector store)
CREATE TABLE IF NOT EXISTS embeddings_map (
  embedding_id TEXT PRIMARY KEY,
  chunk_id TEXT,
  vector_meta_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings_map(chunk_id);

-- schema versioning
CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at INTEGER
);

-- helper view: document-level aggregated stats
CREATE VIEW IF NOT EXISTS document_stats AS
SELECT d.id as document_id,
       d.file_name,
       d.path,
       d.mime_type,
       COUNT(c.id) AS chunk_count,
       MAX(a.timestamp) AS last_access_ms
FROM documents d
LEFT JOIN chunks c ON c.document_id = d.id
LEFT JOIN accesses a ON a.document_id = d.id
GROUP BY d.id;

-- Sample trigger: keep fts_chunks in sync when inserting into chunks
CREATE TRIGGER IF NOT EXISTS chunks_after_insert
AFTER INSERT ON chunks
BEGIN
  INSERT INTO fts_chunks(rowid, text, document_id) VALUES (new.rowid, new.text_snippet, new.document_id);
END;

CREATE TRIGGER IF NOT EXISTS chunks_after_delete
AFTER DELETE ON chunks
BEGIN
  DELETE FROM fts_chunks WHERE rowid = old.rowid;
END;

-- End of schema

-- Notes:
-- 1) The actual blob vectors are stored in the vector store (FAISS/Qdrant). embeddings_map links embedding_id <-> chunk_id.
-- 2) text_path in chunks points to a cached plain-text file stored under the app cache directory to avoid storing large blobs in SQLite.
-- 3) All timestamps are epoch milliseconds to simplify JS/OS interop.
