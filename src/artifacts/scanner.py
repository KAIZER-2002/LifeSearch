from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .extractor import Extractor
from .store import ArtifactStore, _format_timestamp
from src.vector.chunker import TextChunker

logger = logging.getLogger(__name__)


class ArtifactScanner:
    SUPPORTED_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
        ".docx",
        ".png",
        ".jpg",
        ".jpeg",
    }

    EXTENSION_TO_MIME = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }

    def __init__(
        self,
        store: ArtifactStore,
        extractor: Extractor,
        vector_store: Optional[Any] = None,
        embedding_engine: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.chunker = TextChunker()

    def index_folder(
        self, folder_path: str, reindex_on_model_change: bool = False
    ) -> Dict[str, int]:
        base_folder = os.path.abspath(folder_path)
        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f"Index path not found: {base_folder}")

        # --- Embedding model-change detection (safe, non-blocking) ---
        # We never silently perform a massive blocking re-embedding. Old-model
        # vectors are left isolated; only changed/new files get the current
        # model. `reindex_on_model_change` is the explicit conversion path.
        current_model = (
            self.embedding_engine.model_id
            if (self.embedding_engine and getattr(self.embedding_engine, "dimension", 0) > 0)
            else None
        )
        model_mismatch = False
        if current_model and self.vector_store is not None:
            stored = self.vector_store.stored_model_ids()
            if stored and current_model not in stored:
                model_mismatch = True
                if not reindex_on_model_change:
                    logger.warning(
                        "Embedding model changed (stored=%s, current=%s). "
                        "Old-model vectors remain isolated; new files are indexed with the "
                        "current model. Run `index --reindex` to convert everything.",
                        stored,
                        current_model,
                    )
        force_reembed = reindex_on_model_change and model_mismatch

        current_paths: List[str] = []
        processed = 0
        skipped = 0
        errors = 0

        for root, _dirs, files in os.walk(base_folder):
            for file_name in files:
                path = os.path.abspath(os.path.join(root, file_name))
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in self.SUPPORTED_EXTENSIONS:
                    continue
                current_paths.append(path)
                try:
                    stat = os.stat(path)
                except OSError:
                    errors += 1
                    continue

                size = stat.st_size
                modified_at = _format_timestamp(stat.st_mtime)
                created_at = _format_timestamp(stat.st_ctime)
                if not force_reembed and not self.store.artifact_needs_index(path, size, modified_at):
                    skipped += 1
                    continue

                mime_type = self.EXTENSION_TO_MIME.get(extension, "application/octet-stream")
                extracted_text, extract_error = self.extractor.extract_text(path, mime_type)
                content_hash = self._hash_file(path)
                try:
                    art_id = self.store.upsert_artifact(
                        path=path,
                        file_name=file_name,
                        mime_type=mime_type,
                        size=size,
                        created_at=created_at,
                        modified_at=modified_at,
                        extracted_text=extracted_text,
                        content_hash=content_hash,
                        extract_error=extract_error,
                    )
                    processed += 1

                    # Semantic Indexing with Failure Isolation
                    if (
                        self.vector_store is not None
                        and self.embedding_engine is not None
                        and getattr(self.embedding_engine, "dimension", 0) > 0
                        and extracted_text
                    ):
                        try:
                            source_type = "ocr_text" if mime_type.startswith("image/") else "document_text"
                            chunks = self.chunker.chunk_text(extracted_text, art_id, source_type=source_type)
                            if chunks:
                                embeddings = self.embedding_engine.embed_batch([c.text for c in chunks])
                                valid_chunks = [c for i, c in enumerate(chunks) if len(embeddings[i]) > 0]
                                valid_embs = [e for e in embeddings if len(e) > 0]
                                if valid_chunks:
                                    self.vector_store.save_chunks(
                                        artifact_id=art_id,
                                        chunks=valid_chunks,
                                        embeddings=valid_embs,
                                        model_id=self.embedding_engine.model_id,
                                        dimension=self.embedding_engine.dimension,
                                    )
                        except Exception as vec_exc:
                            logger.warning(f"Vector embedding failed for {path}: {vec_exc}")

                except Exception:
                    errors += 1

        # Propagate deletions to the vector store so no orphan vectors remain
        # searchable for removed/missing artifacts.
        newly_missing = self.store.mark_missing_artifacts(current_paths)
        if self.vector_store is not None:
            for aid in newly_missing:
                try:
                    self.vector_store.delete_artifact_chunks(aid)
                except Exception as exc:
                    logger.warning(f"Failed to delete vectors for missing artifact {aid}: {exc}")

        return {
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
        }

    def _hash_file(self, path: str) -> Optional[str]:
        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except OSError:
            return None
