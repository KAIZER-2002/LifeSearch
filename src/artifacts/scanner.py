import hashlib
import os
from typing import Dict, Iterable, List, Optional, Tuple

from .extractor import Extractor
from .store import ArtifactStore, _format_timestamp


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

    def __init__(self, store: ArtifactStore, extractor: Extractor):
        self.store = store
        self.extractor = extractor

    def index_folder(self, folder_path: str) -> Dict[str, int]:
        base_folder = os.path.abspath(folder_path)
        if not os.path.isdir(base_folder):
            raise FileNotFoundError(f"Index path not found: {base_folder}")

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
                if not self.store.artifact_needs_index(path, size, modified_at):
                    skipped += 1
                    continue

                mime_type = self.EXTENSION_TO_MIME.get(extension, "application/octet-stream")
                extracted_text, extract_error = self.extractor.extract_text(path, mime_type)
                content_hash = self._hash_file(path)
                try:
                    self.store.upsert_artifact(
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
                except Exception:
                    errors += 1

        self.store.mark_missing_artifacts(current_paths)
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
