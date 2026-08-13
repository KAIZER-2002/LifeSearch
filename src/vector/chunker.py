from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TextChunk:
    chunk_id: str
    artifact_id: int
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    source_type: str = "document_text"  # "document_text" | "ocr_text"
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 15,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        artifact_id: int,
        source_type: str = "document_text",
        metadata: getattr(Dict[str, Any], "__origin__", None) or Dict[str, Any] = None,
    ) -> List[TextChunk]:
        if not text or not text.strip():
            return []

        raw_text = text.strip()
        length = len(raw_text)
        if length < self.min_chunk_size:
            return []

        chunks: List[TextChunk] = []
        start = 0
        chunk_idx = 0

        while start < length:
            end = min(start + self.chunk_size, length)
            
            # Try to break at a paragraph/sentence boundary if near end
            if end < length:
                break_point = max(
                    raw_text.rfind("\n\n", start + self.min_chunk_size, end),
                    raw_text.rfind("\n", start + self.min_chunk_size, end),
                    raw_text.rfind(". ", start + self.min_chunk_size, end),
                )
                if break_point > start:
                    end = break_point + 1

            chunk_str = raw_text[start:end].strip()
            if len(chunk_str) >= self.min_chunk_size:
                cid = f"{artifact_id}_c{chunk_idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        artifact_id=artifact_id,
                        text=chunk_str,
                        start_char=start,
                        end_char=end,
                        chunk_index=chunk_idx,
                        source_type=source_type,
                        metadata=dict(metadata or {}),
                    )
                )
                chunk_idx += 1

            if end >= length:
                break
            # Guarantee the scan window strictly advances. A sentence/
            # paragraph boundary that falls within chunk_overlap of `start`
            # can pin `start` (end - chunk_overlap <= start) so the loop never
            # reaches `length` and `chunks` grows without bound.
            new_start = end - self.chunk_overlap
            start = new_start if new_start > start else start + 1

        return chunks
