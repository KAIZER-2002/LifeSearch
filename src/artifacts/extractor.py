from __future__ import annotations

import os
from typing import Optional, Tuple

from PyPDF2 import PdfReader
from docx import Document

from .ocr import OCREngine


class Extractor:
    def __init__(self, ocr_engine: Optional[OCREngine] = None) -> None:
        self.ocr_engine = ocr_engine

    def extract_text(self, path: str, mime_type: str) -> Tuple[str, Optional[str]]:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".txt", ".md", ".markdown"}:
            return self._read_text_file(path)
        if ext == ".pdf":
            return self._read_pdf(path)
        if ext == ".docx":
            return self._read_docx(path)
        if ext in {".png", ".jpg", ".jpeg"} or (mime_type and mime_type.startswith("image/")):
            return self._read_image(path)
        return "", None

    def _read_image(self, path: str) -> Tuple[str, Optional[str]]:
        if self.ocr_engine is None:
            return "", None
        try:
            return self.ocr_engine.extract_text(path)
        except Exception as exc:
            return "", str(exc)

    def _read_text_file(self, path: str) -> Tuple[str, Optional[str]]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read(), None
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as handle:
                    return handle.read(), None
            except Exception as exc:
                return "", str(exc)
        except Exception as exc:
            return "", str(exc)

    def _read_pdf(self, path: str) -> Tuple[str, Optional[str]]:
        try:
            with open(path, "rb") as handle:
                reader = PdfReader(handle)
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n".join(text_parts), None
        except Exception as exc:
            return "", str(exc)

    def _read_docx(self, path: str) -> Tuple[str, Optional[str]]:
        try:
            document = Document(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
            return "\n".join(paragraphs), None
        except Exception as exc:
            return "", str(exc)
