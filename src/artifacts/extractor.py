import os
from typing import Tuple

from PyPDF2 import PdfReader
from docx import Document


class Extractor:
    def extract_text(self, path: str, mime_type: str) -> Tuple[str, str | None]:
        ext = os.path.splitext(path)[1].lower()
        if ext in {".txt", ".md", ".markdown"}:
            return self._read_text_file(path)
        if ext == ".pdf":
            return self._read_pdf(path)
        if ext == ".docx":
            return self._read_docx(path)
        return "", None

    def _read_text_file(self, path: str) -> Tuple[str, str | None]:
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

    def _read_pdf(self, path: str) -> Tuple[str, str | None]:
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

    def _read_docx(self, path: str) -> Tuple[str, str | None]:
        try:
            document = Document(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text]
            return "\n".join(paragraphs), None
        except Exception as exc:
            return "", str(exc)
