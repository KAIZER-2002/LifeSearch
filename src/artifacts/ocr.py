from __future__ import annotations

import logging
from typing import Optional, Protocol, Tuple

logger = logging.getLogger(__name__)


class OCREngine(Protocol):
    """Protocol for local OCR text extraction from images."""

    def extract_text(self, image_path: str) -> Tuple[str, Optional[str]]:
        """
        Extract text from an image file.

        Returns:
            Tuple of (extracted_text, error_message_or_None)
        """
        ...


class NullOCREngine:
    """Fallback OCR engine used when local OCR backend is unavailable."""

    def extract_text(self, image_path: str) -> Tuple[str, Optional[str]]:
        return "", None


class ONNXOCREngine:
    """Local CPU-based OCR engine powered by rapidocr_onnxruntime."""

    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            self._engine = RapidOCR()
            self._available = True
        except Exception as exc:
            logger.warning(f"Failed to initialize RapidOCR: {exc}")
            self._engine = None
            self._available = False

    def extract_text(self, image_path: str) -> Tuple[str, Optional[str]]:
        if not self._available or self._engine is None:
            return "", "RapidOCR engine not available"

        try:
            result, _elapsed = self._engine(image_path)
            if not result:
                return "", None

            # RapidOCR returns list of items: [box, text, score]
            text_lines = []
            for item in result:
                if len(item) >= 2 and item[1]:
                    text_lines.append(str(item[1]).strip())

            extracted = "\n".join(text_lines)
            return extracted, None
        except Exception as exc:
            return "", str(exc)


def get_default_ocr_engine() -> OCREngine:
    """Factory returning ONNXOCREngine if available, else NullOCREngine."""
    engine = ONNXOCREngine()
    if engine._available:
        return engine
    return NullOCREngine()
