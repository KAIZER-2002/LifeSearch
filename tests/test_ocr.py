import os
import tempfile

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine, ONNXOCREngine, get_default_ocr_engine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore


class CountingOCREngine:
    """Mock OCR engine for testing incremental indexing change detection."""

    def __init__(self, text: str = "Sample OCR Text"):
        self.text = text
        self.call_count = 0

    def extract_text(self, image_path: str):
        self.call_count += 1
        return self.text, None


def _create_test_image(path: str, text: str = "MongoDB connection refused"):
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    spaced_text = "  ".join(text.split())
    draw.text((10, 40), spaced_text, fill=(0, 0, 0))
    img.save(path)


# ---------------------------------------------------------------------------
# 1. NullOCREngine tests
# ---------------------------------------------------------------------------

def test_null_ocr_engine_returns_empty():
    engine = NullOCREngine()
    text, err = engine.extract_text("dummy.png")
    assert text == ""
    assert err is None


# ---------------------------------------------------------------------------
# 2. Extractor fallback when OCR engine is None
# ---------------------------------------------------------------------------

def test_extractor_image_no_ocr_engine():
    extractor = Extractor(ocr_engine=None)
    text, err = extractor.extract_text("image.png", "image/png")
    assert text == ""
    assert err is None


# ---------------------------------------------------------------------------
# 3. Incremental indexing (counting OCR engine)
# ---------------------------------------------------------------------------

def test_incremental_indexing_skips_ocr_when_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        img_path = os.path.join(tmp, "screenshot_001.png")
        _create_test_image(img_path, "MongoDB error screen")

        store = ArtifactStore(db_path)
        counting_ocr = CountingOCREngine(text="MongoDB connection refused")
        extractor = Extractor(ocr_engine=counting_ocr)
        scanner = ArtifactScanner(store, extractor)

        # Index 1: first time
        res1 = scanner.index_folder(tmp)
        assert res1["processed"] == 1
        assert counting_ocr.call_count == 1

        # Index 2: unchanged screenshot
        res2 = scanner.index_folder(tmp)
        assert res2["skipped"] == 1
        assert counting_ocr.call_count == 1  # OCR NOT called again!

        # Modify image
        _create_test_image(img_path, "Updated MongoDB error screen")
        # Touch modification time
        os.utime(img_path, None)

        # Index 3: modified screenshot
        res3 = scanner.index_folder(tmp)
        assert res3["processed"] == 1
        assert counting_ocr.call_count == 2  # OCR called second time!

        store.close()


# ---------------------------------------------------------------------------
# 4. Failure test: corrupted PNG file
# ---------------------------------------------------------------------------

def test_corrupted_image_handling():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "ls.db")
        corrupt_img = os.path.join(tmp, "corrupt.png")
        with open(corrupt_img, "wb") as f:
            f.write(b"NOT_A_REAL_PNG_HEADER_CORRUPTED_BYTES")

        store = ArtifactStore(db_path)
        ocr_engine = get_default_ocr_engine()
        extractor = Extractor(ocr_engine=ocr_engine)
        scanner = ArtifactScanner(store, extractor)

        res = scanner.index_folder(tmp)
        assert res["processed"] == 1  # Ingested, does not crash!

        artifact = store.get_artifact_by_path(corrupt_img)
        assert artifact is not None
        assert artifact["file_name"] == "corrupt.png"

        store.close()


# ---------------------------------------------------------------------------
# 5. Real ONNXOCREngine test (if rapidocr_onnxruntime is available)
# ---------------------------------------------------------------------------

def test_real_onnx_ocr_extraction(tmp_path):
    engine = ONNXOCREngine()
    if not engine._available:
        pytest.skip("rapidocr_onnxruntime model/runtime unavailable")

    img_path = str(tmp_path / "real_test.png")
    _create_test_image(img_path, "MongoDB error 27017")

    text, err = engine.extract_text(img_path)
    assert err is None
    assert isinstance(text, str)
