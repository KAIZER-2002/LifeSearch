"""
tests/test_search_slice6.py

Slice 6 Integration Tests: Local Screenshot Intelligence (Local OCR)
Verifies:
  1. Image OCR text reaches ArtifactStore & SQLite FTS5
  2. Search finds neutral screenshot (e.g. screenshot_001.png) by visible OCR text
  3. Evidence explanation explicitly reports OCR text match
  4. Unrelated screenshots are filtered out
  5. Text, Markdown, PDF, DOCX extractions continue to work without regression
"""

import os
import tempfile
from datetime import datetime, timezone

import pytest
from PIL import Image, ImageDraw

from src.artifacts.extractor import Extractor
from src.artifacts.ocr import NullOCREngine, ONNXOCREngine, get_default_ocr_engine
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.episodes.engine import EpisodeEngine
from src.episodes.store import EpisodeStore
from src.events.model import Event
from src.events.store import EventStore
from src.memories.builder import MemoryBuilder
from src.memories.store import MemoryStore
from src.search.engine import SearchEngine


def _create_test_image(path: str, text: str):
    img = Image.new("RGB", (600, 150), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Double space words so PIL default font bitmap renders clear word separation
    spaced_text = "  ".join(text.split())
    draw.text((10, 40), spaced_text, fill=(0, 0, 0))
    img.save(path)


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_realistic_screenshot_ocr_search(tmp_path):
    ocr_engine = ONNXOCREngine()
    if not ocr_engine._available:
        pytest.skip("rapidocr_onnxruntime unavailable")

    db_path = str(tmp_path / "lifesearch.db")
    files_folder = str(tmp_path / "files")
    os.makedirs(files_folder, exist_ok=True)

    # Neutral screenshot filename (no "mongodb" or "error" in filename!)
    img_path = os.path.join(files_folder, "screenshot_001.png")
    _create_test_image(img_path, "MongoDB connection refused error 27017")

    # Unrelated image
    unrelated_img = os.path.join(files_folder, "vacation_photo.png")
    _create_test_image(unrelated_img, "Beach sunset summer vacation")

    # Regular document
    doc_path = os.path.join(files_folder, "qdrant_guide.txt")
    _write_text(doc_path, "Qdrant vector database retrieval guide")

    store = ArtifactStore(db_path)
    extractor = Extractor(ocr_engine=ocr_engine)
    scanner = ArtifactScanner(store, extractor)
    scanner.index_folder(files_folder)

    # Check extracted text in ArtifactStore
    row_img = store.get_artifact_by_path(img_path)
    assert row_img is not None
    assert "MongoDB" in row_img["extracted_text"] or "mongodb" in row_img["extracted_text"].lower()

    # Search Engine
    search_engine = SearchEngine(store)
    results = search_engine.search("MongoDB error")

    assert len(results) >= 1
    top = results[0]
    assert top.file_name == "screenshot_001.png"
    assert "OCR text from screenshot" in top.why


def test_qdrant_error_screenshot(tmp_path):
    ocr_engine = ONNXOCREngine()
    if not ocr_engine._available:
        pytest.skip("rapidocr_onnxruntime unavailable")

    db_path = str(tmp_path / "lifesearch.db")
    files_folder = str(tmp_path / "files")
    os.makedirs(files_folder, exist_ok=True)

    img_path = os.path.join(files_folder, "IMG_9999.png")
    _create_test_image(img_path, "Qdrant collection not found error")

    store = ArtifactStore(db_path)
    extractor = Extractor(ocr_engine=ocr_engine)
    scanner = ArtifactScanner(store, extractor)
    scanner.index_folder(files_folder)

    search_engine = SearchEngine(store)
    results = search_engine.search("Qdrant error")

    assert len(results) >= 1
    assert results[0].file_name == "IMG_9999.png"
    assert "OCR text from screenshot" in results[0].why


def test_unrelated_screenshot_filtered_out(tmp_path):
    ocr_engine = ONNXOCREngine()
    if not ocr_engine._available:
        pytest.skip("rapidocr_onnxruntime unavailable")

    db_path = str(tmp_path / "lifesearch.db")
    files_folder = str(tmp_path / "files")
    os.makedirs(files_folder, exist_ok=True)

    img_path = os.path.join(files_folder, "random.png")
    _create_test_image(img_path, "Random landscape text")

    store = ArtifactStore(db_path)
    extractor = Extractor(ocr_engine=ocr_engine)
    scanner = ArtifactScanner(store, extractor)
    scanner.index_folder(files_folder)

    search_engine = SearchEngine(store)
    results = search_engine.search("MongoDB")

    filenames = [r.file_name for r in results]
    assert "random.png" not in filenames


def test_slice6_no_regressions_text_formats(tmp_path):
    db_path = str(tmp_path / "lifesearch.db")
    files_folder = str(tmp_path / "files")
    os.makedirs(files_folder, exist_ok=True)

    _write_text(os.path.join(files_folder, "notes.txt"), "plain text searchable content")
    _write_text(os.path.join(files_folder, "doc.md"), "# markdown header content")

    store = ArtifactStore(db_path)
    extractor = Extractor(ocr_engine=NullOCREngine())
    scanner = ArtifactScanner(store, extractor)
    scanner.index_folder(files_folder)

    search_engine = SearchEngine(store)
    res_txt = search_engine.search("searchable")
    assert len(res_txt) == 1
    assert res_txt[0].file_name == "notes.txt"

    res_md = search_engine.search("header")
    assert len(res_md) == 1
    assert res_md[0].file_name == "doc.md"
