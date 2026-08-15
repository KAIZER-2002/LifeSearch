import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def create_pdf_file(path: str) -> None:
    from PyPDF2 import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(path, "wb") as handle:
        writer.write(handle)


def create_docx_file(path: str, content: str) -> None:
    from docx import Document

    document = Document()
    document.add_paragraph(content)
    document.save(path)


def test_indexing_supported_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            extractor = Extractor()
            scanner = ArtifactScanner(store, extractor)

            create_text_file(os.path.join(temp_dir, "notes.txt"), "hello world")
            create_text_file(os.path.join(temp_dir, "readme.md"), "# title\nmarkdown content")
            create_pdf_file(os.path.join(temp_dir, "blank.pdf"))
            create_docx_file(os.path.join(temp_dir, "doc.docx"), "doc content")
            create_text_file(os.path.join(temp_dir, "image.png"), "not a real image")

            result = scanner.index_folder(temp_dir)
            assert result["processed"] == 5
            assert result["skipped"] == 0
            assert result["errors"] == 0
            assert store.artifact_count() == 5

            text_artifact = store.get_artifact_by_path(os.path.join(temp_dir, "notes.txt"))
            assert text_artifact is not None
            assert "hello world" in text_artifact["extracted_text"]

            markdown_artifact = store.get_artifact_by_path(os.path.join(temp_dir, "readme.md"))
            assert markdown_artifact is not None
            assert "markdown content" in markdown_artifact["extracted_text"]

            docx_artifact = store.get_artifact_by_path(os.path.join(temp_dir, "doc.docx"))
            assert docx_artifact is not None
            assert "doc content" in docx_artifact["extracted_text"]

            pdf_artifact = store.get_artifact_by_path(os.path.join(temp_dir, "blank.pdf"))
            assert pdf_artifact is not None
            assert pdf_artifact["mime_type"] == "application/pdf"


def test_incremental_indexing_and_change_detection():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            extractor = Extractor()
            scanner = ArtifactScanner(store, extractor)

            file_path = os.path.join(temp_dir, "note.txt")
            create_text_file(file_path, "version 1")

            first_result = scanner.index_folder(temp_dir)
            assert first_result["processed"] == 1
            assert first_result["skipped"] == 0

            second_result = scanner.index_folder(temp_dir)
            assert second_result["processed"] == 0
            assert second_result["skipped"] == 1

            create_text_file(file_path, "version 2")
            os.utime(file_path, None)

            third_result = scanner.index_folder(temp_dir)
            assert third_result["processed"] == 1
            assert third_result["skipped"] == 0


def test_corrupt_pdf_does_not_crash_indexing():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            corrupt_path = os.path.join(temp_dir, "corrupt.pdf")
            with open(corrupt_path, "wb") as handle:
                handle.write(b"%PDF-1.4\n%%EOF\n")

            result = scanner.index_folder(temp_dir)
            assert result["processed"] == 1
            assert result["errors"] == 0
            artifact = store.get_artifact_by_path(corrupt_path)
            assert artifact is not None
            assert artifact["mime_type"] == "application/pdf"
            assert artifact["extracted_text"] == ""


def test_force_reindex_false_preserves_skip_behavior():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            file_path = os.path.join(temp_dir, "note.txt")
            create_text_file(file_path, "same content")

            first = scanner.index_folder(temp_dir)
            assert first["processed"] == 1
            assert first["skipped"] == 0

            # Unchanged file + force_reindex=False must be skipped (existing behavior).
            second = scanner.index_folder(temp_dir, force_reindex=False)
            assert second["processed"] == 0
            assert second["skipped"] == 1


def test_force_reindex_true_processes_unchanged_artifact():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            file_path = os.path.join(temp_dir, "note.txt")
            create_text_file(file_path, "same content")

            first = scanner.index_folder(temp_dir)
            assert first["processed"] == 1

            skipped_pass = scanner.index_folder(temp_dir, force_reindex=False)
            assert skipped_pass["skipped"] == 1

            # Unchanged file + force_reindex=True must be reprocessed.
            forced = scanner.index_folder(temp_dir, force_reindex=True)
            assert forced["processed"] == 1
            assert forced["skipped"] == 0
            assert store.artifact_count() == 1


def test_progress_callback_receives_updates():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            for i in range(3):
                create_text_file(os.path.join(temp_dir, f"file{i}.txt"), f"content {i}")

            calls = []

            def cb(payload):
                # Make a defensive copy; the scanner must not expose file contents.
                calls.append(dict(payload))

            first = scanner.index_folder(temp_dir, progress_callback=cb)
            assert first["processed"] == 3

            # Called once per supported file, including the processed ones.
            assert len(calls) == 3
            for payload in calls:
                assert set(payload.keys()) == {
                    "processed",
                    "skipped",
                    "errors",
                    "current",
                    "total",
                }
                assert isinstance(payload["processed"], int)
                assert isinstance(payload["skipped"], int)
                assert isinstance(payload["errors"], int)
                assert isinstance(payload["current"], str)
                assert payload["total"] is None
            assert calls[-1]["processed"] == 3
            assert calls[-1]["skipped"] == 0

            # Second pass (no force) should report skipped files via the callback too.
            # Counts reset per job, so processed is 0 and skipped accumulates to 3.
            calls.clear()
            second = scanner.index_folder(temp_dir, progress_callback=cb)
            assert second["skipped"] == 3
            assert len(calls) == 3
            assert calls[-1]["skipped"] == 3
            assert calls[-1]["processed"] == 0


def test_progress_callback_failure_does_not_break_indexing():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            create_text_file(os.path.join(temp_dir, "a.txt"), "alpha")
            create_text_file(os.path.join(temp_dir, "b.txt"), "beta")

            def failing_cb(payload):
                raise RuntimeError("callback boom")

            # Exception raised by the callback must be swallowed; indexing proceeds.
            result = scanner.index_folder(temp_dir, progress_callback=failing_cb)
            assert result["processed"] == 2
            assert result["errors"] == 0
            assert store.artifact_count() == 2
