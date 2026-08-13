import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore
from src.search.engine import SearchEngine


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def test_search_returns_artifacts_by_filename_and_content():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        with ArtifactStore(db_path) as store:
            scanner = ArtifactScanner(store, Extractor())

            create_text_file(os.path.join(temp_dir, "notes.txt"), "searchable content")
            create_text_file(os.path.join(temp_dir, "story.md"), "a markdown story")

            scanner.index_folder(temp_dir)
            engine = SearchEngine(store)

            results = engine.search("searchable")
            assert len(results) == 1
            assert results[0]["file_name"] == "notes.txt"

            path_results = engine.search("story")
            assert len(path_results) == 1
            assert path_results[0]["file_name"] == "story.md"

            file_results = engine.search("notes")
            assert len(file_results) == 1
            assert file_results[0]["file_name"] == "notes.txt"
