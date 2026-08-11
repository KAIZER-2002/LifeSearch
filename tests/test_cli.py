import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.artifacts.extractor import Extractor
from src.artifacts.scanner import ArtifactScanner
from src.artifacts.store import ArtifactStore


def create_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def test_cli_index_search_status():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        artifact_dir = os.path.join(temp_dir, "workspace")
        os.makedirs(artifact_dir, exist_ok=True)
        create_text_file(os.path.join(artifact_dir, "todo.txt"), "buy milk")

        env = os.environ.copy()
        env["LIFESEARCH_DB"] = db_path

        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "index", artifact_dir],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "Indexed 1 files" in result.stdout

        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "status"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "Artifacts: 1 available" in result.stdout

        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "search", "milk"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "todo.txt" in result.stdout


def test_cli_open_prints_path_in_dry_run():
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "lifesearch.db")
        artifact_dir = os.path.join(temp_dir, "workspace")
        os.makedirs(artifact_dir, exist_ok=True)
        file_path = os.path.join(artifact_dir, "todo.txt")
        create_text_file(file_path, "buy milk")
        with ArtifactStore(db_path) as store:
            extractor = Extractor()
            scanner = ArtifactScanner(store, extractor)
            scanner.index_folder(artifact_dir)

            artifact = store.get_artifact_by_path(file_path)
            assert artifact is not None

            env = os.environ.copy()
            env["LIFESEARCH_DB"] = db_path
            env["LIFESEARCH_DRY_RUN"] = "1"

            result = subprocess.run(
                [sys.executable, "-m", "src.cli.main", "open", str(artifact["id"])],
                capture_output=True,
                text=True,
                env=env,
            )
            assert result.returncode == 0
            assert file_path in result.stdout

        env = os.environ.copy()
        env["LIFESEARCH_DB"] = db_path
        env["LIFESEARCH_DRY_RUN"] = "1"

        result = subprocess.run(
            [sys.executable, "-m", "src.cli.main", "open", str(artifact["id"])],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert file_path in result.stdout
