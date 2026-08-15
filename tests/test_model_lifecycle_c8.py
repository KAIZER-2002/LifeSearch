"""
C8-3 focused tests: model installation & runtime readiness.

These tests verify the model lifecycle added in C8-3:
  - missing / present / valid / corrupt model detection
  - deterministic, LLM-free validation
  - idempotent + self-healing install (safe to run repeatedly)
  - no automatic download during search / indexing / engine load
  - CLI `model status` / `model install` behavior
  - no secret / absolute-path leakage in failure output

No large real model download is required: the real-model tests are skipped
when the model is not already installed locally (per C8-3 testing guidance).
"""

import os
import tempfile
from unittest import mock

import pytest

from src.cli import main as cli_main
from src.vector.embeddings import ONNXEmbeddingEngine
from src.vector.model_manager import MODEL_FILES, MODEL_ID, ModelManager

from src.model_lifecycle import (
    get_model_status,
    install_model,
    resolve_model_dir,
    validate_model,
)


# ---------------------------------------------------------------------------
# 1. Model missing -> detected correctly
# ---------------------------------------------------------------------------

def test_model_missing_detected():
    model_dir = tempfile.mkdtemp()
    manager = ModelManager(model_dir)
    assert manager.is_model_installed() is False

    status = validate_model(model_dir)
    assert status["installed"] is False
    assert status["valid"] is False


# ---------------------------------------------------------------------------
# 2. Model present (files exist) -> detected as installed
# ---------------------------------------------------------------------------

def test_model_present_detected(tmp_path):
    model_dir = str(tmp_path / "models" / MODEL_ID)
    os.makedirs(model_dir)
    for fname in MODEL_FILES:
        with open(os.path.join(model_dir, fname), "w", encoding="utf-8") as fh:
            fh.write("placeholder")

    manager = ModelManager(model_dir)
    assert manager.is_model_installed() is True

    # Files are present but are not a real model -> valid must be False.
    status = validate_model(model_dir)
    assert status["installed"] is True
    assert status["valid"] is False


# ---------------------------------------------------------------------------
# 3. Valid model validation succeeds (skipped if no real model locally)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not ModelManager().is_model_installed(),
    reason="real embedding model not installed locally; skip network-free check",
)
def test_valid_model_validation_succeeds():
    status = validate_model(None)
    assert status["installed"] is True
    assert status["valid"] is True
    assert status["error"] is None


# ---------------------------------------------------------------------------
# 4. Invalid / corrupt model -> safe failure (no raise, sanitized error)
# ---------------------------------------------------------------------------

def test_corrupt_model_safe_failure(tmp_path):
    model_dir = str(tmp_path / "models" / MODEL_ID)
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "model.onnx"), "wb") as fh:
        fh.write(b"this is not a valid onnx model")
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as fh:
        fh.write('{"not": "a real tokenizer"}')

    status = validate_model(model_dir)
    assert status["installed"] is True
    assert status["valid"] is False
    assert "error" in status
    # Error must not leak absolute filesystem paths or tracebacks.
    assert os.sep not in status["error"]
    assert "traceback" not in status["error"].lower()


# ---------------------------------------------------------------------------
# 5. Repeated install / status is idempotent (no re-download when valid)
# ---------------------------------------------------------------------------

def test_install_idempotent_no_redownload(tmp_path):
    model_dir = str(tmp_path / "mdl")
    downloads = []

    def fake_retrieve(url, target):
        downloads.append(target)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("x")

    with mock.patch(
        "src.vector.model_manager.urllib.request.urlretrieve", side_effect=fake_retrieve
    ):
        with mock.patch(
            "src.model_lifecycle.validate_model",
            return_value={"installed": True, "valid": True},
        ):
            assert install_model(model_dir) is True
            assert len(downloads) == len(MODEL_FILES)  # first run downloads

            downloads.clear()
            assert install_model(model_dir) is True
            assert downloads == [], "must not re-download when already valid"


# ---------------------------------------------------------------------------
# 5b. Corrupt model is repaired (re-downloaded) on install
# ---------------------------------------------------------------------------

def test_install_repairs_corrupt_model(tmp_path):
    model_dir = str(tmp_path / "mdl")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "model.onnx"), "w", encoding="utf-8") as fh:
        fh.write("corrupt-content")

    downloads = []

    def fake_retrieve(url, target):
        downloads.append(target)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("good")

    with mock.patch(
        "src.vector.model_manager.urllib.request.urlretrieve", side_effect=fake_retrieve
    ):
        with mock.patch(
            "src.model_lifecycle.validate_model",
            return_value={"installed": True, "valid": True},
        ):
            assert install_model(model_dir) is True
            assert downloads, "corrupt model must be re-downloaded"


# ---------------------------------------------------------------------------
# 6. Search / indexing does not unexpectedly download the model
# ---------------------------------------------------------------------------

def test_engine_load_does_not_download(tmp_path):
    with mock.patch("src.vector.model_manager.urllib.request.urlretrieve") as spy:
        engine = ONNXEmbeddingEngine(model_dir=str(tmp_path / "absent_model_dir_xyz"))
        assert engine._available is False
        assert engine.embed_text("anything") == []
    spy.assert_not_called()


def test_search_does_not_download_model(tmp_path):
    from src.artifacts.extractor import Extractor
    from src.artifacts.scanner import ArtifactScanner
    from src.artifacts.store import ArtifactStore
    from src.search.engine import SearchEngine

    db = str(tmp_path / "ls.db")
    note = os.path.join(str(tmp_path), "note.txt")
    with open(note, "w", encoding="utf-8") as fh:
        fh.write("qdrant searchable content for indexing test")

    with mock.patch("src.vector.model_manager.urllib.request.urlretrieve") as spy:
        store = ArtifactStore(db)
        scanner = ArtifactScanner(store, Extractor())
        scanner.index_folder(str(tmp_path))
        engine = SearchEngine(
            store, embedding_engine=ONNXEmbeddingEngine(model_dir=str(tmp_path / "absent_xyz"))
        )
        engine.search("qdrant")
        store.close()
    spy.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Embedding engine behavior when model is missing is clear
# ---------------------------------------------------------------------------

def test_engine_missing_model_clear_state(tmp_path):
    engine = ONNXEmbeddingEngine(model_dir=str(tmp_path / "absent_abc"))
    assert engine._available is False
    assert engine.dimension == 384
    assert engine.model_id == MODEL_ID
    assert engine.embed_text("anything") == []
    assert engine.embed_batch(["a", "b"]) == [[], []]


# ---------------------------------------------------------------------------
# 8. Existing model successfully produces embeddings (skipped if absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not ModelManager().is_model_installed(),
    reason="real embedding model not installed locally; skip network-free check",
)
def test_real_model_produces_embeddings():
    engine = ONNXEmbeddingEngine()
    vector = engine.embed_text("hello world")
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


# ---------------------------------------------------------------------------
# 9. Existing embedding dimension remains unchanged
# ---------------------------------------------------------------------------

def test_dimension_unchanged(tmp_path):
    engine = ONNXEmbeddingEngine(model_dir=str(tmp_path / "absent_def"))
    assert engine.dimension == 384
    assert engine.model_id == MODEL_ID


# ---------------------------------------------------------------------------
# 10. CLI model status / install behavior
# ---------------------------------------------------------------------------

def test_cli_model_status_missing(tmp_path, capsys):
    model_dir = str(tmp_path / "mdl")
    with mock.patch("src.model_lifecycle.resolve_model_dir", return_value=model_dir):
        rc = cli_main.main(["model", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Installed: False" in out
    assert MODEL_ID in out


def test_cli_model_install_success(tmp_path, capsys):
    model_dir = str(tmp_path / "mdl")
    with mock.patch("src.model_lifecycle.resolve_model_dir", return_value=model_dir):
        with mock.patch("src.model_lifecycle.install_model", return_value=True):
            rc = cli_main.main(["model", "install"])
    assert rc == 0
    assert "installed" in capsys.readouterr().out.lower()


def test_cli_model_install_failure(tmp_path, capsys):
    model_dir = str(tmp_path / "mdl")
    with mock.patch("src.model_lifecycle.resolve_model_dir", return_value=model_dir):
        with mock.patch("src.model_lifecycle.install_model", return_value=False):
            rc = cli_main.main(["model", "install"])
    assert rc == 1
    assert "failed" in capsys.readouterr().out.lower()


def test_cli_model_no_subcommand(tmp_path, capsys):
    model_dir = str(tmp_path / "mdl")
    with mock.patch("src.model_lifecycle.resolve_model_dir", return_value=model_dir):
        rc = cli_main.main(["model"])
    assert rc == 1


# ---------------------------------------------------------------------------
# 11. No secrets / path leakage in validation failure output
# ---------------------------------------------------------------------------

def test_no_path_leak_in_validation_error(tmp_path):
    model_dir = str(tmp_path / "mdl")
    os.makedirs(model_dir)
    with open(os.path.join(model_dir, "model.onnx"), "wb") as fh:
        fh.write(b"corrupt")
    with open(os.path.join(model_dir, "tokenizer.json"), "w", encoding="utf-8") as fh:
        fh.write("{}")

    status = validate_model(model_dir)
    assert status["valid"] is False
    err = status.get("error", "")
    assert os.sep not in err, "absolute path must not leak"
    assert "traceback" not in err.lower()


def test_get_model_status_composition(tmp_path):
    model_dir = str(tmp_path / "mdl")
    status = get_model_status(model_dir)
    assert status["model_id"] == MODEL_ID
    assert status["installed"] is False
    assert status["valid"] is False
    assert "model_dir" in status
    assert "files" in status
    assert set(status["files"].keys()) == set(MODEL_FILES.keys())


def test_resolve_model_dir_precedence(tmp_path):
    explicit = str(tmp_path / "explicit")
    env_dir = str(tmp_path / "env")
    assert resolve_model_dir(explicit) == explicit
    with mock.patch.dict(os.environ, {"LIFESEARCH_MODEL_DIR": env_dir}):
        assert resolve_model_dir(None) == env_dir
        # explicit still wins over env
        assert resolve_model_dir(explicit) == explicit
    assert resolve_model_dir(None) is None
