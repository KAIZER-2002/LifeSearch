"""
Model lifecycle / runtime readiness for LifeSearch (C8-3).

This module is a thin, read-only *extension* of the existing model
architecture (src.vector.model_manager.ModelManager and
src.vector.embeddings.ONNXEmbeddingEngine). It does NOT redesign them.

It adds the smallest production-ready layer needed for a fresh install:

  - detect whether the required embedding model exists (ModelManager)
  - validate that it actually loads before use (ONNXEmbeddingEngine load)
  - an explicit, idempotent install/download operation
  - safe handling of missing / corrupt model files
  - model location configurable via LIFESEARCH_MODEL_DIR or constructor

Design constraints honored:
  - No automatic download during normal search / indexing / startup.
    install_model() is ONLY ever called by an explicit CLI command.
  - Validation never raises; failures degrade to a safe, sanitized result.
  - No secrets or absolute filesystem paths are surfaced in error text.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from src.vector.embeddings import ONNXEmbeddingEngine
from src.vector.model_manager import MODEL_FILES, MODEL_ID, ModelManager

logger = logging.getLogger(__name__)

# Configurable model location (consistent with LIFESEARCH_DB for the DB).
MODEL_DIR_ENV = "LIFESEARCH_MODEL_DIR"


def resolve_model_dir(explicit: Optional[str] = None) -> Optional[str]:
    """Resolve the model directory.

    Precedence: explicit argument > LIFESEARCH_MODEL_DIR env > ModelManager default.
    Returns None when neither is set, letting ModelManager apply its own default.
    """
    if explicit:
        return explicit
    env = os.environ.get(MODEL_DIR_ENV)
    return env or None


def validate_model(model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Genuinely attempt to load the model; report validity safely.

    Reuses the existing ONNXEmbeddingEngine load path as the source of truth
    for "is this model actually usable". Never raises.

    Returns a sanitized dict:
        {"installed": bool, "valid": bool, "error": <safe string or None>}
    The error string intentionally omits absolute paths / tracebacks so it is
    safe to surface through CLI and HTTP status surfaces.
    """
    manager = ModelManager(resolve_model_dir(model_dir))
    if not manager.is_model_installed():
        return {"installed": False, "valid": False, "error": "model files are missing"}

    try:
        # Real validation: a working engine is one that successfully loads.
        engine = ONNXEmbeddingEngine(model_dir=manager.model_dir)
        if engine._available:
            return {"installed": True, "valid": True, "error": None}
        return {"installed": True, "valid": False, "error": "model failed to load"}
    except Exception:
        # Defensive: ONNXEmbeddingEngine is expected to swallow load errors
        # internally, but never leak internals if it does not.
        return {"installed": True, "valid": False, "error": "model failed to load"}


def get_model_status(model_dir: Optional[str] = None) -> Dict[str, Any]:
    """Compose a clear installed/missing/invalid status report."""
    manager = ModelManager(resolve_model_dir(model_dir))
    installed = manager.is_model_installed()
    valid = validate_model(model_dir).get("valid", False) if installed else False
    return {
        "model_id": MODEL_ID,
        "installed": installed,
        "valid": valid,
        "model_dir": manager.model_dir,
        "files": {
            fname: os.path.exists(os.path.join(manager.model_dir, fname))
            for fname in MODEL_FILES
        },
    }


def install_model(model_dir: Optional[str] = None) -> bool:
    """Explicit, user-initiated model download. Idempotent and self-healing.

    Behavior:
      - Already installed AND valid      -> no download, return True.
      - Present but invalid/corrupt      -> remove + re-download, then validate.
      - Missing                           -> download, then validate.

    This is NEVER called silently during app startup, indexing, or search.
    Returns True only when the model installs AND validates successfully.
    """
    manager = ModelManager(resolve_model_dir(model_dir))

    # Already valid: do not re-download unnecessarily.
    if manager.is_model_installed() and validate_model(model_dir).get("valid"):
        logger.info("Model %s already installed and valid; skipping download.", MODEL_ID)
        return True

    # Repair: remove any existing (possibly corrupt) files before re-download.
    if manager.is_model_installed():
        for fname in MODEL_FILES:
            path = os.path.join(manager.model_dir, fname)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

    # Delegate the actual download to the existing ModelManager.
    if not manager.install_model():
        return False

    return bool(validate_model(model_dir).get("valid", False))
