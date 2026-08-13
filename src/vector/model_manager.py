from __future__ import annotations

import logging
import os
import urllib.request
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODEL_ID = "all-MiniLM-L6-v2-onnx"
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".lifesearch", "models", MODEL_ID)

# Public ONNX model artifacts on HuggingFace Hub
MODEL_FILES = {
    "model.onnx": "https://huggingface.co/optimum/all-MiniLM-L6-v2/resolve/main/model.onnx",
    "tokenizer.json": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
}


class ModelManager:
    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.model_dir = model_dir or DEFAULT_MODEL_DIR

    def is_model_installed(self) -> bool:
        if not os.path.exists(self.model_dir):
            return False
        for fname in MODEL_FILES:
            fpath = os.path.join(self.model_dir, fname)
            if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
                return False
        return True

    def get_status(self) -> Dict[str, Any]:
        installed = self.is_model_installed()
        return {
            "model_id": MODEL_ID,
            "installed": installed,
            "model_dir": self.model_dir,
            "files": {
                fname: os.path.exists(os.path.join(self.model_dir, fname))
                for fname in MODEL_FILES
            },
        }

    def install_model(self) -> bool:
        """
        Explicit, user-initiated model download.
        NEVER called silently or automatically during app startup/indexing.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            for fname, url in MODEL_FILES.items():
                target_path = os.path.join(self.model_dir, fname)
                if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
                    logger.info(f"Downloading {fname} to {target_path}...")
                    urllib.request.urlretrieve(url, target_path)
            return self.is_model_installed()
        except Exception as exc:
            logger.error(f"Failed to install model {MODEL_ID}: {exc}")
            return False
