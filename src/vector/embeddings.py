from __future__ import annotations

import logging
import os
from typing import List, Optional, Protocol, Tuple

import numpy as np

from .model_manager import MODEL_ID, ModelManager

logger = logging.getLogger(__name__)


class EmbeddingEngine(Protocol):
    """Model-agnostic local text embedding protocol."""

    @property
    def model_id(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_text(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...


class NullEmbeddingEngine:
    """Fallback embedding engine used when local model is unavailable."""

    @property
    def model_id(self) -> str:
        return "null"

    @property
    def dimension(self) -> int:
        return 0

    def embed_text(self, text: str) -> List[float]:
        return []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[] for _ in texts]


class ONNXEmbeddingEngine:
    """Local CPU-based text embedding engine powered by ONNX Runtime."""

    def __init__(self, model_dir: Optional[str] = None) -> None:
        self.manager = ModelManager(model_dir)
        self._model_dir = self.manager.model_dir
        self._available = False
        self._session = None
        self._tokenizer = None
        self._dim = 384
        self._model_id = MODEL_ID

        if self.manager.is_model_installed():
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                onnx_path = os.path.join(self._model_dir, "model.onnx")
                tok_path = os.path.join(self._model_dir, "tokenizer.json")

                # Suppress ONNX runtime verbose logs
                opts = ort.SessionOptions()
                opts.log_severity_level = 3

                self._session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
                self._tokenizer = Tokenizer.from_file(tok_path)
                self._tokenizer.enable_truncation(max_length=512)
                self._tokenizer.enable_padding(length=512)
                self._available = True
            except Exception as exc:
                logger.warning(f"Failed to initialize ONNX embedding engine: {exc}")
                self._available = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip() or not self._available:
            return []
        res = self.embed_batch([text])
        return res[0] if res else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts or not self._available or self._session is None or self._tokenizer is None:
            return [[] for _ in texts]

        try:
            encoded_list = self._tokenizer.encode_batch(texts)
            input_ids = np.array([e.ids for e in encoded_list], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded_list], dtype=np.int64)

            # Some HuggingFace ONNX exports require token_type_ids
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            inp_names = [inp.name for inp in self._session.get_inputs()]
            if "token_type_ids" in inp_names:
                inputs["token_type_ids"] = np.zeros_like(input_ids)

            outputs = self._session.run(None, inputs)
            # Output tensor shape: [batch, seq_len, 384]
            last_hidden_state = outputs[0]

            # Mean pooling over attention mask
            mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
            sum_embeddings = np.sum(last_hidden_state * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 normalization
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-9, a_max=None)
            normalized = mean_pooled / norms

            return normalized.tolist()
        except Exception as exc:
            logger.error(f"Error in ONNX embedding batch: {exc}")
            return [[] for _ in texts]
