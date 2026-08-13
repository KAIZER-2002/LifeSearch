from src.vector.embeddings import NullEmbeddingEngine, ONNXEmbeddingEngine


def test_null_embedding_engine():
    engine = NullEmbeddingEngine()
    assert engine.model_id == "null"
    assert engine.dimension == 0
    assert engine.embed_text("test") == []
    assert engine.embed_batch(["a", "b"]) == [[], []]


def test_onnx_embedding_engine_uninstalled():
    # Points to empty non-existent model dir
    engine = ONNXEmbeddingEngine(model_dir="/tmp/non_existent_model_dir_xyz")
    assert engine._available is False
    assert engine.embed_text("sample") == []
