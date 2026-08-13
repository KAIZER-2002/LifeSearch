from src.vector.chunker import TextChunker, TextChunk


def test_chunker_basic_splitting():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    sample_text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    )

    chunks = chunker.chunk_text(sample_text, artifact_id=42, source_type="document_text")
    assert len(chunks) >= 2
    for idx, c in enumerate(chunks):
        assert c.artifact_id == 42
        assert c.chunk_index == idx
        assert c.chunk_id == f"42_c{idx}"
        assert c.source_type == "document_text"
        assert len(c.text) >= 10


def test_chunker_short_text_discarded():
    chunker = TextChunker(min_chunk_size=50)
    chunks = chunker.chunk_text("Too short text", artifact_id=1)
    assert chunks == []


def test_chunker_ocr_source_type():
    chunker = TextChunker(chunk_size=200, min_chunk_size=10)
    chunks = chunker.chunk_text("MongoDB connection refused localhost 27017", artifact_id=10, source_type="ocr_text")
    assert len(chunks) == 1
    assert chunks[0].source_type == "ocr_text"


def test_chunker_non_advancing_boundary_no_infinite_loop():
    """Regression: a sentence boundary within chunk_overlap of `start` must
    not pin the scan window (start=103 -> start=103) and spin forever.

    Reproduces the exact input/params that previously triggered the
    unbounded chunk growth in TextChunker.chunk_text.
    """
    import threading

    chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_size=10)
    text = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
    )

    result: dict = {}
    error: dict = {}

    def _run():
        try:
            result["chunks"] = chunker.chunk_text(text, artifact_id=42, source_type="document_text")
        except Exception as exc:  # pragma: no cover - defensive
            error["exc"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=5.0)

    # If the loop regresses to infinite, this assertion fails instead of
    # hanging the whole suite.
    assert not t.is_alive(), "chunk_text did not terminate (infinite-loop regression)"
    assert "exc" not in error, f"chunk_text raised: {error.get('exc')}"

    chunks = result["chunks"]
    assert len(chunks) >= 2, "expected the text to be split into multiple chunks"
    assert all(len(c.text) >= 10 for c in chunks), "no sub-minimum chunks should leak"
    assert chunks[0].start_char == 0, "first chunk should start at the beginning"
