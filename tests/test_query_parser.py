from src.search.query_parser import QueryParser


def test_pdf_query_extracts_file_type():
    parser = QueryParser()
    res = parser.parse("Find the Qdrant PDF I downloaded around May.")
    assert "qdrant" in res.terms
    assert res.file_type == "pdf"
    assert res.has_temporal is True
    assert res.time_expression == "around May"
    assert res.intent == "search"


def test_screenshot_intent_detected():
    parser = QueryParser()
    res = parser.parse("Show me the screenshot where I had the MongoDB error.")
    assert "mongodb" in res.terms
    assert "error" in res.terms
    assert res.intent == "screenshot"
    assert res.file_type == "image"


def test_activity_intent_detected():
    parser = QueryParser()
    res = parser.parse("What files did I work on last Tuesday?")
    assert res.intent == "activity"
    assert res.has_temporal is True
    assert res.time_expression == "last Tuesday"


def test_filename_hint_extracted():
    parser = QueryParser()
    res = parser.parse("Which file did I edit right after reading retrieval.py?")
    assert res.filename_hint == "retrieval.py"
    assert res.file_type == "py"


def test_stopwords_removed():
    parser = QueryParser()
    res = parser.parse("Find notes that mention QLoRA.")
    assert "qlora" in res.topic_hints
    assert "find" not in res.terms
    assert "that" not in res.terms
