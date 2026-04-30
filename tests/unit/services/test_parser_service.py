from app.services.parser_service import parser_service


def test_split_text_simple():
    text = "hello world"
    chunks = parser_service.split_text(text, chunk_size=5, chunk_overlap=0)
    assert chunks == ["hello", "world"]


def test_split_text_paragraphs():
    text = "Paragraph 1\n\nParagraph 2"
    # Disable overlap to verify clean split
    chunks = parser_service.split_text(text, chunk_size=15, chunk_overlap=0)
    assert "Paragraph 1" in chunks
    assert "Paragraph 2" in chunks


def test_split_text_sentences():
    text = "Sentence one. Sentence two."
    chunks = parser_service.split_text(text, chunk_size=15)
    assert any("Sentence one" in c for c in chunks)
    assert any("Sentence two" in c for c in chunks)


def test_split_text_no_overlap():
    text = "aaaaabbbbb"
    chunks = parser_service.split_text(text, chunk_size=5, chunk_overlap=0)
    assert chunks == ["aaaaa", "bbbbb"]


def test_sanitize_text():
    text = "My name is John Doe and my email is john.doe@example.com"
    sanitized = parser_service.sanitize_text(text)
    assert "John Doe" not in sanitized
    assert "john.doe@example.com" not in sanitized
    assert "<PERSON>" in sanitized or "<EMAIL_ADDRESS>" in sanitized


def test_split_text_small_string():
    """Verify that text smaller than chunk_size returns as a single chunk."""
    text = "Short text"
    chunks = parser_service.split_text(text, chunk_size=100)
    assert chunks == ["Short text"]


def test_parse_pdf_integration(monkeypatch):
    """
    Verify PDF extraction logic.
    Mocks fitz (PyMuPDF) to avoid needing a physical file.
    """
    from unittest.mock import MagicMock

    # Mock PyMuPDF document and page
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Extracted PDF text."

    mock_doc = MagicMock()
    mock_doc.__enter__.return_value = [mock_page]

    monkeypatch.setattr("fitz.open", lambda **kwargs: mock_doc)

    # Test without sanitization first
    text = parser_service.parse_pdf(b"dummy_bytes", sanitize=False)
    assert text == "Extracted PDF text."

    # Test with sanitization
    text_sanitized = parser_service.parse_pdf(b"dummy_bytes", sanitize=True)
    assert "Extracted PDF text." in text_sanitized
