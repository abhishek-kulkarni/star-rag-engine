# Built-in generics are preferred in Python 3.9+ (PEP 585)

import io

import docx  # python-docx
import fitz  # PyMuPDF
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from app.config.settings import settings


class ParserService:
    """
    Handles extraction of text from various file formats and chunking strategies.
    Integrates Microsoft Presidio for PII sanitization as per Rule 2.5.
    """

    def __init__(self):
        self.analyzer = AnalyzerEngine(
            default_score_threshold=settings.PII_CONFIDENCE_THRESHOLD
        )
        self.anonymizer = AnonymizerEngine()

    def parse(self, content: bytes, filename: str, sanitize: bool = True) -> str:
        """Routes content to the appropriate parser based on file extension."""
        ext = filename.split(".")[-1].lower() if "." in filename else ""

        if ext == "pdf":
            return self.parse_pdf(content, sanitize)
        elif ext == "docx":
            return self.parse_docx(content, sanitize)
        elif ext == "txt":
            return self.parse_txt(content, sanitize)
        else:
            # Fallback: try to decode as UTF-8 text if unknown
            try:
                return self.parse_txt(content, sanitize)
            except Exception as e:
                raise ValueError(f"Unsupported file format: {ext}") from e

    def parse_pdf(self, content: bytes, sanitize: bool = True) -> str:
        """Extracts raw text from PDF bytes and optionally sanitizes PII."""
        text = ""
        with fitz.open(stream=content, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()

        if sanitize:
            return self.sanitize_text(text)
        return text

    def parse_docx(self, content: bytes, sanitize: bool = True) -> str:
        """Extracts raw text from DOCX bytes and optionally sanitizes PII."""
        doc = docx.Document(io.BytesIO(content))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])

        if sanitize:
            return self.sanitize_text(text)
        return text

    def parse_txt(self, content: bytes, sanitize: bool = True) -> str:
        """Extracts raw text from TXT bytes and optionally sanitizes PII."""
        text = content.decode("utf-8")

        if sanitize:
            return self.sanitize_text(text)
        return text

    def sanitize_text(self, text: str) -> str:
        """
        Masks sensitive PII (Names, Emails, Phones) using Microsoft Presidio.
        Required for security compliance before vectorization.
        """
        results = self.analyzer.analyze(
            text=text,
            entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION", "URL"],
            language="en",
        )

        anonymized_result = self.anonymizer.anonymize(
            text=text, analyzer_results=results
        )

        return anonymized_result.text

    def split_text(
        self,
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ) -> list[str]:
        """
        Principal-level Recursive Character Splitting.
        Prioritizes high-level semantic boundaries (paragraphs) and drills down
        to sentences or words only for oversized blocks.
        """
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]

        if not separators:
            return [text.strip()]

        if len(text) <= chunk_size:
            return [text.strip()]

        # Find the first separator in the hierarchy that exists in the text
        separator = separators[-1] if separators else ""
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                new_separators = []
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1 :]
                break

        # Split the text by the selected separator
        final_chunks = []
        parts = text.split(separator) if separator != "" else list(text)

        current_doc = ""
        for part in parts:
            # If current_doc + part fits, keep building
            if len(current_doc) + len(part) + len(separator) <= chunk_size:
                if current_doc:
                    current_doc += separator + part
                else:
                    current_doc = part
            else:
                # current_doc is full. Process it.
                if current_doc:
                    # If this block is somehow still > chunk_size, recurse
                    if len(current_doc) > chunk_size:
                        final_chunks.extend(
                            self.split_text(
                                current_doc, chunk_size, chunk_overlap, new_separators
                            )
                        )
                    else:
                        final_chunks.append(current_doc.strip())

                # Start the next block.
                # Handle overlap by taking the tail of the previous doc.
                if chunk_overlap > 0 and current_doc:
                    overlap_text = current_doc[-chunk_overlap:]
                    current_doc = overlap_text + separator + part
                else:
                    current_doc = part

        # Final cleanup
        if current_doc:
            if len(current_doc) > chunk_size:
                final_chunks.extend(
                    self.split_text(
                        current_doc, chunk_size, chunk_overlap, new_separators
                    )
                )
            else:
                final_chunks.append(current_doc.strip())

        return final_chunks


parser_service = ParserService()
