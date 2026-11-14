import logging
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from .models import DocumentPage


logger = logging.getLogger(__name__)


def count_pdf_pages(path: Path) -> int:
    with path.open("rb") as pdf_file:
        return len(PdfReader(pdf_file).pages)


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    page_numbers: list[int] | None = None,
) -> list[DocumentPage]:
    pages = []
    with path.open("rb") as pdf_file:
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        selected = list(range(1, total_pages + 1)) if page_numbers is None else page_numbers
        if any(type(number) is not int or not 1 <= number <= total_pages for number in selected):
            raise ValueError(f"Page numbers must be integers between 1 and {total_pages}")
        if len(set(selected)) != len(selected):
            raise ValueError("Page numbers must be unique")
        for page_number in sorted(selected):
            pdf_page = reader.pages[page_number - 1]
            text = normalize_text(pdf_page.extract_text() or "")
            if not text:
                logger.warning("No text extracted from %s, page %s", path, page_number)
            pages.append(DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"document_type": "pdf", "is_empty": not text},
            ))

    return pages


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
