import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.errors import PdfReadError
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from starter.src.ingestion import count_pdf_pages, load_pdf, normalize_text


class IngestionTests(unittest.TestCase):
    def test_normalizes_unicode_and_horizontal_whitespace(self):
        self.assertEqual(normalize_text("  Tie\u0302\u0301ng\t  Vie\u0323\u0302t\u00a0 "), "Tiếng Việt")

    def test_preserves_lines_and_paragraph_boundaries(self):
        text = "  Heading\r\n first   line \r\n\r\n \t\r\n Second paragraph \rEnd  "
        self.assertEqual(normalize_text(text), "Heading\nfirst line\n\nSecond paragraph\nEnd")

    def test_blank_text_becomes_empty(self):
        self.assertEqual(normalize_text(" \r\n\t\u00a0"), "")
        self.assertEqual(normalize_text(""), "")

    def test_normalization_is_idempotent(self):
        text = " Tie\u0302\u0301ng\tVie\u0323\u0302t\r\n\r\n\r\n next "
        normalized = normalize_text(text)
        self.assertEqual(normalize_text(normalized), normalized)

    def test_loads_real_pdf_and_retains_blank_physical_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pdf"
            writer = PdfWriter()
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            for text in ["First   page", "", "Third page"]:
                page = writer.add_blank_page(width=300, height=300)
                if not text:
                    continue
                page[NameObject("/Resources")] = DictionaryObject({
                    NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
                })
                stream = DecodedStreamObject()
                stream.set_data(f"BT /F1 12 Tf 20 250 Td ({text}) Tj ET".encode("ascii"))
                page[NameObject("/Contents")] = stream
            writer.write(path)
            writer.close()

            self.assertEqual(count_pdf_pages(path), 3)
            pages = load_pdf(path, "sample", "https://example.com/sample.pdf")
            with self.assertNoLogs("starter.src.ingestion", level="WARNING"):
                selected = load_pdf(path, "sample", "sample.pdf", page_numbers=[1, 3])
            self.assertEqual([page.page for page in selected], [1, 3])
            self.assertEqual([page.text for page in selected], ["First page", "Third page"])
            for invalid in [[0], [4], [True], [1, 1]]:
                with self.assertRaises(ValueError):
                    load_pdf(path, "sample", "sample.pdf", page_numbers=invalid)

        self.assertEqual([page.page for page in pages], [1, 2, 3])
        self.assertEqual([page.text for page in pages], ["First page", "", "Third page"])
        self.assertEqual([page.metadata["is_empty"] for page in pages], [False, True, False])
        for page in pages:
            self.assertEqual(page.document_id, "sample")
            self.assertEqual(page.source, "https://example.com/sample.pdf")
            self.assertEqual(page.metadata["document_type"], "pdf")

    def test_missing_pdf_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_pdf(Path(directory) / "missing.pdf", "missing", "missing.pdf")

    def test_invalid_pdf_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.pdf"
            path.write_bytes(b"%PDF-1.7\nnot a valid PDF")
            with self.assertRaises(PdfReadError):
                load_pdf(path, "invalid", "invalid.pdf")


if __name__ == "__main__":
    unittest.main()
