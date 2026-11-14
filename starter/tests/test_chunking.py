import unittest

from starter.src.chunking import by_paragraph, fixed_size
from starter.src.models import DocumentPage


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage("book", "book.pdf", 3, "abcdefghij", {"section": "ML"})

    def test_fixed_size_overlaps_without_redundant_tail(self):
        chunks = fixed_size(self.page, size=4, overlap=2)
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "cdef", "efgh", "ghij"])

    def test_fixed_size_keeps_short_tail(self):
        chunks = fixed_size(self.page, size=4, overlap=0)
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "efgh", "ij"])

    def test_paragraphs_are_grouped_without_splitting_single_line_breaks(self):
        self.page.text = "ab\ncd\n\nxy\n \nlast"
        chunks = by_paragraph(self.page, max_size=9)
        self.assertEqual([chunk.text for chunk in chunks], ["ab\ncd\n\nxy", "last"])

    def test_oversized_paragraph_is_split_without_losing_text(self):
        self.page.text = "ab\n\nabcdefghij\n\nxy"
        chunks = by_paragraph(self.page, max_size=4)
        self.assertEqual([chunk.text for chunk in chunks], ["ab", "abcd", "efgh", "ij", "xy"])

    def test_blank_pages_have_no_chunks(self):
        self.page.text = " \n\t "
        self.assertEqual(fixed_size(self.page, 4, 1), [])
        self.assertEqual(by_paragraph(self.page, 4), [])

    def test_invalid_sizes_and_overlaps_are_rejected(self):
        for size, overlap in [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)]:
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    fixed_size(self.page, size, overlap)
        for size in [0, -1]:
            with self.assertRaises(ValueError):
                by_paragraph(self.page, size)

    def test_source_metadata_and_stable_distinct_ids(self):
        for chunker in [lambda page: fixed_size(page, 4, 2), lambda page: by_paragraph(page, 4)]:
            chunks = chunker(self.page)
            self.assertEqual([c.chunk_id for c in chunks], [c.chunk_id for c in chunker(self.page)])
            self.assertEqual(len({c.chunk_id for c in chunks}), len(chunks))
            for chunk in chunks:
                self.assertEqual((chunk.document_id, chunk.source, chunk.page), ("book", "book.pdf", 3))
                self.assertEqual(chunk.metadata, {"section": "ML"})
            chunks[0].metadata["section"] = "changed"
            self.assertEqual(self.page.metadata, {"section": "ML"})
            self.assertEqual(chunks[1].metadata, {"section": "ML"})

    def test_ids_distinguish_strategies_and_changed_content(self):
        self.assertNotEqual(fixed_size(self.page, 4, 0)[0].chunk_id,
                            by_paragraph(self.page, 4)[0].chunk_id)
        original_id = fixed_size(self.page, 4, 0)[0].chunk_id
        self.page.text = "XXXXXXXXXX"
        self.assertNotEqual(fixed_size(self.page, 4, 0)[0].chunk_id, original_id)


if __name__ == "__main__":
    unittest.main()
