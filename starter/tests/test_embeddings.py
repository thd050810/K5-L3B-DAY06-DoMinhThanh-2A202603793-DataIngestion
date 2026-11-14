import math
import unittest

from starter.src.embeddings import MyEmbedder


class EmbeddingValidationTests(unittest.TestCase):
    def test_invalid_batch_size_is_rejected(self):
        for batch_size in [0, -1]:
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    MyEmbedder(batch_size=batch_size)


class LocalEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = MyEmbedder(batch_size=2, device="cpu")

    def test_document_vectors_are_finite_normalized_and_384_dimensions(self):
        vectors = self.embedder.embed_documents(["Machine learning uses data.", "A cat sleeps."])
        self.assertEqual(self.embedder.dimensions, 384)
        self.assertEqual(len(vectors), 2)
        for vector in vectors:
            self.assertEqual(len(vector), 384)
            self.assertTrue(all(isinstance(value, float) and math.isfinite(value) for value in vector))
            self.assertAlmostEqual(sum(value * value for value in vector), 1.0, places=5)

    def test_batch_preserves_order_and_matches_query_encoding(self):
        texts = ["A dog plays outside.", "Linear algebra studies vectors.", "Fresh bread is warm."]
        vectors = self.embedder.embed_documents(texts)
        self.assertEqual(len(vectors), 3)
        for text, vector in zip(texts, vectors):
            query = self.embedder.embed_query(text)
            self.assertEqual(len(query), 384)
            for actual, expected in zip(vector, query):
                self.assertAlmostEqual(actual, expected, places=5)

    def test_semantically_related_text_ranks_higher(self):
        query = self.embedder.embed_query("What is machine learning?")
        vectors = self.embedder.embed_documents([
            "Machine learning trains models to learn patterns from data.",
            "The chef bakes a chocolate cake in the kitchen.",
        ])
        scores = [sum(a * b for a, b in zip(query, vector)) for vector in vectors]
        self.assertGreater(scores[0], scores[1])

    def test_empty_document_list_has_no_vectors(self):
        self.assertEqual(self.embedder.embed_documents([]), [])

    def test_blank_inputs_are_rejected(self):
        for text in ["", " \n\t"]:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    self.embedder.embed_documents(["Valid text", text])
                with self.assertRaises(ValueError):
                    self.embedder.embed_query(text)


if __name__ == "__main__":
    unittest.main()
