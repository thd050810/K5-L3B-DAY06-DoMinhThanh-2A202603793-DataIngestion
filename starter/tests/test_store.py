import os
import unittest
from uuid import uuid4

import psycopg

from starter.src.models import Chunk
from starter.src.store import PgVectorStore


class StoreValidationTests(unittest.TestCase):
    def test_dimensions_must_be_positive(self):
        for dimensions in [0, -1, True, 16001]:
            with self.assertRaises(ValueError):
                PgVectorStore("unused", dimensions)

    def test_invalid_vectors_fail_before_connecting(self):
        store = PgVectorStore("invalid", 2)
        chunk = Chunk("c1", "book", "book.pdf", 1, "text")
        for vectors in [[], [[1.0]], [[float("nan"), 1.0]], [[0.0, 0.0]]]:
            with self.assertRaises(ValueError):
                store.upsert("A", [chunk], vectors)
        with self.assertRaises(ValueError):
            store.search("A", [1.0, 0.0], top_k=0)


@unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL to test real pgvector")
class StoreIntegrationTests(unittest.TestCase):
    def test_reingestion_search_filter_and_strategy_isolation(self):
        url = os.environ["TEST_DATABASE_URL"]
        store = PgVectorStore(url, 2)
        store.create_schema()
        strategy = f"test:{uuid4().hex}"
        other = strategy + ":other"
        chunks = [
            Chunk("first", "book", "book.pdf", 1, "old", {"section": "ML"}),
            Chunk("second", "book", "book.pdf", 2, "other", {"section": "Math"}),
        ]
        try:
            store.upsert(strategy, chunks, [[1.0, 0.0], [0.0, 1.0]])
            store.upsert(other, chunks[:1], [[1.0, 0.0]])
            self.assertEqual(store.search(strategy, [1.0, 0.0])[0]["chunk_id"], "first")
            filtered = store.search(strategy, [1.0, 0.0], filters={"section": "Math"})
            self.assertEqual([hit["chunk_id"] for hit in filtered], ["second"])
            self.assertEqual(store.search(strategy, [1.0, 0.0], filters={"section": "absent"}), [])
            replacement = Chunk("new", "book", "book.pdf", 1, "new text", {"section": "ML"})
            store.upsert(strategy, [replacement], [[1.0, 0.0]])
            store.upsert(strategy, [replacement], [[1.0, 0.0]])
            self.assertEqual([hit["chunk_id"] for hit in store.search(strategy, [1.0, 0.0])], ["new"])
            self.assertEqual(store.search(other, [1.0, 0.0])[0]["chunk_id"], "first")
        finally:
            with psycopg.connect(url) as connection:
                connection.execute("DELETE FROM ingestion_chunks_2 WHERE strategy = ANY(%s)", ([strategy, other],))


if __name__ == "__main__":
    unittest.main()
