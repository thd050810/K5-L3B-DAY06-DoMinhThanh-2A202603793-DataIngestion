import unittest

from starter.src.evaluation import evaluate_query, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.question = {
            "id": "q1", "split": "dev", "question": "What learns?",
            "expected_answer": "Models", "gold_source_id": "book", "gold_page": 2,
            "evidence_quote": "Models learn from data.",
        }
        self.hit = {
            "chunk_id": "c1", "document_id": "book", "source": "book.pdf", "page": 2,
            "text": "Models learn\n from data.", "score": 0.9, "metadata": {"section": "ML"},
        }

    def test_requires_correct_document_page_and_complete_evidence(self):
        for changes in [{"document_id": "other"}, {"page": 3}, {"text": "Models learn"}]:
            trace = evaluate_query(self.question, [{**self.hit, **changes}])
            self.assertFalse(trace["hit_at_1"])
            self.assertFalse(trace["hit_at_3"])

    def test_matches_normalized_evidence_and_preserves_trace(self):
        trace = evaluate_query(self.question, [self.hit])
        self.assertTrue(trace["hit_at_1"])
        self.assertTrue(trace["hit_at_3"])
        self.assertEqual(trace["hits"][0]["score"], 0.9)
        self.assertEqual(trace["question"], self.question)
        self.assertIsNone(trace["grounding_score"])

    def test_only_first_three_hits_count(self):
        miss = {**self.hit, "page": 1}
        trace = evaluate_query(self.question, [miss, self.hit])
        self.assertFalse(trace["hit_at_1"])
        self.assertTrue(trace["hit_at_3"])
        trace = evaluate_query(self.question, [miss, miss, miss, self.hit])
        self.assertFalse(trace["hit_at_3"])
        self.assertEqual(len(trace["hits"]), 3)
        self.assertFalse(evaluate_query(self.question, [])["hit_at_1"])

    def test_alternative_evidence_is_tied_to_its_source(self):
        self.question["alternative_gold"] = [{
            "gold_source_id": "second", "gold_page": 5, "evidence_quote": "Learning uses examples.",
        }]
        hit = {**self.hit, "document_id": "second", "page": 5, "text": "Learning uses examples."}
        self.assertTrue(evaluate_query(self.question, [hit])["hit_at_3"])
        hit["text"] = self.question["evidence_quote"]
        self.assertFalse(evaluate_query(self.question, [hit])["hit_at_3"])

    def test_blank_gold_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_query({**self.question, "evidence_quote": "  "}, [self.hit])

    def test_summary_separates_strategy_split_and_filter_compliance(self):
        question = {**self.question, "filter": {"section": "ML"}}
        success = {**evaluate_query(question, [self.hit]), "strategy": "C"}
        failure = {**evaluate_query(self.question, []), "strategy": "C"}
        test = {**evaluate_query({**question, "split": "test"}, [self.hit]), "strategy": "A"}
        summary = summarize([success, failure, test])
        self.assertEqual(summary["C"]["dev"]["questions"], 2)
        self.assertEqual(summary["C"]["dev"]["hit_at_1_count"], 1)
        self.assertEqual(summary["C"]["dev"]["hit_at_3"], 0.5)
        self.assertEqual(summary["C"]["dev"]["filter_compliance"], 1.0)
        self.assertEqual(summary["A"]["test"]["questions"], 1)
        self.assertEqual(summarize([]), {})

    def test_empty_results_do_not_count_as_filter_success(self):
        trace = evaluate_query({**self.question, "filter": {"section": "ML"}}, [])
        self.assertFalse(trace["filter_compliant"])


if __name__ == "__main__":
    unittest.main()
