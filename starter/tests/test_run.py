import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from starter.src.models import DocumentPage
from starter.src.run import load_benchmark, load_manifest, write_json


class RunInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "input.json"
        self.pages = [DocumentPage("book", "book.pdf", 1, "Models learn from data.", {"section": "ML"})]
        self.questions = [
            {"id": f"q{index}", "split": "dev" if index < 15 else "test",
             "question": "What learns?", "expected_answer": "Models", "gold_source_id": "book",
             "gold_page": 1, "evidence_quote": "Models learn from data.",
             **({"filter": {"section": "ML"}} if index in [0, 1, 15] else {})}
            for index in range(20)
        ]

    def save(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def make_manifest(self):
        pdf_path = self.path.parent / "book.pdf"
        writer = PdfWriter()
        font = DictionaryObject({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        })
        for number in range(1, 31):
            page = writer.add_blank_page(width=300, height=300)
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 20 250 Td (Models learn from data. Page {number}.) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = stream
        writer.write(pdf_path)
        writer.close()
        self.save([{
            "document_id": "book", "source": "book.pdf", "path": "book.pdf",
            "content_pages": list(range(1, 31)), "metadata": {"section": "ML"},
            "page_metadata": {"2": {"section": "Math"}},
        }])

    def test_manifest_loads_real_pdf_with_verified_page_metadata(self):
        self.make_manifest()
        pages, report = load_manifest(self.path)
        self.assertEqual(len(pages), 30)
        self.assertEqual(pages[0].metadata["section"], "ML")
        self.assertEqual(pages[1].metadata["section"], "Math")
        self.assertEqual(pages[-1].page, 30)
        self.assertEqual(report[0]["physical_pages"], 30)
        self.assertEqual(len(report[0]["sha256"]), 64)

    def test_manifest_rejects_insufficient_content_pages(self):
        self.make_manifest()
        manifest = json.loads(self.path.read_text())
        manifest[0]["content_pages"] = [1, 2]
        self.save(manifest)
        with self.assertRaises(ValueError):
            load_manifest(self.path)

    @unittest.skipUnless(os.getenv("TEST_DATABASE_URL"), "Set TEST_DATABASE_URL for full pipeline integration")
    def test_full_cli_outputs_real_traces_and_summary(self):
        self.make_manifest()
        benchmark = self.path.parent / "benchmark.json"
        for question in self.questions:
            question["alternative_gold"] = [
                {"gold_source_id": "book", "gold_page": page, "evidence_quote": "Models learn from data."}
                for page in range(2, 31)
            ]
        write_json(benchmark, self.questions)
        output = self.path.parent / "results"
        environment = {**os.environ, "DATABASE_URL": os.environ["TEST_DATABASE_URL"]}
        result = subprocess.run([
            sys.executable, "-m", "starter.src.run", "--manifest", str(self.path),
            "--benchmark", str(benchmark), "--output", str(output), "--split", "all",
        ], cwd=Path(__file__).resolve().parents[2], env=environment, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        folder = next(output.iterdir())
        summary = json.loads((folder / "summary.json").read_text())
        self.assertEqual(summary["C"]["test"]["hit_at_3"], 1.0)
        self.assertEqual(summary["A"]["dev"]["questions"], 15)
        self.assertEqual(len((folder / "traces.jsonl").read_text().splitlines()), 60)

    def test_accepts_valid_benchmark(self):
        self.save(self.questions)
        self.assertEqual(load_benchmark(self.path, self.pages), self.questions)

    def test_filter_must_allow_at_least_one_gold_source_page(self):
        self.pages.append(DocumentPage("other", "other.pdf", 1, "Other text", {"section": "Math"}))
        self.questions[0]["filter"] = {"section": "Math"}
        self.save(self.questions)
        with self.assertRaises(ValueError):
            load_benchmark(self.path, self.pages)

    def test_rejects_bad_ids_splits_evidence_pages_and_filters(self):
        for change in [
            {"id": "q1"}, {"split": "validation"}, {"evidence_quote": "Missing evidence"},
            {"evidence_quote": " "}, {"gold_page": 2}, {"filter": {"source": "book.pdf"}},
            {"filter": {"section": "absent"}}, {"expected_answer": ""},
        ]:
            with self.subTest(change=change):
                self.save([{**self.questions[0], **change}, *self.questions[1:]])
                with self.assertRaises(ValueError):
                    load_benchmark(self.path, self.pages)

    def test_rejects_insufficient_questions_or_filters(self):
        for questions in [self.questions[:19], [{k: v for k, v in q.items() if k != "filter"} for q in self.questions]]:
            self.save(questions)
            with self.assertRaises(ValueError):
                load_benchmark(self.path, self.pages)

    def test_rejects_missing_manifest_or_missing_content_pages(self):
        self.save([{"document_id": "book", "path": "missing.pdf", "source": "book.pdf"}])
        with self.assertRaises(ValueError):
            load_manifest(self.path)
        self.save([])
        with self.assertRaises(ValueError):
            load_manifest(self.path)

    def test_writes_valid_unicode_json(self):
        write_json(self.path, {"text": "Tiếng Việt", "score": None})
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), {"text": "Tiếng Việt", "score": None})


if __name__ == "__main__":
    unittest.main()
