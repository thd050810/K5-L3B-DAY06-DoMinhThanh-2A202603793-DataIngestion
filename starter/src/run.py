import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .evaluation import evaluate_query, evidence_text, gold_evidence, summarize
from .ingestion import count_pdf_pages, load_pdf
from .models import DocumentPage


STARTER = Path(__file__).resolve().parents[1]
FILTER_FIELDS = {"section", "year", "version", "category", "audience", "language", "effective_date", "document_type"}
logger = logging.getLogger(__name__)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> tuple[list[DocumentPage], list[dict]]:
    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest must be a nonempty JSON array")
    pages = []
    report = []
    seen = set()
    for document in documents:
        for field in ["document_id", "source", "path"]:
            if not isinstance(document.get(field), str) or not document[field].strip():
                raise ValueError(f"Manifest requires a nonempty {field}")
        document_id = document["document_id"]
        if document_id in seen:
            raise ValueError(f"Duplicate document_id: {document_id}")
        seen.add(document_id)
        selected = document.get("content_pages")
        if not isinstance(selected, list) or not selected:
            raise ValueError(f"{document_id}: explicitly list verified content_pages")
        if any(type(number) is not int or number < 1 for number in selected) or len(set(selected)) != len(selected):
            raise ValueError(f"{document_id}: content_pages must contain unique positive integers")
        metadata = document.get("metadata", {})
        overrides = document.get("page_metadata", {})
        if not isinstance(metadata, dict) or not isinstance(overrides, dict):
            raise ValueError("metadata and page_metadata must be JSON objects")
        pdf_path = (path.parent / document["path"]).resolve()
        physical_pages = count_pdf_pages(pdf_path)
        if max(selected) > physical_pages:
            raise ValueError(f"{document_id}: content_pages exceeds PDF page count")
        document_pages = load_pdf(pdf_path, document_id, document["source"], page_numbers=selected)
        for page in document_pages:
            if not page.text:
                raise ValueError(f"{document_id}: selected page {page.page} has no text; check OCR or content_pages")
            page_metadata = overrides.get(str(page.page), {})
            if not isinstance(page_metadata, dict):
                raise ValueError("Each page_metadata value must be a JSON object")
            page.metadata = {**metadata, **page_metadata, **page.metadata}
            pages.append(page)
        report.append({
            **document,
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "physical_pages": physical_pages,
            "checked_pages": sorted(selected),
            "empty_pages": [page.page for page in document_pages if not page.text],
        })
    if len(pages) < 30:
        raise ValueError(f"At least 30 verified content pages are required; found {len(pages)}")
    return sorted(pages, key=lambda page: (page.document_id, page.page)), report


def load_benchmark(path: Path, pages: list[DocumentPage]) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Benchmark must be a JSON array")
    source_pages = {(page.document_id, page.page): page for page in pages}
    seen = set()
    for question in questions:
        for field in ["id", "question", "expected_answer"]:
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"Question requires a nonempty {field}")
        if question["id"] in seen:
            raise ValueError(f"Duplicate question ID: {question['id']}")
        seen.add(question["id"])
        if question.get("split") not in {"dev", "test"}:
            raise ValueError("Question split must be dev or test")
        gold = gold_evidence(question)
        for entry in gold:
            page = source_pages.get((entry["gold_source_id"], entry["gold_page"]))
            if page is None or evidence_text(entry["evidence_quote"]) not in evidence_text(page.text):
                raise ValueError(f"{question['id']}: gold evidence is absent from the selected source/page")
        filters = question.get("filter", {})
        if not isinstance(filters, dict) or not set(filters) <= FILTER_FIELDS:
            raise ValueError(f"Filters must use fields: {sorted(FILTER_FIELDS)}")
        if any(type(value) not in {str, int, float, bool} for value in filters.values()):
            raise ValueError("Filter values must be non-null JSON scalars")
        if filters and not any(
            all(source_pages[(entry["gold_source_id"], entry["gold_page"])].metadata.get(key) == value
                for key, value in filters.items())
            for entry in gold
        ):
            raise ValueError(f"{question['id']}: filter excludes every accepted gold source/page")
    if sum(q["split"] == "dev" for q in questions) < 15 or sum(q["split"] == "test" for q in questions) < 5:
        raise ValueError("Benchmark requires at least 15 dev and 5 test questions")
    filtered = [q for q in questions if q.get("filter")]
    if len(filtered) < 3 or not any(q["split"] == "test" for q in filtered):
        raise ValueError("Benchmark requires at least 3 filtered questions, including 1 test question")
    return questions


def main() -> None:
    load_dotenv(STARTER / ".env")
    parser = argparse.ArgumentParser(description="Ingest PDFs and benchmark A/B/C using local embeddings and pgvector")
    parser.add_argument("--manifest", type=Path, default=STARTER / "data/manifest.json")
    parser.add_argument("--benchmark", type=Path, default=STARTER / "data/benchmark.json")
    parser.add_argument("--output", type=Path, default=STARTER / "results")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--fixed-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--paragraph-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()
    if args.fixed_size <= 0 or not 0 <= args.overlap < args.fixed_size:
        parser.error("Require fixed-size > 0 and 0 <= overlap < fixed-size")
    if args.paragraph_size <= 0 or args.batch_size <= 0:
        parser.error("paragraph-size and batch-size must be positive")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("Set DATABASE_URL in starter/.env or the environment")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pages, corpus = load_manifest(args.manifest)
    benchmark = load_benchmark(args.benchmark, pages)
    questions = [q for q in benchmark if args.split == "all" or q["split"] == args.split]

    from .embeddings import MyEmbedder
    from .store import PgVectorStore

    logger.info("Loaded %s content pages and %s %s questions", len(pages), len(questions), args.split)
    embedder = MyEmbedder(args.model, batch_size=args.batch_size, device=args.device)
    store = PgVectorStore(database_url, embedder.dimensions)
    store.create_schema()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {
        "run_id": run_id, "model": embedder.model_name, "dimensions": embedder.dimensions,
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(args.benchmark.read_bytes()).hexdigest(),
        "search": "exact cosine", "embedding_normalized": True,
    })
    write_json(output / "corpus.json", corpus)
    write_json(output / "benchmark.json", benchmark)
    write_json(output / "pages.json", [asdict(page) for page in pages])
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    traces = []
    resources = {}
    coherence = []
    paragraph_chunks = None
    paragraph_vectors = None
    with (output / "traces.jsonl").open("w", encoding="utf-8") as trace_file:
        for strategy in ["A", "B", "C"]:
            started = perf_counter()
            if strategy == "C":
                chunks, vectors = paragraph_chunks, paragraph_vectors
                embedding_seconds = 0.0
            else:
                chunks = [
                    chunk for page in pages
                    for chunk in (fixed_size(page, args.fixed_size, args.overlap) if strategy == "A"
                                  else by_paragraph(page, args.paragraph_size))
                ]
                embedding_started = perf_counter()
                vectors = embedder.embed_documents([chunk.text for chunk in chunks])
                embedding_seconds = perf_counter() - embedding_started
                if strategy == "B":
                    paragraph_chunks, paragraph_vectors = chunks, vectors
            namespace = f"{run_id}:{strategy}"
            store.upsert(namespace, chunks, vectors)
            resources[strategy] = {
                "namespace": namespace, "chunks": len(chunks),
                "characters": sum(len(chunk.text) for chunk in chunks),
                "embedding_seconds": embedding_seconds,
                "ingestion_seconds": perf_counter() - started,
                "embedding_reused_from": "B" if strategy == "C" else None,
                "api_cost": 0,
            }
            write_json(output / f"chunks_{strategy}.json", [asdict(chunk) for chunk in chunks])
            first_chunks = {}
            for chunk in chunks:
                first_chunks.setdefault((chunk.document_id, chunk.page), chunk)
            for page in pages[:10]:
                chunk = first_chunks[(page.document_id, page.page)]
                coherence.append({
                    "strategy": strategy, "position": "first text on page",
                    "chunk": asdict(chunk), "coherence_score": None, "reason": "",
                })
            for question in questions:
                filters = question.get("filter") if strategy == "C" else None
                searched = perf_counter()
                hits = store.search(namespace, query_vectors[question["id"]], top_k=3, filters=filters)
                trace = {
                    **evaluate_query(question, hits), "strategy": strategy,
                    "applied_filter": filters or {}, "search_seconds": perf_counter() - searched,
                }
                traces.append(trace)
                trace_file.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
                trace_file.flush()
            logger.info("Strategy %s: %s chunks evaluated", strategy, len(chunks))
    write_json(output / "summary.json", summarize(traces))
    write_json(output / "resources.json", resources)
    write_json(output / "coherence_review.json", coherence)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
