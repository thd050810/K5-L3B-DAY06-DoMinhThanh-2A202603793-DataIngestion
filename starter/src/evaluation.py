import unicodedata
from typing import Any


def evidence_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def gold_evidence(question: dict[str, Any]) -> list[dict[str, Any]]:
    entries = [question, *question.get("alternative_gold", [])]
    for entry in entries:
        if not isinstance(entry.get("gold_source_id"), str) or not entry["gold_source_id"].strip():
            raise ValueError("Gold evidence requires a nonempty gold_source_id")
        if type(entry.get("gold_page")) is not int or entry["gold_page"] < 1:
            raise ValueError("Gold evidence requires a positive integer gold_page")
        if not isinstance(entry.get("evidence_quote"), str) or not evidence_text(entry["evidence_quote"]):
            raise ValueError("Gold evidence requires a nonempty evidence_quote")
    return entries


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    gold = gold_evidence(question)
    filters = question.get("filter") or {}
    ranked_hits = []
    for rank, hit in enumerate(hits[:3], start=1):
        text = evidence_text(hit["text"])
        matches = any(
            hit.get("document_id") == entry["gold_source_id"]
            and hit.get("page") == entry["gold_page"]
            and evidence_text(entry["evidence_quote"]) in text
            for entry in gold
        )
        filter_match = all(hit.get("metadata", {}).get(key) == value for key, value in filters.items())
        ranked_hits.append({**hit, "rank": rank, "evidence_match": matches, "filter_match": filter_match})

    return {
        "question": question,
        "split": question["split"],
        "hits": ranked_hits,
        "hit_at_1": bool(ranked_hits and ranked_hits[0]["evidence_match"]),
        "hit_at_3": any(hit["evidence_match"] for hit in ranked_hits),
        "filter_compliant": (
            bool(ranked_hits) and all(hit["filter_match"] for hit in ranked_hits)
            if filters else None
        ),
        "grounding_score": None,
        "grounding_reason": "",
        "metadata_correct": None,
        "metadata_reason": "",
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for trace in traces:
        key = (trace["strategy"], trace["split"])
        groups.setdefault(key, []).append(trace)

    summary = {}
    for (strategy, split), group in groups.items():
        count = len(group)
        hit1 = sum(trace["hit_at_1"] for trace in group)
        hit3 = sum(trace["hit_at_3"] for trace in group)
        filtered = [trace for trace in group if trace["filter_compliant"] is not None]
        compliant = sum(trace["filter_compliant"] for trace in filtered)
        summary.setdefault(strategy, {})[split] = {
            "questions": count,
            "hit_at_1_count": hit1,
            "hit_at_1": hit1 / count,
            "hit_at_3_count": hit3,
            "hit_at_3": hit3 / count,
            "filter_questions": len(filtered),
            "filter_compliant_count": compliant,
            "filter_compliance": compliant / len(filtered) if filtered else None,
        }
    return summary
