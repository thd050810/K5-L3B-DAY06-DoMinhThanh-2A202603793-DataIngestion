import hashlib
import json
import re

from .models import Chunk, DocumentPage


def _make_chunk(page: DocumentPage, text: str, key: str) -> Chunk:
    identity = json.dumps(
        [page.document_id, page.source, page.page, key, text],
        ensure_ascii=False,
    )
    return Chunk(
        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=page.metadata.copy(),
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")

    if not page.text.strip():
        return []

    chunks = []
    for start in range(0, len(page.text), size - overlap):
        text = page.text[start:start + size]
        if text.strip():
            key = f"fixed:{size}:{overlap}:{start}"
            chunks.append(_make_chunk(page, text, key))
        if start + size >= len(page.text):
            break
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")

    text = page.text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    texts = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_size:
            if current:
                texts.append(current)
                current = ""
            texts.extend(
                paragraph[start:start + max_size]
                for start in range(0, len(paragraph), max_size)
                if paragraph[start:start + max_size].strip()
            )
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_size:
            current = candidate
        else:
            texts.append(current)
            current = paragraph

    if current:
        texts.append(current)

    return [
        _make_chunk(page, text, f"paragraph:{max_size}:{index}")
        for index, text in enumerate(texts)
    ]
