import math
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if type(dimensions) is not int or not 1 <= dimensions <= 16000:
            raise ValueError("dimensions must be an integer between 1 and 16000")
        self.database_url = database_url
        self.dimensions = dimensions
        self._table = f"ingestion_chunks_{dimensions}"

    def _vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions or not all(math.isfinite(value) for value in vector):
            raise ValueError("Vector must have the configured dimensions and finite values")
        if not any(vector):
            raise ValueError("Cosine similarity requires a nonzero vector")
        return "[" + ",".join(str(float(value)) for value in vector) + "]"

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
            """)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have one vector")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a strategy")
        rows = [
            (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
             chunk.text, Jsonb(chunk.metadata), self._vector_literal(vector))
            for chunk, vector in zip(chunks, vectors)
        ]
        if not rows:
            return
        document_ids = sorted({chunk.document_id for chunk in chunks})
        with psycopg.connect(self.database_url, connect_timeout=10) as connection:
            connection.execute(
                f"DELETE FROM {self._table} WHERE strategy = %s AND document_id = ANY(%s)",
                (strategy, document_ids),
            )
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""INSERT INTO {self._table}
                        (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                    rows,
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        vector = self._vector_literal(query_vector)
        with psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=10) as connection:
            rows = connection.execute(
                f"""SELECT chunk_id, document_id, source, page, text, metadata,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self._table}
                    WHERE strategy = %s AND metadata @> %s
                    ORDER BY embedding <=> %s::vector, chunk_id
                    LIMIT %s""",
                (vector, strategy, Jsonb(filters or {}), vector, top_k),
            ).fetchall()
        return rows
