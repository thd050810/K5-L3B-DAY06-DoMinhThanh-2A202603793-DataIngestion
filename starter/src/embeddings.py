from typing import Protocol

from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class MyEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        dimensions = self._model.get_embedding_dimension()
        if dimensions is None or dimensions <= 0:
            raise ValueError("Model must provide a positive embedding dimension")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Texts must not be blank")

        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    def embed_query(self, question: str) -> list[float]:
        return self.embed_documents([question])[0]
