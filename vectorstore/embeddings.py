"""Local embedding generation via sentence-transformers (no API key required)."""

import logging
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around a sentence-transformers model for text embedding."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents for storage in the vector store."""
        embeddings = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search."""
        embedding = self._model.encode([text], show_progress_bar=False, convert_to_numpy=True)
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    model = EmbeddingModel()
    vec = model.embed_query("What are the side effects of ibuprofen?")
    logger.info("Embedding dimension: %d", model.dimension)
    logger.info("First 5 values: %s", vec[:5])
