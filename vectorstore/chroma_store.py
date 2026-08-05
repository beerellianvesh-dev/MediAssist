"""ChromaDB persistent vector store for drug label chunks."""

import logging
import sys
import uuid
from pathlib import Path

import chromadb

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import settings
from vectorstore.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ChromaStore:
    """Wraps a ChromaDB collection with embedding + citation-friendly metadata.

    Connects to a remote ChromaDB server (docker-compose service) when
    `settings.chroma_host` is set, otherwise falls back to a local
    on-disk persistent client — useful for running without Docker.
    """

    def __init__(self, collection_name: str | None = None, embedding_model: EmbeddingModel | None = None):
        self.collection_name = collection_name or settings.chroma_collection_name
        self.embedder = embedding_model or EmbeddingModel()

        # Anonymized telemetry is disabled: this chromadb/posthog version pairing
        # throws (harmless) errors on every call, and we don't need the analytics.
        client_settings = chromadb.Settings(anonymized_telemetry=False)

        if settings.chroma_host:
            logger.info("Connecting to remote ChromaDB at %s:%s", settings.chroma_host, settings.chroma_port)
            self._client = chromadb.HttpClient(
                host=settings.chroma_host, port=settings.chroma_port, settings=client_settings
            )
        else:
            persist_dir = Path(settings.chroma_persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Using local persistent ChromaDB at %s", persist_dir)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=client_settings)

        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list[dict]) -> int:
        """Embed and add a batch of chunk dicts (as produced by chunker.Chunk.to_dict()).

        Deduplicates by a stable id derived from (set_id, field, chunk_index)
        so re-ingesting the same source data does not create duplicates.
        """
        if not chunks:
            return 0

        ids, texts, metadatas = [], [], []
        for chunk in chunks:
            chunk_id = "_".join(
                str(chunk.get(k, "na")) for k in ("set_id", "field", "chunk_index")
            ) or str(uuid.uuid4())
            ids.append(chunk_id)
            texts.append(chunk["text"])
            metadatas.append(
                {
                    "drug_name": chunk.get("drug_name", ""),
                    "section_name": chunk.get("section_name", ""),
                    "source_url": chunk.get("source_url", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                }
            )

        existing = set(self._collection.get(ids=ids)["ids"])
        new_indices = [i for i, cid in enumerate(ids) if cid not in existing]
        if not new_indices:
            logger.info("All %d chunks already present, nothing to add", len(chunks))
            return 0

        new_ids = [ids[i] for i in new_indices]
        new_texts = [texts[i] for i in new_indices]
        new_metadatas = [metadatas[i] for i in new_indices]
        embeddings = self.embedder.embed_documents(new_texts)

        self._collection.add(
            ids=new_ids,
            embeddings=embeddings,
            documents=new_texts,
            metadatas=new_metadatas,
        )
        logger.info("Added %d new chunks (%d were duplicates)", len(new_ids), len(chunks) - len(new_ids))
        return len(new_ids)

    def similarity_search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Return the top_k most similar chunks to `query`, with metadata and distance."""
        top_k = top_k or settings.retrieval_top_k
        query_embedding = self.embedder.embed_query(query)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def list_sources(self) -> dict:
        """Return the unique set of drug names currently ingested, plus a total count."""
        records = self._collection.get(include=["metadatas"])
        drug_names = sorted({m.get("drug_name", "") for m in records["metadatas"] if m.get("drug_name")})
        return {"drugs": drug_names, "total_documents": len(records["ids"])}

    def get_collection_stats(self) -> dict:
        """Return basic stats about the collection for health/monitoring endpoints."""
        count = self._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
        }


if __name__ == "__main__":
    store = ChromaStore()
    logger.info("Stats: %s", store.get_collection_stats())
