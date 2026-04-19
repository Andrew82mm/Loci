import os
import chromadb
from chromadb.utils import embedding_functions
from llm_memory.config import EMBEDDING_MODEL
from llm_memory.colors import log_rag


_META_FILENAME = "embedding_model.txt"


class VectorStore:
    """Thin wrapper around a Chroma persistent collection."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)

        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="memory_vault",
            embedding_function=self.embedding_func,
        )
        self._check_model_drift()

    def _meta_path(self) -> str:
        return os.path.join(self.db_path, _META_FILENAME)

    def _check_model_drift(self) -> bool:
        """Returns True if reindex was triggered due to model change."""
        meta = self._meta_path()
        if os.path.exists(meta):
            with open(meta) as f:
                stored = f.read().strip()
            if stored != EMBEDDING_MODEL:
                log_rag(f"Модель эмбеддинга изменилась ({stored} → {EMBEDDING_MODEL}), запускаю переиндексацию.")
                self.reset()
                return True
        with open(meta, "w") as f:
            f.write(EMBEDDING_MODEL)
        return False

    def reset(self) -> None:
        self.client.delete_collection("memory_vault")
        self.collection = self.client.get_or_create_collection(
            name="memory_vault",
            embedding_function=self.embedding_func,
        )

    def reload_client(self) -> None:
        """Recreate the Chroma client after an external restore of the DB directory."""
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="memory_vault",
            embedding_function=self.embedding_func,
        )

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]) -> None:
        self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    def delete_by_source(self, source: str) -> None:
        try:
            existing = self.collection.get(where={"source": source})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

    def query(self, query_text: str, n_results: int = 5) -> dict:
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["distances", "metadatas", "documents"],
        )

    def count(self) -> int:
        return self.collection.count()
