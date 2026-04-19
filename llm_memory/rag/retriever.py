import os
import json
import math
from llm_memory.rag.vector import VectorStore
from llm_memory.rag.chunker import chunk_markdown
from llm_memory.storage.filesystem import StorageManager
from llm_memory.models import RetrievedChunk
from llm_memory.config import MIN_SIMILARITY
from llm_memory.colors import log_rag, log_warn


def _distance_to_score(distance: float) -> float:
    """Convert squared-L2 distance (all-MiniLM default) to cosine similarity proxy."""
    # For normalised vectors: cosine_sim = 1 - dist/2
    return max(0.0, 1.0 - distance / 2.0)


class RAGEngine:
    def __init__(self, storage: StorageManager, graph_index=None) -> None:
        self.storage = storage
        self.graph_index = graph_index  # injected to avoid circular imports
        db_path = os.path.join(storage.paths["system"], "chroma_db")
        self.vector = VectorStore(db_path)
        self._sync_index()

    # ── Indexing ──────────────────────────────────────────────────────────

    def _sync_index(self) -> None:
        dirs_to_scan = [
            self.storage.paths["knowledge"],
            self.storage.paths.get("knowledge_global", ""),
        ]
        count = 0
        for directory in dirs_to_scan:
            if not directory or not os.path.isdir(directory):
                continue
            for filename in os.listdir(directory):
                if filename.endswith(".md"):
                    filepath = os.path.join(directory, filename)
                    if self.storage.is_file_changed(filepath):
                        self.index_file(filepath)
                        count += 1

        for key in ["pinned_file", "context_file"]:
            fp = self.storage.paths[key]
            if os.path.exists(fp) and self.storage.is_file_changed(fp):
                self.index_file(fp)
                count += 1

        if count:
            log_rag(f"Проиндексировано файлов: {count}")

    def index_file(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            return

        abs_path = os.path.abspath(filepath).replace("\\", "/")
        self.vector.delete_by_source(abs_path)

        chunks = chunk_markdown(content, source=abs_path)
        if not chunks:
            return

        ids = [f"{abs_path}::chunk{i}" for i in range(len(chunks))]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {"source": abs_path, "chunk": i, "heading_path": json.dumps(c["heading_path"])}
            for i, c in enumerate(chunks)
        ]
        self.vector.upsert(ids=ids, documents=documents, metadatas=metadatas)

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> list[str]:
        try:
            raw = self.vector.query(query, n_results=n_results)
        except Exception as exc:
            log_warn(f"RAG search failed: {exc}")
            return []

        if not raw["documents"] or not raw["documents"][0]:
            return []

        distances = raw["distances"][0] if raw.get("distances") else []
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]

        total = len(docs)
        source_files: set[str] = set()
        for doc, dist, meta in zip(docs, distances, metas):
            score = _distance_to_score(dist) if distances else 1.0
            if score >= MIN_SIMILARITY:
                source_files.add(meta["source"])

        filtered = total - len(source_files)
        if filtered:
            log_rag(f"Отфильтровано {filtered}/{total} чанков ниже порога схожести")

        # Graph expansion via injected index
        if self.graph_index is not None:
            expanded: set[str] = set(source_files)
            for filepath in source_files:
                try:
                    neighbors = self.graph_index.get_connected_nodes(filepath)
                    for neighbor_name in neighbors:
                        neighbor_path = self.graph_index.get_entity_path(neighbor_name)
                        if neighbor_path:
                            expanded.add(neighbor_path)
                except Exception:
                    pass
            source_files = expanded

        final_context: list[str] = []
        for path in source_files:
            _, content = self.storage.read_file(path)
            if content:
                label = os.path.relpath(path, self.storage.base_path)
                final_context.append(f"[{label}]\n{content}")

        if final_context:
            log_rag(f"Найдено источников: {len(final_context)} (запрос: «{query[:40]}»)")

        return final_context

    def reindex_all(self) -> None:
        log_rag("Полная переиндексация...")
        self.vector.reset()
        with open(self.storage.paths["index_file"], "w") as f:
            json.dump({}, f)
        self.vector.reload_client()
        self._sync_index()
        log_rag("Переиндексация завершена.")

    def reload_after_restore(self) -> None:
        """Reload Chroma client after snapshot restore (avoids reindex_all)."""
        self.vector.reload_client()
