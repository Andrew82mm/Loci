# rag_engine.py
import os
import chromadb
from chromadb.utils import embedding_functions
from storage import StorageManager
from colors import log_rag, log_warn

# Размер чанка в символах и перекрытие
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

class RAGEngine:
    def __init__(self, storage: StorageManager):
        self.storage = storage
        db_path = os.path.join(storage.paths["system"], "chroma_db")
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name="memory_vault",
            embedding_function=self.embedding_func,
        )
        self._sync_index()

    # ── Индексирование ────────────────────────────────────────────────────

    def _sync_index(self):
        """Индексирует только изменившиеся файлы (mtime-based)."""
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

    def index_file(self, filepath: str):
        """Индексирует файл чанками. Удаляет старые чанки перед добавлением."""
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return

        doc_id_base = os.path.abspath(filepath).replace("\\", "/")
        chunks = self._split_chunks(content)

        # Удаляем старые чанки этого файла
        try:
            existing = self.collection.get(where={"source": doc_id_base})
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        if not chunks:
            return

        ids       = [f"{doc_id_base}::chunk{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_id_base, "chunk": i} for i in range(len(chunks))]

        self.collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)

    def _split_chunks(self, text: str) -> list[str]:
        """Делит текст на чанки с перекрытием."""
        if len(text) <= CHUNK_SIZE:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    # ── Поиск ─────────────────────────────────────────────────────────────

    def search(self, query: str, n_results: int = 5) -> list[str]:
        """
        Гибридный поиск:
          1. Векторный поиск по чанкам
          2. Графовый обход соседей (1 уровень)
        Возвращает список строк с содержимым + указанием источника.
        """
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            log_warn(f"RAG search failed: {e}")
            return []

        if not results["documents"] or not results["documents"][0]:
            return []

        # Собираем уникальные исходные файлы из найденных чанков
        source_files: set[str] = set()
        for meta_list in results["metadatas"]:
            for meta in meta_list:
                source_files.add(meta["source"])

        # Графовый обход — добавляем соседей
        from knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph(self.storage)
        expanded_files = set(source_files)

        for filepath in source_files:
            neighbors = kg.get_connected_nodes(filepath)
            for neighbor_name in neighbors:
                neighbor_path = kg.get_entity_path(neighbor_name)
                if neighbor_path:
                    expanded_files.add(neighbor_path)

        # Читаем содержимое всех найденных файлов
        final_context = []
        for path in expanded_files:
            _, content = self.storage.read_file(path)
            if content:
                label = os.path.relpath(path, self.storage.base_path)
                final_context.append(f"[{label}]\n{content}")

        if final_context:
            log_rag(f"Найдено источников: {len(final_context)} (запрос: «{query[:40]}»)")

        return final_context

    def reindex_all(self):
        """Принудительная переиндексация всех файлов."""
        log_rag("Полная переиндексация...")
        # Сбрасываем коллекцию
        self.client.delete_collection("memory_vault")
        self.collection = self.client.get_or_create_collection(
            name="memory_vault",
            embedding_function=self.embedding_func,
        )
        # Обнуляем индекс mtime
        import json
        with open(self.storage.paths["index_file"], "w") as f:
            json.dump({}, f)
        self._sync_index()
        log_rag("Переиндексация завершена.")
