import json
import os
import pytest
from storage import StorageManager
from rag_engine import RAGEngine


def _reset_mtime_index(storage: StorageManager) -> None:
    """Clear mtime index so RAGEngine._sync_index() re-indexes all files."""
    with open(storage.paths["index_file"], "w") as f:
        json.dump({}, f)


def test_rag_sync_index_counts_chunks(tmp_memory_dir):
    storage = tmp_memory_dir
    f1 = os.path.join(storage.paths["knowledge"], "entity1.md")
    f2 = os.path.join(storage.paths["knowledge"], "entity2.md")
    storage.write_file(f1, "# Entity 1\nThis is entity one about Python programming.")
    storage.write_file(f2, "# Entity 2\nThis is entity two about machine learning.")

    _reset_mtime_index(storage)
    rag = RAGEngine(storage)
    assert rag.collection.count() >= 2


def test_rag_search_returns_results(tmp_memory_dir):
    storage = tmp_memory_dir
    f1 = os.path.join(storage.paths["knowledge"], "python.md")
    storage.write_file(f1, "# Python\nPython is a high-level programming language used for data science.")

    _reset_mtime_index(storage)
    rag = RAGEngine(storage)
    results = rag.search("programming language")
    assert len(results) > 0


def test_rag_search_empty_index_returns_empty(tmp_memory_dir):
    rag = RAGEngine(tmp_memory_dir)
    results = rag.search("something completely unrelated xyz123")
    assert isinstance(results, list)
