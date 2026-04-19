from unittest.mock import MagicMock


def test_irrelevant_query_returns_empty(tmp_path):
    """Query with no relevant data should return empty list after threshold filtering."""
    from loci.rag.retriever import RAGEngine
    from loci.storage.filesystem import StorageManager

    storage = StorageManager(base_path=str(tmp_path / "memory"))

    engine = RAGEngine(storage, graph_index=None)

    # Patch vector query to return high distances (irrelevant)
    engine.vector.query = MagicMock(return_value={
        "documents": [["doc1", "doc2"]],
        "distances": [[1.9, 1.95]],  # Very high distance → ~0 similarity
        "metadatas": [[{"source": "a.md"}, {"source": "b.md"}]],
    })

    results = engine.search("something completely unrelated")
    assert results == []


def test_relevant_query_passes_threshold(tmp_path):
    """Query with very similar docs should return results."""
    import os

    from loci.rag.retriever import RAGEngine
    from loci.storage.filesystem import StorageManager

    storage = StorageManager(base_path=str(tmp_path / "memory"))
    engine = RAGEngine(storage, graph_index=None)

    # Create a real file for the result
    knowledge_dir = storage.paths["knowledge"]
    test_file = os.path.join(knowledge_dir, "test.md")
    with open(test_file, "w") as f:
        f.write("# Test\nThis is relevant content.")
    abs_path = os.path.abspath(test_file).replace("\\", "/")

    engine.vector.query = MagicMock(return_value={
        "documents": [["This is relevant content."]],
        "distances": [[0.1]],  # Low distance → high similarity
        "metadatas": [[{"source": abs_path}]],
    })

    results = engine.search("relevant content")
    assert len(results) == 1
