import pytest
from knowledge_graph import KnowledgeGraph


def test_kg_extract_creates_entity_file(tmp_memory_dir, mock_llm):
    mock_llm.append("- [[Alice]]: works at Acme Corp\n- [[Alice]] --(knows)--> [[Bob]]")
    kg = KnowledgeGraph(tmp_memory_dir)
    kg.extract_and_save_facts("Alice works at Acme Corp and knows Bob.")

    alice_path = kg.get_entity_path("Alice")
    assert alice_path is not None
    _, content = tmp_memory_dir.read_file(alice_path)
    assert "Alice" in content


def test_kg_extract_handles_llm_error(tmp_memory_dir, mock_llm):
    mock_llm.append("Error: API unavailable")
    kg = KnowledgeGraph(tmp_memory_dir)
    result = kg.extract_and_save_facts("Some text about things.")

    assert result.startswith("Error:")
    assert kg.get_entity_path("Some") is None


def test_kg_get_connected_nodes(tmp_memory_dir, mock_llm):
    mock_llm.append("- [[Alice]] --(works_at)--> [[Acme]]\n- [[Alice]] --(knows)--> [[Bob]]")
    kg = KnowledgeGraph(tmp_memory_dir)
    kg.extract_and_save_facts("Alice works at Acme and knows Bob.")

    alice_path = kg.get_entity_path("Alice")
    assert alice_path is not None
    neighbors = kg.get_connected_nodes(alice_path)
    assert "Acme" in neighbors or "Bob" in neighbors
