import json

import pytest

from loci.graph.extractor import KnowledgeGraph


@pytest.fixture
def kg(tmp_memory_dir):
    return KnowledgeGraph(tmp_memory_dir)


# ── _parse_json_facts ──────────────────────────────────────────────────────

def test_valid_json_returns_facts(kg):
    response = json.dumps({
        "facts": [
            {"subject": "Alice", "predicate": "works_at", "object": "Acme", "raw_text": "Alice works at Acme"},
            {"subject": "Alice", "predicate": "is", "object": "engineer", "raw_text": "Alice is an engineer"},
        ]
    })
    facts = kg._parse_json_facts(response, "Alice works at Acme. Alice is an engineer.")
    assert len(facts) == 2
    assert facts[0].subject == "Alice"
    assert facts[0].predicate == "works_at"
    assert facts[0].object == "Acme"


def test_invalid_json_returns_empty(kg):
    facts = kg._parse_json_facts("this is not json at all", "source")
    assert facts == []


def test_json_with_preamble_is_extracted(kg):
    """LLM sometimes prefixes JSON with text; regex fallback should find it."""
    response = 'Sure! Here are the facts:\n{"facts": [{"subject": "X", "predicate": "is", "object": "Y", "raw_text": "X is Y"}]}'
    facts = kg._parse_json_facts(response, "X is Y")
    assert len(facts) == 1
    assert facts[0].subject == "X"


def test_empty_facts_array(kg):
    response = json.dumps({"facts": []})
    facts = kg._parse_json_facts(response, "nothing here")
    assert facts == []


def test_invalid_fact_item_is_skipped(kg):
    """A fact missing required fields should be silently dropped."""
    response = json.dumps({
        "facts": [
            {"predicate": "uses"},  # missing subject
            {"subject": "Bob", "predicate": "knows", "object": "Alice", "raw_text": "Bob knows Alice"},
        ]
    })
    facts = kg._parse_json_facts(response, "Bob knows Alice")
    assert len(facts) == 1
    assert facts[0].subject == "Bob"


def test_pydantic_validation_on_facts(kg):
    """Confidence defaults to 1.0 and contested to False."""
    response = json.dumps({
        "facts": [
            {"subject": "A", "predicate": "b", "object": "C", "raw_text": "A b C"},
        ]
    })
    facts = kg._parse_json_facts(response, "A b C")
    assert facts[0].confidence == 1.0
    assert facts[0].contested is False


# ── extract_and_save_facts integration ────────────────────────────────────

def test_extract_and_save_creates_files(kg, mock_llm):
    mock_llm.append(json.dumps({
        "facts": [
            {"subject": "Bob", "predicate": "uses", "object": "Python", "raw_text": "Bob uses Python"},
        ]
    }))
    kg.extract_and_save_facts("Bob uses Python for data science.")
    import os
    entity_file = os.path.join(kg.storage.paths["knowledge"], "Bob.md")
    assert os.path.exists(entity_file)


def test_extract_falls_back_to_legacy_on_json_parse_failure(kg, mock_llm):
    """When response is legacy markdown (not JSON), fallback parsing creates files."""
    mock_llm.append("- [[Carol]]: works remotely")
    kg.extract_and_save_facts("Carol works remotely.")
    import os
    entity_file = os.path.join(kg.storage.paths["knowledge"], "Carol.md")
    assert os.path.exists(entity_file)
