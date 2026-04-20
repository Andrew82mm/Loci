import pytest

from loci.graph.judge import FactJudge
from loci.models import Fact


def _make_fact(raw_text: str) -> Fact:
    return Fact(
        subject="Alice",
        predicate="works_at",
        object="Acme",
        raw_text=raw_text,
        source_chunk="",
    )


SOURCE = "Alice works at Acme Corp as an engineer."


def test_judge_yes_keeps_fact(mock_llm):
    mock_llm.append("yes")
    facts = [_make_fact("Alice works at Acme")]
    result = FactJudge().validate(facts, SOURCE)
    assert len(result) == 1


def test_judge_no_discards_fact(mock_llm):
    mock_llm.append("no")
    facts = [_make_fact("Alice lives in Paris")]
    result = FactJudge().validate(facts, SOURCE)
    assert result == []


def test_judge_mixed_response(mock_llm):
    mock_llm.append("yes")
    mock_llm.append("no")
    facts = [
        _make_fact("Alice works at Acme"),
        _make_fact("Alice lives in Paris"),
    ]
    result = FactJudge().validate(facts, SOURCE)
    assert len(result) == 1
    assert result[0].raw_text == "Alice works at Acme"


def test_judge_empty_input():
    result = FactJudge().validate([], SOURCE)
    assert result == []


def test_judge_error_response_keeps_fact(mock_llm):
    """LLM errors should fail-open (keep the fact)."""
    mock_llm.append("Error: timeout")
    facts = [_make_fact("Alice works at Acme")]
    result = FactJudge().validate(facts, SOURCE)
    assert len(result) == 1


def test_enable_fact_validation_flag(tmp_memory_dir, mock_llm, monkeypatch):
    """When ENABLE_FACT_VALIDATION=True, judge is invoked from extractor."""
    import json
    from loci.graph.extractor import KnowledgeGraph

    monkeypatch.setattr("loci.graph.extractor.ENABLE_FACT_VALIDATION", True)
    kg = KnowledgeGraph(tmp_memory_dir)

    # First LLM call: structured extraction
    mock_llm.append(json.dumps({
        "facts": [{"subject": "Dave", "predicate": "is", "object": "manager", "raw_text": "Dave is manager"}]
    }))
    # Second LLM call: judge says "no"
    mock_llm.append("no")

    kg.extract_and_save_facts("Dave is manager.")

    import os
    entity_file = os.path.join(tmp_memory_dir.paths["knowledge"], "Dave.md")
    assert not os.path.exists(entity_file)


def test_disable_fact_validation_skips_judge(tmp_memory_dir, mock_llm, monkeypatch):
    """When ENABLE_FACT_VALIDATION=False, judge is not called."""
    import json
    from loci.graph.extractor import KnowledgeGraph

    monkeypatch.setattr("loci.graph.extractor.ENABLE_FACT_VALIDATION", False)
    kg = KnowledgeGraph(tmp_memory_dir)

    mock_llm.append(json.dumps({
        "facts": [{"subject": "Eve", "predicate": "is", "object": "dev", "raw_text": "Eve is dev"}]
    }))
    # If judge were called it would consume from mock_llm and we'd get an error
    kg.extract_and_save_facts("Eve is dev.")

    import os
    entity_file = os.path.join(tmp_memory_dir.paths["knowledge"], "Eve.md")
    assert os.path.exists(entity_file)
    # mock_llm should still have responses (judge was never called)
    assert mock_llm == []
