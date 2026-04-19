from datetime import datetime

import pytest

from loci.graph.index import GraphIndex
from loci.models import Fact


@pytest.fixture
def graph(tmp_path):
    return GraphIndex(str(tmp_path / "relations.db"))


def _fact(subject: str, predicate: str, obj: str) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        raw_text=f"{subject} {predicate} {obj}",
        source_chunk="",
        extracted_at=datetime.now(),
    )


def test_add_and_neighbors_out(graph):
    graph.add(_fact("A", "uses", "B"))
    assert "B" in graph.neighbors("A", direction="out")


def test_neighbors_in(graph):
    graph.add(_fact("A", "uses", "B"))
    assert "A" in graph.neighbors("B", direction="in")


def test_neighbors_both(graph):
    graph.add(_fact("A", "uses", "B"))
    assert "B" in graph.neighbors("A", direction="both")
    assert "A" in graph.neighbors("B", direction="both")


def test_query_by_subject(graph):
    graph.add(_fact("Alice", "works_at", "ACME"))
    graph.add(_fact("Bob", "works_at", "ACME"))
    results = graph.query(subject="Alice")
    assert len(results) == 1
    assert results[0].object == "ACME"


def test_query_by_predicate(graph):
    graph.add(_fact("Alice", "works_at", "ACME"))
    graph.add(_fact("Alice", "lives_in", "Paris"))
    results = graph.query(predicate="works_at")
    assert len(results) == 1


def test_duplicate_fact_upserted(graph):
    graph.add(_fact("A", "knows", "B"))
    graph.add(_fact("A", "knows", "B"))
    results = graph.query(subject="A")
    assert len(results) == 1
