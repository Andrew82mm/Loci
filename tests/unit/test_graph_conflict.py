import pytest

from loci.graph.index import GraphIndex
from loci.models import Fact


@pytest.fixture
def graph(tmp_path):
    db_path = str(tmp_path / "_system" / "relations.db")
    return GraphIndex(db_path)


def _fact(subject: str, predicate: str, obj: str | None) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        raw_text=f"{subject} {predicate} {obj or ''}".strip(),
        source_chunk="",
    )


def test_no_conflict_same_triple(graph):
    """Adding the same triple twice → not contested."""
    f = _fact("A", "works_at", "B")
    graph.add(f)
    graph.add(f)
    facts = graph.query(subject="A", predicate="works_at")
    assert len(facts) == 1
    assert facts[0].contested is False


def test_conflict_different_object(graph):
    """A works_at B and A works_at C → both contested."""
    graph.add(_fact("A", "works_at", "B"))
    graph.add(_fact("A", "works_at", "C"))
    facts = graph.query(subject="A", predicate="works_at")
    assert len(facts) == 2
    assert all(f.contested for f in facts)


def test_no_conflict_different_predicate(graph):
    """Same subject, different predicates → no conflict."""
    graph.add(_fact("A", "works_at", "B"))
    graph.add(_fact("A", "knows", "B"))
    facts_works = graph.query(subject="A", predicate="works_at")
    facts_knows = graph.query(subject="A", predicate="knows")
    assert facts_works[0].contested is False
    assert facts_knows[0].contested is False


def test_conflict_null_object(graph):
    """Null object vs non-null object for same (subject, predicate) → conflict."""
    graph.add(_fact("A", "is", None))
    graph.add(_fact("A", "is", "engineer"))
    facts = graph.query(subject="A", predicate="is")
    assert all(f.contested for f in facts)


def test_schema_migration_adds_contested_column(tmp_path):
    """Opening an old DB without contested column should migrate cleanly."""
    import sqlite3
    db_path = str(tmp_path / "old.db")
    # Create table without contested column (legacy schema)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE relations (
            id INTEGER PRIMARY KEY,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT,
            source_file TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            extracted_at TEXT NOT NULL,
            UNIQUE(subject, predicate, object)
        )
    """)
    conn.commit()
    conn.close()

    # Opening with GraphIndex should migrate without error
    g = GraphIndex(db_path)
    g.add(_fact("X", "is", "Y"))
    facts = g.query(subject="X")
    assert facts[0].contested is False
