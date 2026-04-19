from datetime import datetime

from loci.models import CycleResult, Fact, Message, RetrievedChunk, Snapshot


def test_message_defaults():
    msg = Message(role="user", content="hello")
    assert msg.role == "user"
    assert isinstance(msg.timestamp, datetime)


def test_fact_fields():
    fact = Fact(
        subject="Alice",
        predicate="works_at",
        object="ACME",
        raw_text="Alice works_at ACME",
        source_chunk="some text",
    )
    assert fact.confidence == 1.0
    assert fact.object == "ACME"


def test_retrieved_chunk():
    chunk = RetrievedChunk(content="text", source="path/to/file.md", score=0.9, reason="vector")
    assert chunk.reason == "vector"


def test_snapshot():
    snap = Snapshot(name="snap1", timestamp=datetime.now(), path="/tmp/snap1")
    assert not snap.includes_chroma
    assert snap.parent_snapshot is None


def test_cycle_result_ok():
    r = CycleResult(ok=True, clear_buffer=True)
    assert r.ok
    assert r.error == ""


def test_cycle_result_fail():
    r = CycleResult(ok=False, clear_buffer=False, error="boom")
    assert not r.ok
    assert r.error == "boom"
