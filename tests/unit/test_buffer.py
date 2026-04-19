from datetime import datetime

from loci.buffer import ConversationBuffer
from loci.models import Message


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content, timestamp=datetime.now())


def test_add_and_all():
    buf = ConversationBuffer(keep_recent_k=4)
    buf.add(_msg("user", "hello"))
    buf.add(_msg("assistant", "hi"))
    assert len(buf.all()) == 2


def test_recent():
    buf = ConversationBuffer(keep_recent_k=4)
    for i in range(10):
        buf.add(_msg("user", f"msg{i}"))
    assert len(buf.recent(4)) == 4
    assert buf.recent(4)[-1].content == "msg9"


def test_to_summarize_returns_all_but_last_k():
    buf = ConversationBuffer(keep_recent_k=4)
    for i in range(10):
        buf.add(_msg("user", f"msg{i}"))
    to_sum = buf.to_summarize()
    assert len(to_sum) == 6
    assert to_sum[-1].content == "msg5"


def test_to_summarize_empty_when_few():
    buf = ConversationBuffer(keep_recent_k=4)
    for i in range(3):
        buf.add(_msg("user", f"msg{i}"))
    assert buf.to_summarize() == []


def test_clear_summarized_keeps_last_k():
    buf = ConversationBuffer(keep_recent_k=4)
    for i in range(20):
        buf.add(_msg("user", f"msg{i}"))
    buf.clear_summarized()
    assert len(buf) == 4
    assert buf.all()[-1].content == "msg19"


def test_roundtrip_dicts():
    buf = ConversationBuffer(keep_recent_k=4)
    buf.add(_msg("user", "hello"))
    buf.add(_msg("assistant", "world"))
    restored = ConversationBuffer.from_dicts(buf.to_dicts())
    assert len(restored) == 2
    assert restored.all()[0].content == "hello"
