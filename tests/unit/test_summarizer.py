import pytest
from datetime import datetime
from llm_memory.models import Message
from llm_memory.summarizer import SummarizationPipeline


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content, timestamp=datetime.now())


class FakeStorage:
    def __init__(self, tmp_path):
        import os, json
        self.base_path = str(tmp_path)
        system = tmp_path / "_system"
        system.mkdir(parents=True, exist_ok=True)
        (tmp_path / "archive").mkdir(exist_ok=True)
        (tmp_path / "context.md").write_text("")
        (tmp_path / "pinned.md").write_text("")
        (system / "task.md").write_text("")
        self.paths = {
            "task_file": str(system / "task.md"),
            "context_file": str(tmp_path / "context.md"),
            "archive": str(tmp_path / "archive"),
        }
        self.snapshot_name = "snap_001"
        self.restore_called = False

    def create_snapshot(self, label="", parent_snapshot=None):
        return self.snapshot_name

    def restore_snapshot(self, name, silent=False):
        self.restore_called = True
        return True

    def write_file(self, path, content, metadata=None):
        import builtins
        with builtins.open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def append_to_archive(self, messages):
        return ""


class FakeExtractor:
    def extract_and_save_facts(self, text):
        pass


@pytest.fixture
def fake_storage(tmp_path):
    return FakeStorage(tmp_path)


@pytest.fixture
def fake_extractor():
    return FakeExtractor()


def test_summarizer_clears_buffer_on_success(fake_storage, fake_extractor, monkeypatch):
    responses = ["New task: do stuff.", "Summary of the conversation."]

    def fake_generate(model, sys_p, user_p, **kw):
        return responses.pop(0) if responses else "MOCK"

    monkeypatch.setattr("llm_memory.summarizer.llm_client.generate", fake_generate)

    pipeline = SummarizationPipeline(fake_storage, fake_extractor)
    msgs = [_msg("user", "hi"), _msg("assistant", "hello")]
    result = pipeline.run_cycle(msgs)

    assert result.ok
    assert result.clear_buffer


def test_summarizer_preserves_buffer_on_error_response(fake_storage, fake_extractor, monkeypatch):
    """LLM returns Error: prefix — buffer must NOT be cleared."""
    def fake_generate(model, sys_p, user_p, **kw):
        return "Error: rate limit"

    monkeypatch.setattr("llm_memory.summarizer.llm_client.generate", fake_generate)

    pipeline = SummarizationPipeline(fake_storage, fake_extractor)
    msgs = [_msg("user", "hi")]
    result = pipeline.run_cycle(msgs)

    assert not result.ok
    assert not result.clear_buffer
    assert fake_storage.restore_called


def test_summarizer_preserves_buffer_on_empty_summary(fake_storage, fake_extractor, monkeypatch):
    """LLM returns empty string — buffer must NOT be cleared."""
    call_count = {"n": 0}

    def fake_generate(model, sys_p, user_p, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "task update"
        return ""

    monkeypatch.setattr("llm_memory.summarizer.llm_client.generate", fake_generate)

    pipeline = SummarizationPipeline(fake_storage, fake_extractor)
    msgs = [_msg("user", "hi")]
    result = pipeline.run_cycle(msgs)

    assert not result.ok
    assert not result.clear_buffer
