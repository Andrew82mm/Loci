import pytest
from unittest.mock import MagicMock, patch
from storage import StorageManager
import core_engine


def test_chat_buffer_grows(tmp_path):
    """Buffer grows by 2 per chat call (user msg + assistant msg)."""
    storage = StorageManager(base_path=str(tmp_path / "memory"))
    rag_mock = MagicMock()
    rag_mock.search.return_value = []

    with (
        patch("core_engine.StorageManager", return_value=storage),
        patch("core_engine.RAGEngine", return_value=rag_mock),
        patch("core_engine.KnowledgeGraph", return_value=MagicMock()),
        patch.object(core_engine.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = core_engine.MemoryEngine()
        engine.chat("Hello")
        assert len(engine.buffer) == 2
        engine.chat("How are you?")
        assert len(engine.buffer) == 4


def test_summarization_triggers_at_threshold(tmp_path, monkeypatch):
    """Summarization runs when buffer reaches SUMMARIZE_EVERY_N_MSG * 2."""
    storage = StorageManager(base_path=str(tmp_path / "memory"))
    rag_mock = MagicMock()
    rag_mock.search.return_value = []
    summarize_mock = MagicMock()

    monkeypatch.setattr(core_engine, "SUMMARIZE_EVERY_N_MSG", 2)

    with (
        patch("core_engine.StorageManager", return_value=storage),
        patch("core_engine.RAGEngine", return_value=rag_mock),
        patch("core_engine.KnowledgeGraph", return_value=MagicMock()),
        patch.object(core_engine.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = core_engine.MemoryEngine()
        engine._run_summarization_cycle = summarize_mock

        engine.chat("msg 1")  # buffer = 2, threshold = 4, no trigger
        assert summarize_mock.call_count == 0

        engine.chat("msg 2")  # buffer = 4 >= 4, trigger
        assert summarize_mock.call_count == 1
