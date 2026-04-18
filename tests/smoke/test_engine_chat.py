import pytest
from unittest.mock import MagicMock, patch
from llm_memory.storage.filesystem import StorageManager
import llm_memory.engine as engine_mod


def test_chat_buffer_grows(tmp_path):
    """Buffer grows by 2 per chat call (user msg + assistant msg)."""
    storage = StorageManager(base_path=str(tmp_path / "memory"))
    rag_mock = MagicMock()
    rag_mock.search.return_value = []

    with (
        patch("llm_memory.engine.StorageManager", return_value=storage),
        patch("llm_memory.engine.RAGEngine", return_value=rag_mock),
        patch("llm_memory.engine.KnowledgeGraph", return_value=MagicMock()),
        patch.object(engine_mod.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = engine_mod.MemoryEngine()
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

    monkeypatch.setattr(engine_mod, "SUMMARIZE_EVERY_N_MSG", 2)

    with (
        patch("llm_memory.engine.StorageManager", return_value=storage),
        patch("llm_memory.engine.RAGEngine", return_value=rag_mock),
        patch("llm_memory.engine.KnowledgeGraph", return_value=MagicMock()),
        patch.object(engine_mod.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = engine_mod.MemoryEngine()
        engine._run_summarization_cycle = summarize_mock

        engine.chat("msg 1")  # buffer = 2, threshold = 4, no trigger
        assert summarize_mock.call_count == 0

        engine.chat("msg 2")  # buffer = 4 >= 4, trigger
        assert summarize_mock.call_count == 1
