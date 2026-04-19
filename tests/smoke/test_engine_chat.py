import pytest
from unittest.mock import MagicMock, patch
from llm_memory.storage.filesystem import StorageManager
import llm_memory.engine as engine_mod


def _engine_patches(storage):
    """Context manager that patches all heavy dependencies in MemoryEngine."""
    return (
        patch("llm_memory.engine.StorageManager", return_value=storage),
        patch("llm_memory.engine.RAGEngine", return_value=MagicMock(search=MagicMock(return_value=[]))),
        patch("llm_memory.engine.KnowledgeGraph", return_value=MagicMock()),
        patch("llm_memory.engine.GraphIndex", return_value=MagicMock()),
        patch("llm_memory.engine.SummarizationPipeline", return_value=MagicMock()),
    )


def test_chat_buffer_grows(tmp_path):
    """Buffer grows by 2 per chat call (user msg + assistant msg)."""
    storage = StorageManager(base_path=str(tmp_path / "memory"))

    with (
        patch("llm_memory.engine.StorageManager", return_value=storage),
        patch("llm_memory.engine.RAGEngine", return_value=MagicMock(search=MagicMock(return_value=[]))),
        patch("llm_memory.engine.KnowledgeGraph", return_value=MagicMock()),
        patch("llm_memory.engine.GraphIndex", return_value=MagicMock()),
        patch("llm_memory.engine.SummarizationPipeline", return_value=MagicMock()),
        patch.object(engine_mod.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = engine_mod.MemoryEngine()
        engine.chat("Hello")
        assert len(engine.buffer) == 2
        engine.chat("How are you?")
        assert len(engine.buffer) == 4


def test_summarization_triggers_at_threshold(tmp_path, monkeypatch):
    """Summarization runs when buffer token count exceeds SUMMARIZE_TOKEN_THRESHOLD."""
    storage = StorageManager(base_path=str(tmp_path / "memory"))
    rag_mock = MagicMock()
    rag_mock.search.return_value = []
    summarize_mock = MagicMock()

    # Lower threshold so a few messages trigger it
    monkeypatch.setattr(engine_mod, "SUMMARIZE_TOKEN_THRESHOLD", 10)

    with (
        patch("llm_memory.engine.StorageManager", return_value=storage),
        patch("llm_memory.engine.RAGEngine", return_value=MagicMock(search=MagicMock(return_value=[]))),
        patch("llm_memory.engine.KnowledgeGraph", return_value=MagicMock()),
        patch("llm_memory.engine.GraphIndex", return_value=MagicMock()),
        patch("llm_memory.engine.SummarizationPipeline", return_value=MagicMock()),
        patch.object(engine_mod.llm_client, "generate", return_value="Mock answer. References: none"),
    ):
        engine = engine_mod.MemoryEngine()
        engine._run_summarization_cycle = summarize_mock

        # First message: short, should not trigger yet
        engine.chat("hi")
        # Second message with enough tokens to push over threshold=10
        engine.chat("word " * 20)
        assert summarize_mock.call_count >= 1
