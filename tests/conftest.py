import pytest
from storage import StorageManager


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Temporary memory directory with initialized StorageManager."""
    return StorageManager(base_path=str(tmp_path / "memory"))


@pytest.fixture
def mock_llm(monkeypatch):
    """Provides a list to prepend expected LLM responses to."""
    responses: list[str] = []

    def fake_generate(model: str, sys_prompt: str, user_prompt: str, **kw: object) -> str:
        return responses.pop(0) if responses else "MOCK"

    monkeypatch.setattr("llm_client.llm_client.generate", fake_generate)
    return responses
