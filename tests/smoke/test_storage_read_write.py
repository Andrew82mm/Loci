import pytest
from storage import StorageManager


def test_write_read_roundtrip_with_metadata(tmp_memory_dir):
    storage = tmp_memory_dir
    path = storage.paths["context_file"]
    storage.write_file(path, "Hello world", metadata={"key": "value", "num": 42})
    meta, body = storage.read_file(path)
    assert body == "Hello world"
    assert meta["key"] == "value"
    assert meta["num"] == 42


def test_write_read_roundtrip_no_metadata(tmp_memory_dir):
    storage = tmp_memory_dir
    path = storage.paths["context_file"]
    storage.write_file(path, "Plain content")
    _, body = storage.read_file(path)
    assert body == "Plain content"


def test_read_nonexistent_returns_empty(tmp_memory_dir):
    storage = tmp_memory_dir
    meta, body = storage.read_file("/nonexistent/file.md")
    assert meta == {}
    assert body == ""


def test_append_deduplicates(tmp_memory_dir):
    storage = tmp_memory_dir
    path = storage.paths["context_file"]
    storage.write_file(path, "")
    storage.append_to_file(path, ["line one", "line two"])
    storage.append_to_file(path, ["line two", "line three"])
    _, body = storage.read_file(path)
    lines = [line for line in body.splitlines() if line.strip()]
    assert lines.count("line two") == 1
    assert "line one" in lines
    assert "line three" in lines
