import os

import pytest

from loci.storage.wal import WriteAheadLog


@pytest.fixture
def wal(tmp_path):
    removed: list[str] = []
    wal_path = str(tmp_path / "_system" / "wal.jsonl")
    w = WriteAheadLog(wal_path, removed.append)
    w._removed = removed
    return w


def test_begin_creates_pending(wal):
    entry_id = wal.begin("write", "/some/file.md")
    pending = wal._get_pending()
    assert any(e["id"] == entry_id for e in pending)


def test_commit_removes_from_pending(wal):
    entry_id = wal.begin("write", "/some/file.md")
    wal.commit(entry_id)
    pending = wal._get_pending()
    assert not any(e["id"] == entry_id for e in pending)


def test_recovery_removes_from_index_when_file_exists(tmp_path):
    """Pending entry for an existing file should trigger index removal."""
    existing_file = tmp_path / "data.md"
    existing_file.write_text("content")

    removed: list[str] = []
    wal_path = str(tmp_path / "_system" / "wal.jsonl")

    # Simulate a crash: write pending but never commit
    w1 = WriteAheadLog.__new__(WriteAheadLog)
    w1._path = wal_path
    w1._counter = 0
    os.makedirs(os.path.dirname(wal_path), exist_ok=True)
    w1._append({"id": "crash_001", "op": "write", "path": str(existing_file), "status": "pending"})

    # Recovery runs in __init__
    WriteAheadLog(wal_path, removed.append)
    assert str(existing_file) in removed


def test_recovery_does_nothing_when_file_missing(tmp_path):
    """Pending entry for a non-existent file → nothing removed, just aborted."""
    removed: list[str] = []
    wal_path = str(tmp_path / "_system" / "wal.jsonl")
    os.makedirs(os.path.dirname(wal_path), exist_ok=True)

    w1 = WriteAheadLog.__new__(WriteAheadLog)
    w1._path = wal_path
    w1._counter = 0
    w1._append({"id": "crash_002", "op": "write", "path": "/nonexistent/file.md", "status": "pending"})

    WriteAheadLog(wal_path, removed.append)
    assert removed == []


def test_compaction_fires_at_threshold(tmp_path):
    removed: list[str] = []
    wal_path = str(tmp_path / "_system" / "wal.jsonl")
    w = WriteAheadLog(wal_path, removed.append)
    w.MAX_ENTRIES = 10  # lower threshold for test

    for i in range(6):
        eid = w.begin("write", f"/file{i}.md")
        w.commit(eid)

    # Not compacted yet (6 * 2 = 12 entries > 10 → compacted after 6th commit)
    bak_files = [f for f in os.listdir(os.path.dirname(wal_path)) if f.endswith(".bak")]
    assert len(bak_files) >= 1


def test_filesystem_write_uses_wal(tmp_memory_dir, tmp_path):
    """write_file should create WAL begin+commit entries."""
    target = os.path.join(tmp_memory_dir.paths["knowledge"], "test.md")
    tmp_memory_dir.write_file(target, "hello")

    wal_path = os.path.join(tmp_memory_dir.paths["system"], "wal.jsonl")
    assert os.path.exists(wal_path)
    # No pending entries after a clean write
    assert tmp_memory_dir._wal._get_pending() == []
