import os
import pytest
from storage import StorageManager


def test_snapshot_create_and_list(tmp_memory_dir):
    storage = tmp_memory_dir
    storage.write_file(storage.paths["context_file"], "state v1")
    storage.create_snapshot(label="v1")
    snaps = storage.list_snapshots()
    assert len(snaps) >= 1
    assert any("v1" in s["name"] for s in snaps)


def test_snapshot_restore_recovers_state(tmp_memory_dir):
    storage = tmp_memory_dir
    storage.write_file(storage.paths["context_file"], "original")
    snap_name = os.path.basename(storage.create_snapshot(label="orig"))

    storage.write_file(storage.paths["context_file"], "modified")
    _, before = storage.read_file(storage.paths["context_file"])
    assert before == "modified"

    storage.restore_snapshot(snap_name)
    _, after = storage.read_file(storage.paths["context_file"])
    assert after == "original"


def test_snapshot_meta_has_timestamp(tmp_memory_dir):
    storage = tmp_memory_dir
    storage.create_snapshot(label="check_meta")
    snaps = storage.list_snapshots()
    snap = next(s for s in snaps if "check_meta" in s["name"])
    assert "timestamp" in snap
