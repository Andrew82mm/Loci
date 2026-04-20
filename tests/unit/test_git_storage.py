import os
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git binary not available"
)


@pytest.fixture
def git_storage(tmp_path):
    from loci.storage.git_backed import GitBackedStorage
    return GitBackedStorage(base_path=str(tmp_path / "memory"))


def test_git_init_creates_dot_git(git_storage):
    assert os.path.isdir(os.path.join(git_storage.base_path, ".git"))


def test_create_snapshot_returns_sha(git_storage):
    git_storage.write_file(git_storage.paths["context_file"], "some content")
    sha = git_storage.create_snapshot(label="test")
    assert len(sha) == 40  # full git SHA


def test_list_snapshots_returns_commits(git_storage):
    git_storage.write_file(git_storage.paths["context_file"], "v1")
    git_storage.create_snapshot(label="snap1")
    git_storage.write_file(git_storage.paths["context_file"], "v2")
    git_storage.create_snapshot(label="snap2")

    snaps = git_storage.list_snapshots()
    labels = [s["label"] for s in snaps]
    assert "snap2" in labels
    assert "snap1" in labels


def test_restore_snapshot_restores_content(git_storage):
    # Write v1 and snapshot it
    git_storage.write_file(git_storage.paths["context_file"], "version one")
    sha1 = git_storage.create_snapshot(label="v1")

    # Overwrite with v2 and snapshot
    git_storage.write_file(git_storage.paths["context_file"], "version two")
    git_storage.create_snapshot(label="v2")

    # Restore to v1
    ok = git_storage.restore_snapshot(sha1)
    assert ok

    _, content = git_storage.read_file(git_storage.paths["context_file"])
    assert "version one" in content


def test_prune_snapshots_removes_old_tarballs(git_storage, tmp_path):
    """Chroma tarballs beyond keep_last are removed."""
    chroma_snaps_dir = os.path.join(git_storage.paths["system"], "chroma_snapshots")
    os.makedirs(chroma_snaps_dir, exist_ok=True)

    # Create fake tarballs for non-existent SHAs
    for i in range(5):
        fake_sha = f"{'a' * 39}{i}"
        open(os.path.join(chroma_snaps_dir, f"{fake_sha}.tar.gz"), "w").close()

    # Create real snapshots so list_snapshots returns something
    git_storage.write_file(git_storage.paths["context_file"], "keep")
    git_storage.create_snapshot(label="keep")

    git_storage.prune_snapshots(keep_last=1)

    # Fake tarballs should be removed (they're not in git log)
    remaining = os.listdir(chroma_snaps_dir)
    assert all(f.endswith(".tar.gz") for f in remaining)
    assert len(remaining) <= 1  # only the real snapshot or nothing
