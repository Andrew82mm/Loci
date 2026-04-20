import os
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

from loci.colors import log_snapshot, log_warn
from loci.storage.filesystem import StorageManager


class GitBackedStorage(StorageManager):
    """
    Snapshot backend that uses a git repository inside project_memory/.
    Text files are versioned by git (delta-compressed).
    Chroma DB is stored as gzip tarballs in _system/chroma_snapshots/.

    .gitignore excludes:
      - _system/chroma_db/      (large binary, managed via tarballs)
      - _system/wal.jsonl       (transient WAL)
      - _system/chroma_snapshots/  (binary tarballs, not in git)
      - _system/file_index.json (mtime index, regenerable)
    """

    _GITIGNORE_CONTENT = (
        "_system/chroma_db/\n"
        "_system/wal.jsonl\n"
        "_system/chroma_snapshots/\n"
        "_system/file_index.json\n"
    )

    def __init__(self, base_path: str | None = None) -> None:
        from loci.config import MEMORY_DIR
        super().__init__(base_path or MEMORY_DIR)
        if not shutil.which("git"):
            raise RuntimeError("git binary not found — cannot use GitBackedStorage")
        self._git_init()

    # ── Git helpers ────────────────────────────────────────────────────────

    def _run_git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", self.base_path, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def _git_init(self) -> None:
        git_dir = os.path.join(self.base_path, ".git")
        if os.path.exists(git_dir):
            return
        self._run_git("init")
        self._run_git("config", "user.email", "loci@memory.local")
        self._run_git("config", "user.name", "Loci Memory")
        gitignore_path = os.path.join(self.base_path, ".gitignore")
        with open(gitignore_path, "w", encoding="utf-8") as f:
            f.write(self._GITIGNORE_CONTENT)
        # Initial commit so HEAD is valid
        self._run_git("add", "-A")
        try:
            self._run_git("commit", "-m", "init: initial memory state")
        except subprocess.CalledProcessError:
            pass  # nothing to commit on a truly empty dir

    # ── Snapshot overrides ─────────────────────────────────────────────────

    def create_snapshot(self, label: str = "", parent_snapshot: str | None = None) -> str:
        # Persist Chroma to tarball (SHA not yet known, use tmp name)
        chroma_src = os.path.join(self.paths["system"], "chroma_db")
        chroma_snaps_dir = os.path.join(self.paths["system"], "chroma_snapshots")
        os.makedirs(chroma_snaps_dir, exist_ok=True)

        tmp_tar = os.path.join(chroma_snaps_dir, "_pending.tar.gz")
        has_chroma = os.path.exists(chroma_src)
        if has_chroma:
            with tarfile.open(tmp_tar, "w:gz") as tar:
                tar.add(chroma_src, arcname="chroma_db")

        # Git commit
        msg = label or "snapshot"
        if parent_snapshot:
            msg += f" (parent: {parent_snapshot[:8]})"
        self._run_git("add", "-A")
        if self._run_git("status", "--porcelain"):
            self._run_git("commit", "-m", msg)

        sha = self._run_git("rev-parse", "HEAD")

        # Rename tarball to SHA-based name
        if has_chroma and os.path.exists(tmp_tar):
            os.rename(tmp_tar, os.path.join(chroma_snaps_dir, f"{sha}.tar.gz"))
        elif os.path.exists(tmp_tar):
            os.remove(tmp_tar)

        log_snapshot(f"Git commit: {sha[:8]} ({label or 'snapshot'})")
        return sha

    def list_snapshots(self) -> list[dict[str, Any]]:
        try:
            output = self._run_git("log", "--format=%H|%s|%ai")
        except subprocess.CalledProcessError:
            return []

        chroma_snaps_dir = os.path.join(self.paths["system"], "chroma_snapshots")
        snaps: list[dict[str, Any]] = []
        for line in output.splitlines():
            if not line.strip():
                continue
            parts = line.split("|", 2)
            sha = parts[0]
            label = parts[1] if len(parts) > 1 else ""
            ts = parts[2] if len(parts) > 2 else ""
            snaps.append({
                "name": sha,
                "label": label,
                "timestamp": ts,
                "path": "",
                "includes_chroma": os.path.exists(
                    os.path.join(chroma_snaps_dir, f"{sha}.tar.gz")
                ),
                "parent_snapshot": None,
            })
        return snaps

    def restore_snapshot(self, snapshot_name: str, silent: bool = False) -> bool:
        sha = snapshot_name

        # 1. Save current state before restoring (for undo support)
        before_sha = self.create_snapshot(label="before_restore", parent_snapshot=sha)
        self._last_before_restore = before_sha

        # 2. Restore tracked text files to sha's exact tree
        try:
            # Files that exist in sha's tree
            sha_files = set(
                self._run_git("ls-tree", "-r", "--name-only", sha).splitlines()
            )
            # Files currently tracked
            current_files = set(self._run_git("ls-files").splitlines())

            # Delete files present in current but absent in sha
            for rel in current_files - sha_files:
                full = os.path.join(self.base_path, rel)
                if os.path.exists(full):
                    os.remove(full)

            # Restore files from sha
            if sha_files:
                self._run_git("checkout", sha, "--", ".")
        except subprocess.CalledProcessError as exc:
            if not silent:
                log_warn(f"Git restore failed: {exc.stderr}")
            return False

        # Commit the restored state so history is preserved
        self._run_git("add", "-A")
        try:
            self._run_git("commit", "-m", f"restore: revert to {sha[:8]}", "--allow-empty")
        except subprocess.CalledProcessError:
            pass

        # 3. Restore Chroma from tarball
        chroma_snaps_dir = os.path.join(self.paths["system"], "chroma_snapshots")
        tar_path = os.path.join(chroma_snaps_dir, f"{sha}.tar.gz")
        chroma_dst = os.path.join(self.paths["system"], "chroma_db")
        if os.path.exists(tar_path):
            if os.path.exists(chroma_dst):
                shutil.rmtree(chroma_dst)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(self.paths["system"])

        if not silent:
            log_snapshot(f"Git откат: {sha[:8]}")
        return True

    def prune_snapshots(self, keep_last: int = 50, keep_tagged: bool = True) -> None:
        """Remove old Chroma tarballs beyond keep_last snapshots.

        Git history itself is kept (rewriting history would be destructive).
        Only the binary Chroma tarballs are pruned.
        """
        snaps = self.list_snapshots()
        to_keep = {s["name"] for s in snaps[:keep_last]}

        chroma_snaps_dir = os.path.join(self.paths["system"], "chroma_snapshots")
        if not os.path.exists(chroma_snaps_dir):
            return

        for fname in os.listdir(chroma_snaps_dir):
            if not fname.endswith(".tar.gz"):
                continue
            sha = fname[: -len(".tar.gz")]
            if sha not in to_keep:
                os.remove(os.path.join(chroma_snaps_dir, fname))
