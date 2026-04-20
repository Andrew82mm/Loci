import json
import os
import shutil
from datetime import datetime
from typing import Callable

from loci.colors import log_warn


class WriteAheadLog:
    """
    Append-only write-ahead log for StorageManager.

    Lifecycle per write_file call:
      1. begin(path)  → appends {"id": ..., "status": "pending"}
      2. write file + update mtime index (the actual work)
      3. commit(id)   → appends {"id": ..., "status": "committed"}

    On StorageManager startup, _recover() finds entries that never reached
    "committed" and removes those paths from the mtime index so RAGEngine
    will re-sync them on next _sync_index() call.
    """

    MAX_ENTRIES = 100

    def __init__(self, wal_path: str, remove_from_index_fn: Callable[[str], None]) -> None:
        self._path = wal_path
        self._remove_from_index = remove_from_index_fn
        self._counter = 0
        self._recover()

    # ── Public API ─────────────────────────────────────────────────────────

    def begin(self, op: str, path: str) -> str:
        self._counter += 1
        ts = datetime.now().isoformat()
        entry_id = f"{ts}_{self._counter:04d}"
        self._append({"id": entry_id, "op": op, "path": path, "ts": ts, "status": "pending"})
        return entry_id

    def commit(self, entry_id: str) -> None:
        self._append({"id": entry_id, "status": "committed"})
        self._maybe_compact()

    # ── Recovery ───────────────────────────────────────────────────────────

    def _recover(self) -> None:
        pending = self._get_pending()
        for entry in pending:
            path = entry.get("path", "")
            if path and os.path.exists(path):
                # File was written but mtime index may be stale → force re-sync
                self._remove_from_index(path)
                log_warn(f"[WAL] Recovered pending write: {os.path.basename(path)}")
            self._append({"id": entry["id"], "status": "aborted"})

    def _get_pending(self) -> list[dict]:
        entries = self._read_all()
        # Build map: id → latest entry
        latest: dict[str, dict] = {}
        for entry in entries:
            eid = entry.get("id", "")
            if eid:
                latest[eid] = entry
        return [e for e in latest.values() if e.get("status") == "pending"]

    # ── I/O ────────────────────────────────────────────────────────────────

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self._path):
            return []
        entries: list[dict] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def _append(self, entry: dict) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _maybe_compact(self) -> None:
        entries = self._read_all()
        if len(entries) < self.MAX_ENTRIES:
            return

        # Build latest-status map
        latest: dict[str, dict] = {}
        for entry in entries:
            eid = entry.get("id", "")
            if eid:
                latest[eid] = entry

        # Archive full log before compacting
        backup = self._path + f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        shutil.copy2(self._path, backup)

        # Keep only non-committed entries
        surviving = [e for e in latest.values() if e.get("status") != "committed"]
        with open(self._path, "w", encoding="utf-8") as f:
            for entry in surviving:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
