import os
import json
import shutil
from datetime import datetime
import yaml
from llm_memory.config import MEMORY_DIR
from llm_memory.colors import log_system, log_snapshot, log_warn


class StorageManager:
    def __init__(self, base_path=MEMORY_DIR):
        self.base_path = base_path
        self.paths = {
            "system":       os.path.join(base_path, "_system"),
            "snapshots":    os.path.join(base_path, "_system", "snapshots"),
            "archive":      os.path.join(base_path, "archive"),
            "knowledge_global":  os.path.join(base_path, "knowledge", "_global"),
            "knowledge_project": os.path.join(base_path, "knowledge", "default"),
            "knowledge":    os.path.join(base_path, "knowledge", "default"),
            "context_file": os.path.join(base_path, "context.md"),
            "pinned_file":  os.path.join(base_path, "pinned.md"),
            "task_file":    os.path.join(base_path, "_system", "task.md"),
            "history_file": os.path.join(base_path, "_system", "conversation_buffer.json"),
            "index_file":   os.path.join(base_path, "_system", "file_index.json"),
        }
        self._init_dirs()

    def _init_dirs(self):
        for key in ["system", "snapshots", "archive",
                    "knowledge_global", "knowledge_project"]:
            os.makedirs(self.paths[key], exist_ok=True)
        for key in ["context_file", "pinned_file", "task_file"]:
            if not os.path.exists(self.paths[key]):
                self.write_file(self.paths[key], "")

    # ── Файловые операции ──────────────────────────────────────────────────

    def write_file(self, filepath, content, metadata=None):
        full_content = ""
        if metadata:
            full_content = "---\n" + yaml.dump(metadata, allow_unicode=True) + "---\n"
        full_content += content
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_content)
        self._update_index(filepath)

    def read_file(self, filepath):
        """Возвращает (metadata_dict, body_str)."""
        if not os.path.exists(filepath):
            return {}, ""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                metadata = yaml.safe_load(parts[1]) or {}
                return metadata, parts[2].strip()
        return {}, content

    def append_to_file(self, filepath, new_lines: list[str]):
        """Дописывает строки в файл только если их там ещё нет (дедупликация)."""
        _, existing = self.read_file(filepath)
        existing_set = set(existing.splitlines())
        to_add = [l for l in new_lines if l.strip() and l.strip() not in existing_set]
        if not to_add:
            return
        with open(filepath, "a", encoding="utf-8") as f:
            for line in to_add:
                f.write(line + "\n")
        self._update_index(filepath)

    # ── Индекс файлов (mtime) ──────────────────────────────────────────────

    def _update_index(self, filepath):
        index = self._load_index()
        index[os.path.abspath(filepath)] = os.path.getmtime(filepath)
        with open(self.paths["index_file"], "w", encoding="utf-8") as f:
            json.dump(index, f)

    def _load_index(self) -> dict:
        if os.path.exists(self.paths["index_file"]):
            with open(self.paths["index_file"], "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def is_file_changed(self, filepath) -> bool:
        index = self._load_index()
        abs_path = os.path.abspath(filepath)
        if abs_path not in index:
            return True
        try:
            return os.path.getmtime(filepath) > index[abs_path]
        except FileNotFoundError:
            return False

    # ── Снэпшоты и откат ──────────────────────────────────────────────────

    def create_snapshot(self, label: str = "") -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"snapshot_{timestamp}" + (f"_{label}" if label else "")
        snapshot_path = os.path.join(self.paths["snapshots"], name)
        os.makedirs(snapshot_path, exist_ok=True)

        for key in ["context_file", "pinned_file", "task_file"]:
            src = self.paths[key]
            if os.path.exists(src):
                shutil.copy2(src, snapshot_path)

        knowledge_root = os.path.join(self.base_path, "knowledge")
        if os.path.exists(knowledge_root):
            shutil.copytree(knowledge_root,
                            os.path.join(snapshot_path, "knowledge"),
                            dirs_exist_ok=True)

        meta = {"timestamp": timestamp, "label": label}
        with open(os.path.join(snapshot_path, "meta.json"), "w") as f:
            json.dump(meta, f)

        log_snapshot(f"Создан: {name}")
        return snapshot_path

    def list_snapshots(self) -> list[dict]:
        snaps = []
        snap_dir = self.paths["snapshots"]
        if not os.path.exists(snap_dir):
            return snaps
        for name in sorted(os.listdir(snap_dir), reverse=True):
            path = os.path.join(snap_dir, name)
            meta_file = os.path.join(path, "meta.json")
            meta = {}
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    try:
                        meta = json.load(f)
                    except Exception:
                        pass
            snaps.append({"name": name, "path": path, **meta})
        return snaps

    def restore_snapshot(self, snapshot_name: str) -> bool:
        snap_path = os.path.join(self.paths["snapshots"], snapshot_name)
        if not os.path.exists(snap_path):
            log_warn(f"Снэпшот не найден: {snapshot_name}")
            return False

        self.create_snapshot(label="before_restore")

        for fname in ["context.md", "pinned.md", "task.md"]:
            src = os.path.join(snap_path, fname)
            dst = self.paths["task_file"] if fname == "task.md" else \
                  os.path.join(self.base_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)

        knowledge_snap = os.path.join(snap_path, "knowledge")
        knowledge_dst  = os.path.join(self.base_path, "knowledge")
        if os.path.exists(knowledge_snap):
            if os.path.exists(knowledge_dst):
                shutil.rmtree(knowledge_dst)
            shutil.copytree(knowledge_snap, knowledge_dst)

        log_snapshot(f"Откат выполнен: {snapshot_name}")
        return True

    # ── Архив ─────────────────────────────────────────────────────────────

    def append_to_archive(self, messages: list) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.paths["archive"], f"msg_{timestamp}.md")
        content = "# Conversation Log\n\n"
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", "")
            content += f"**{role}**: {text}\n\n---\n\n"
        self.write_file(filepath, content, {"timestamp": timestamp})
        return filepath

    # ── Namespace-хелперы ─────────────────────────────────────────────────

    def set_project(self, project_name: str):
        self.paths["knowledge_project"] = os.path.join(
            self.base_path, "knowledge", project_name
        )
        self.paths["knowledge"] = self.paths["knowledge_project"]
        os.makedirs(self.paths["knowledge"], exist_ok=True)
        log_system(f"Проект переключён: {project_name}")
