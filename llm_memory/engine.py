import json
import os
import re
from datetime import datetime
from llm_memory.config import SUMMARIZE_EVERY_N_MSG, MODEL_SMART, MODEL_FAST
from llm_memory.llm.client import llm_client
from llm_memory.storage.filesystem import StorageManager
from llm_memory.rag.retriever import RAGEngine
from llm_memory.graph.extractor import KnowledgeGraph
from llm_memory.colors import log_system, log_ok, log_warn, separator


class MemoryEngine:
    def __init__(self):
        self.storage = StorageManager()
        self.rag     = RAGEngine(self.storage)
        self.kg      = KnowledgeGraph(self.storage)
        self.buffer  = self._load_buffer()

    # ── Буфер ─────────────────────────────────────────────────────────────

    def _load_buffer(self) -> list:
        path = self.storage.paths["history_file"]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def _save_buffer(self):
        with open(self.storage.paths["history_file"], "w", encoding="utf-8") as f:
            json.dump(self.buffer, f, ensure_ascii=False, indent=2)

    # ── Основной чат ──────────────────────────────────────────────────────

    def chat(self, user_input: str) -> tuple[str, list[str]]:
        """Возвращает (response_text, references)."""
        self.buffer.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_buffer()

        rag_contexts  = self.rag.search(user_input)
        pinned        = self.storage.read_file(self.storage.paths["pinned_file"])[1]
        current_task  = self.storage.read_file(self.storage.paths["task_file"])[1]
        summary       = self.storage.read_file(self.storage.paths["context_file"])[1]

        rag_block = "\n\n".join(rag_contexts) if rag_contexts else "None"

        system_prompt = f"""You are an AI assistant with persistent long-term memory.

## Current Main Task
{current_task or "Not defined yet"}

## Pinned Information (High Priority)
{pinned or "None"}

## Context Summary
{summary or "None"}

## Relevant Knowledge Retrieved
{rag_block}

---
Instructions:
- Answer the user's question considering all the context above.
- At the very end of your response, add a line starting with "References:" followed by
  a comma-separated list of the source file labels you actually used from
  "Relevant Knowledge Retrieved". If you used none, write "References: none".
"""

        response = llm_client.generate(MODEL_SMART, system_prompt, user_input)

        references = self._extract_references(response)

        self.buffer.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_buffer()

        if len(self.buffer) >= SUMMARIZE_EVERY_N_MSG * 2:
            self._run_summarization_cycle()

        return response, references

    def _extract_references(self, response: str) -> list[str]:
        match = re.search(r"References:\s*(.+)", response, re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.lower() in ("none", "—", "-", ""):
            return []
        return [r.strip() for r in raw.split(",") if r.strip()]

    # ── Суммаризация ──────────────────────────────────────────────────────

    def _run_summarization_cycle(self):
        separator()
        log_system("Запускаю цикл суммаризации...")

        self.storage.create_snapshot()

        full_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.buffer
        )

        task_prompt = (
            "Analyze the conversation history below and define or update "
            "the main goal in ONE concise sentence.\n\n" + full_text
        )
        task = llm_client.generate(
            MODEL_FAST, "You are an analyst. Be concise.", task_prompt, temperature=0.0
        )
        if not task.startswith("Error:"):
            self.storage.write_file(self.storage.paths["task_file"], task)
            log_ok(f"Задача обновлена: {task[:80]}...")

        summary_prompt = (
            "Summarize the following conversation. Keep only:\n"
            "- Key facts and decisions\n"
            "- Open questions\n"
            "- Important entities mentioned\n"
            "Remove all chit-chat and filler.\n\n" + full_text
        )
        summary = llm_client.generate(
            MODEL_FAST, "You are a concise summarizer.", summary_prompt, temperature=0.0
        )
        if not summary.startswith("Error:"):
            self.storage.write_file(self.storage.paths["context_file"], summary)
            log_ok("Контекст обновлён.")

            self.kg.extract_and_save_facts(summary)

        self.storage.append_to_archive(self.buffer)
        log_ok(f"Архивировано {len(self.buffer)} сообщений.")

        self.rag._sync_index()

        self.buffer = []
        self._save_buffer()
        log_system("Суммаризация завершена, буфер очищен.")
        separator()

    # ── Ручное управление ─────────────────────────────────────────────────

    def manual_edit(self, filename: str, new_content: str) -> bool:
        if filename in ("pinned", "pinned.md"):
            target = self.storage.paths["pinned_file"]
        elif filename in ("context", "context.md"):
            target = self.storage.paths["context_file"]
        elif filename in ("task", "task.md"):
            target = self.storage.paths["task_file"]
        else:
            safe = re.sub(r'[/\\:*?"<>|]', "_", filename.removesuffix(".md"))
            target = os.path.join(self.storage.paths["knowledge"], f"{safe}.md")

        if os.path.exists(target):
            self.storage.create_snapshot(label="manual_edit")
            self.storage.write_file(target, new_content)
            self.rag.index_file(target)
            log_ok(f"Файл обновлён и переиндексирован: {os.path.basename(target)}")
            return True
        else:
            log_warn(f"Файл не найден: {filename}")
            return False

    def pin(self, text: str):
        _, current = self.storage.read_file(self.storage.paths["pinned_file"])
        new_content = current + f"\n- {text}"
        self.storage.write_file(self.storage.paths["pinned_file"], new_content)
        self.rag.index_file(self.storage.paths["pinned_file"])
        log_ok(f"Закреплено: {text[:60]}")

    def rollback(self, snapshot_name: str = "") -> bool:
        if not snapshot_name:
            snaps = self.storage.list_snapshots()
            real_snaps = [s for s in snaps if "before_restore" not in s["name"]]
            if not real_snaps:
                log_warn("Нет доступных снэпшотов для отката.")
                return False
            snapshot_name = real_snaps[0]["name"]

        ok = self.storage.restore_snapshot(snapshot_name)
        if ok:
            self.buffer = self._load_buffer()
            self.rag.reindex_all()
        return ok

    def list_snapshots(self) -> list[dict]:
        return self.storage.list_snapshots()
