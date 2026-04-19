import os
import json
import re
from datetime import datetime

from llm_memory.config import (
    SUMMARIZE_EVERY_N_MSG,
    SUMMARIZE_TOKEN_THRESHOLD,
    SUMMARIZE_MAX_MESSAGES,
    KEEP_RECENT_K,
    MODEL_SMART,
)
from llm_memory.models import Message
from llm_memory.buffer import ConversationBuffer
from llm_memory.summarizer import SummarizationPipeline
from llm_memory.llm.client import llm_client
from llm_memory.storage.filesystem import StorageManager
from llm_memory.rag.retriever import RAGEngine
from llm_memory.graph.extractor import KnowledgeGraph
from llm_memory.graph.index import GraphIndex
from llm_memory.colors import log_ok, log_warn


class MemoryEngine:
    def __init__(self) -> None:
        self.storage = StorageManager()

        db_path = os.path.join(self.storage.paths["system"], "relations.db")
        self.graph_index = GraphIndex(db_path)

        self.rag = RAGEngine(self.storage, graph_index=self.graph_index)
        self.kg  = KnowledgeGraph(self.storage)
        self.summarizer = SummarizationPipeline(self.storage, self.kg)

        self.buffer = ConversationBuffer(keep_recent_k=KEEP_RECENT_K)
        self._load_buffer()

    # ── Buffer persistence ────────────────────────────────────────────────

    def _load_buffer(self) -> None:
        path = self.storage.paths["history_file"]
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    self.buffer = ConversationBuffer.from_dicts(data, keep_recent_k=KEEP_RECENT_K)
                    return
                except (json.JSONDecodeError, KeyError):
                    pass
        self.buffer = ConversationBuffer(keep_recent_k=KEEP_RECENT_K)

    def _save_buffer(self) -> None:
        with open(self.storage.paths["history_file"], "w", encoding="utf-8") as f:
            json.dump(self.buffer.to_dicts(), f, ensure_ascii=False, indent=2)

    # ── Chat ──────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> tuple[str, list[str]]:
        self.buffer.add(Message(role="user", content=user_input, timestamp=datetime.now()))
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

        self.buffer.add(Message(role="assistant", content=response, timestamp=datetime.now()))
        self._save_buffer()

        if self._should_summarize():
            self._run_summarization_cycle()

        return response, references

    def _should_summarize(self) -> bool:
        if len(self.buffer) >= SUMMARIZE_MAX_MESSAGES:
            return True
        return self.buffer.total_tokens() >= SUMMARIZE_TOKEN_THRESHOLD

    def _extract_references(self, response: str) -> list[str]:
        match = re.search(r"References:\s*(.+)", response, re.IGNORECASE)
        if not match:
            return []
        raw = match.group(1).strip()
        if raw.lower() in ("none", "—", "-", ""):
            return []
        return [r.strip() for r in raw.split(",") if r.strip()]

    # ── Summarization ──────────────────────────────────────────────────────

    def _run_summarization_cycle(self) -> None:
        messages = self.buffer.to_summarize()
        if not messages:
            return
        result = self.summarizer.run_cycle(messages)
        if result.clear_buffer:
            self.buffer.clear_summarized()
            self._save_buffer()
            self.rag._sync_index()

    # ── Manual controls ───────────────────────────────────────────────────

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
        log_warn(f"Файл не найден: {filename}")
        return False

    def pin(self, text: str) -> None:
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
            self._load_buffer()
            self.rag.reload_after_restore()
        return ok

    def list_snapshots(self) -> list[dict]:
        return self.storage.list_snapshots()
