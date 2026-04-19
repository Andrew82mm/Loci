from llm_memory.models import CycleResult, Message
from llm_memory.llm.client import llm_client
from llm_memory.config import MODEL_FAST
from llm_memory.colors import log_ok, log_warn, log_system, separator


class SummarizationError(Exception):
    pass


class SummarizationPipeline:
    """
    Orchestrates one summarization cycle.
    Critical invariant: buffer is never cleared unless the cycle fully succeeds.
    """

    def __init__(self, storage, extractor) -> None:
        self.storage = storage
        self.extractor = extractor

    def run_cycle(self, messages: list[Message]) -> CycleResult:
        separator()
        log_system("Запускаю цикл суммаризации...")

        snapshot_name = self.storage.create_snapshot(label="pre_summarize")

        full_text = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages
        )

        try:
            task = self._update_task(full_text)
            summary = self._summarize(full_text)

            if not summary or summary.startswith("Error:"):
                raise SummarizationError(f"invalid summary: {summary!r}")

            if task and not task.startswith("Error:"):
                self.storage.write_file(self.storage.paths["task_file"], task)
                log_ok(f"Задача обновлена: {task[:80]}")

            self.storage.write_file(self.storage.paths["context_file"], summary)
            log_ok("Контекст обновлён.")

            self.extractor.extract_and_save_facts(summary)

            raw_dicts = [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
                for m in messages
            ]
            self.storage.append_to_archive(raw_dicts)
            log_ok(f"Архивировано {len(messages)} сообщений.")

            log_system("Суммаризация завершена.")
            separator()
            return CycleResult(ok=True, clear_buffer=True)

        except Exception as exc:
            log_warn(f"Суммаризация провалилась: {exc}. Буфер сохранён.")
            try:
                self.storage.restore_snapshot(snapshot_name, silent=True)
            except Exception:
                pass
            separator()
            return CycleResult(ok=False, clear_buffer=False, error=str(exc))

    def _update_task(self, full_text: str) -> str:
        prompt = (
            "Analyze the conversation history below and define or update "
            "the main goal in ONE concise sentence.\n\n" + full_text
        )
        return llm_client.generate(
            MODEL_FAST, "You are an analyst. Be concise.", prompt, temperature=0.0
        )

    def _summarize(self, full_text: str) -> str:
        prompt = (
            "Summarize the following conversation. Keep only:\n"
            "- Key facts and decisions\n"
            "- Open questions\n"
            "- Important entities mentioned\n"
            "Remove all chit-chat and filler.\n\n" + full_text
        )
        return llm_client.generate(
            MODEL_FAST, "You are a concise summarizer.", prompt, temperature=0.0
        )
