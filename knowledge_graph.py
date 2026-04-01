# knowledge_graph.py
import os
import re
from llm_client import llm_client
from config import MODEL_FAST
from storage import StorageManager
from colors import log_knowledge, log_warn

class KnowledgeGraph:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    def extract_and_save_facts(self, text_chunk: str) -> str:
        """
        L2: Извлекает атомарные факты из текста и сохраняет в knowledge/*.md.
        Возвращает сырой ответ LLM.
        """
        prompt = f"""Extract atomic facts and entities from the text below.
Format output strictly as a list of Markdown entries.
Use WikiLinks [[Entity Name]] for all entities.
If a relation exists, format it as: - [[Entity A]] --(relation)--> [[Entity B]]
If it's a standalone fact about an entity, format it as: - [[Entity]]: <fact>

Text:
{text_chunk}

Output (only the list items, one per line):"""

        response = llm_client.generate(
            MODEL_FAST,
            "You are a knowledge graph extraction engine. Output ONLY the list items, no preamble.",
            prompt,
            temperature=0.0,
        )

        if response.startswith("Error:"):
            log_warn(f"KG extraction failed: {response}")
            return response

        self._parse_and_update_files(response)
        return response

    def _parse_and_update_files(self, markdown_list: str):
        """
        Разбирает список фактов и дописывает их в файлы сущностей.
        Дедупликация — через StorageManager.append_to_file.
        """
        lines = [l.strip() for l in markdown_list.splitlines() if l.strip()]

        # Группируем строки по упомянутым сущностям
        entity_lines: dict[str, list[str]] = {}

        for line in lines:
            # Находим все [[Entity]] в строке
            entities = re.findall(r'\[\[(.*?)\]\]', line)
            if not entities:
                continue
            # Первая сущность в строке считается «владельцем» факта
            primary = entities[0].strip()
            entity_lines.setdefault(primary, []).append(line)

        for entity, fact_lines in entity_lines.items():
            safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity)
            filepath = os.path.join(self.storage.paths["knowledge"], f"{safe_name}.md")

            # Создаём файл сущности, если его нет
            if not os.path.exists(filepath):
                log_knowledge(f"Новая сущность: {entity}")
                self.storage.write_file(
                    filepath,
                    f"# {entity}\n\n",
                    {"type": "entity", "pinned": False}
                )

            # Дописываем только новые строки (дедупликация внутри append_to_file)
            self.storage.append_to_file(filepath, fact_lines)
            log_knowledge(f"  +{len(fact_lines)} факт(ов) → {safe_name}.md")

    def get_connected_nodes(self, filepath: str) -> list[str]:
        """Возвращает имена сущностей, на которые ссылается файл."""
        _, content = self.storage.read_file(filepath)
        links = re.findall(r'\[\[(.*?)\]\]', content)
        return list(set(links))

    def get_entity_path(self, entity_name: str) -> str | None:
        """Возвращает путь к файлу сущности, если он существует."""
        safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity_name)
        path = os.path.join(self.storage.paths["knowledge"], f"{safe_name}.md")
        return path if os.path.exists(path) else None
