import json
import os
import re
from datetime import datetime

from loci.colors import log_knowledge, log_warn
from loci.config import ENABLE_FACT_VALIDATION, MODEL_FAST
from loci.llm.client import llm_client
from loci.models import Fact
from loci.storage.filesystem import StorageManager

_JSON_EXTRACTION_PROMPT = """\
Extract facts from the text below as JSON.
Return a JSON object with a "facts" array. Each item must have:
  "subject"   : entity name (string)
  "predicate" : relationship type, e.g. "uses", "is", "works_at" (string)
  "object"    : related entity or value, null if standalone (string | null)
  "raw_text"  : the original sentence that supports this fact (string)

Return ONLY valid JSON, no other text.
If no facts are found, return {"facts": []}.

Text:
"""


class KnowledgeGraph:
    def __init__(self, storage: StorageManager):
        self.storage = storage

    # ── Primary entry point ────────────────────────────────────────────────

    def extract_and_save_facts(self, text_chunk: str) -> str:
        """Extract facts with a single LLM call.

        Tries JSON parsing first; if that yields nothing, falls back to
        legacy WikiLink-markdown parsing on the same response — no second
        LLM round-trip required.  Returns the raw LLM response.
        """
        response = llm_client.generate(
            MODEL_FAST,
            "You are a knowledge extraction engine. Output ONLY valid JSON.",
            _JSON_EXTRACTION_PROMPT + text_chunk,
            temperature=0.0,
        )
        if response.startswith("Error:"):
            log_warn(f"KG extraction failed: {response}")
            return response

        facts = self._parse_json_facts(response, text_chunk)
        if facts:
            if ENABLE_FACT_VALIDATION:
                from loci.graph.judge import FactJudge
                facts = FactJudge().validate(facts, text_chunk)
            self._save_facts_to_files(facts)
            return response

        # Fallback: same response, legacy WikiLink-markdown parsing
        self._parse_and_update_files(response)
        return response

    # ── Structured JSON extraction (standalone helper) ─────────────────────

    def extract_facts_json(self, text_chunk: str) -> list[Fact]:
        """Call LLM and parse structured JSON facts. Returns [] on any failure."""
        response = llm_client.generate(
            MODEL_FAST,
            "You are a knowledge extraction engine. Output ONLY valid JSON.",
            _JSON_EXTRACTION_PROMPT + text_chunk,
            temperature=0.0,
        )
        if response.startswith("Error:"):
            log_warn(f"Structured extraction LLM error: {response}")
            return []
        return self._parse_json_facts(response, text_chunk)

    def _parse_json_facts(self, response: str, source_chunk: str) -> list[Fact]:
        data: dict | None = None

        # Try direct parse
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract a JSON object from a response that has surrounding text
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if data is None:
            log_warn("Could not parse JSON from extraction response")
            return []

        facts: list[Fact] = []
        for item in data.get("facts", []):
            try:
                fact = Fact(
                    subject=str(item["subject"]),
                    predicate=str(item["predicate"]),
                    object=item.get("object"),
                    raw_text=str(
                        item.get(
                            "raw_text",
                            f"{item['subject']} {item['predicate']} {item.get('object', '')}",
                        )
                    ),
                    source_chunk=source_chunk[:200],
                    extracted_at=datetime.now(),
                    confidence=float(item.get("confidence", 1.0)),
                )
                facts.append(fact)
            except (KeyError, ValueError) as exc:
                log_warn(f"Skipping invalid fact item: {exc}")

        return facts

    def _save_facts_to_files(self, facts: list[Fact]) -> None:
        entity_lines: dict[str, list[str]] = {}
        for fact in facts:
            entity_lines.setdefault(fact.subject, []).append(fact.raw_text)

        for entity, lines in entity_lines.items():
            safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity)
            filepath = os.path.join(self.storage.paths["knowledge"], f"{safe_name}.md")
            if not os.path.exists(filepath):
                log_knowledge(f"Новая сущность: {entity}")
                self.storage.write_file(
                    filepath,
                    f"# {entity}\n\n",
                    {"type": "entity", "pinned": False},
                )
            self.storage.append_to_file(filepath, lines)
            log_knowledge(f"  +{len(lines)} факт(ов) → {safe_name}.md")

    # ── Legacy markdown extraction (fallback) ──────────────────────────────

    def _legacy_extract_and_save(self, text_chunk: str) -> str:
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

    def _parse_and_update_files(self, markdown_list: str) -> None:
        lines = [line.strip() for line in markdown_list.splitlines() if line.strip()]
        entity_lines: dict[str, list[str]] = {}
        for line in lines:
            entities = re.findall(r"\[\[(.*?)\]\]", line)
            if not entities:
                continue
            primary = entities[0].strip()
            entity_lines.setdefault(primary, []).append(line)
        for entity, fact_lines in entity_lines.items():
            safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity)
            filepath = os.path.join(self.storage.paths["knowledge"], f"{safe_name}.md")
            if not os.path.exists(filepath):
                log_knowledge(f"Новая сущность: {entity}")
                self.storage.write_file(
                    filepath,
                    f"# {entity}\n\n",
                    {"type": "entity", "pinned": False},
                )
            self.storage.append_to_file(filepath, fact_lines)
            log_knowledge(f"  +{len(fact_lines)} факт(ов) → {safe_name}.md")

    # ── Compat helpers ─────────────────────────────────────────────────────

    def get_connected_nodes(self, filepath: str) -> list[str]:
        _, content = self.storage.read_file(filepath)
        links = re.findall(r"\[\[(.*?)\]\]", content)
        return list(set(links))

    def get_entity_path(self, entity_name: str) -> str | None:
        safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity_name)
        path = os.path.join(self.storage.paths["knowledge"], f"{safe_name}.md")
        return path if os.path.exists(path) else None
