import os
import re
import sqlite3
from datetime import datetime
from typing import Literal

from loci.models import Fact

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT,
    source_file TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    extracted_at TEXT NOT NULL,
    UNIQUE(subject, predicate, object)
);
CREATE INDEX IF NOT EXISTS idx_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_object ON relations(object);
"""


class GraphIndex:
    def __init__(self, db_path: str) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_CREATE_SQL)
        self._conn.commit()

    def add(self, fact: Fact) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO relations
                (subject, predicate, object, source_file, confidence, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                fact.subject,
                fact.predicate,
                fact.object,
                "",
                fact.confidence,
                fact.extracted_at.isoformat(),
            ),
        )
        self._conn.commit()

    def neighbors(
        self, entity: str, direction: Literal["out", "in", "both"] = "both"
    ) -> list[str]:
        result: set[str] = set()
        if direction in ("out", "both"):
            cur = self._conn.execute(
                "SELECT object FROM relations WHERE subject = ? AND object IS NOT NULL",
                (entity,),
            )
            result.update(row[0] for row in cur.fetchall())
        if direction in ("in", "both"):
            cur = self._conn.execute(
                "SELECT subject FROM relations WHERE object = ?",
                (entity,),
            )
            result.update(row[0] for row in cur.fetchall())
        return list(result)

    def query(
        self,
        subject: str | None = None,
        predicate: str | None = None,
        obj: str | None = None,
    ) -> list[Fact]:
        conditions = []
        params: list = []
        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate)
        if obj is not None:
            conditions.append("object = ?")
            params.append(obj)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cur = self._conn.execute(
            f"SELECT subject, predicate, object, source_file, confidence, extracted_at FROM relations {where}",
            params,
        )
        facts = []
        for row in cur.fetchall():
            facts.append(
                Fact(
                    subject=row[0],
                    predicate=row[1],
                    object=row[2],
                    raw_text=f"{row[0]} {row[1]} {row[2] or ''}".strip(),
                    source_chunk=row[3],
                    extracted_at=datetime.fromisoformat(row[5]),
                    confidence=row[4],
                )
            )
        return facts

    # ── Compat shim: KnowledgeGraph interface used by RAGEngine ──────────

    def get_connected_nodes(self, filepath: str) -> list[str]:
        """Return entity names linked from the given file path via regex (legacy compat)."""
        if not os.path.exists(filepath):
            return []
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        return list(set(re.findall(r"\[\[(.*?)\]\]", content)))

    def get_entity_path(self, entity_name: str, knowledge_dir: str = "") -> str | None:
        if not knowledge_dir:
            return None
        safe_name = re.sub(r'[/\\:*?"<>|]', "_", entity_name)
        path = os.path.join(knowledge_dir, f"{safe_name}.md")
        return path if os.path.exists(path) else None

    def close(self) -> None:
        self._conn.close()
