import math
import os
import sqlite3
import unicodedata
from typing import Callable, Optional

from loci.models import Entity

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    name      TEXT PRIMARY KEY,
    canonical TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_canonical ON entities(canonical);
"""

# Threshold for cosine similarity to consider two entity names the same
_EMBED_THRESHOLD = 0.85


class EntityResolver:
    """
    Resolves entity name variants to a single canonical form.

    Resolution order:
      1. Exact match (after NFC + lower + strip normalisation)
      2. Alias lookup (name stored with a different canonical)
      3. Embedding cosine similarity >= _EMBED_THRESHOLD (optional)
      4. Create new canonical

    The embed_fn parameter is optional. When provided it should accept a
    string and return a list[float] embedding vector.  When absent,
    embedding-based matching is skipped.
    """

    def __init__(
        self,
        db_path: str,
        embed_fn: Optional[Callable[[str], list[float]]] = None,
    ) -> None:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._embed_fn = embed_fn
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_CREATE_SQL)
        self._conn.commit()

    # ── Public API ─────────────────────────────────────────────────────────

    @staticmethod
    def normalize(name: str) -> str:
        return unicodedata.normalize("NFC", name).lower().strip()

    def resolve(self, name: str) -> Entity:
        norm = self.normalize(name)

        # 1 & 2: exact / alias lookup (stored in same table)
        cur = self._conn.execute(
            "SELECT canonical FROM entities WHERE name = ?", (norm,)
        )
        row = cur.fetchone()
        if row:
            return Entity(name=row[0])

        # 3: embedding similarity
        if self._embed_fn is not None:
            canonical = self._find_by_embedding(norm)
            if canonical:
                self._conn.execute(
                    "INSERT OR IGNORE INTO entities (name, canonical) VALUES (?, ?)",
                    (norm, canonical),
                )
                self._conn.commit()
                return Entity(name=canonical, aliases=[name])

        # 4: create new canonical
        self._conn.execute(
            "INSERT OR IGNORE INTO entities (name, canonical) VALUES (?, ?)",
            (norm, norm),
        )
        self._conn.commit()
        return Entity(name=norm)

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register alias as pointing to canonical."""
        norm_alias = self.normalize(alias)
        norm_canonical = self.normalize(canonical)
        self._conn.execute(
            "INSERT OR REPLACE INTO entities (name, canonical) VALUES (?, ?)",
            (norm_alias, norm_canonical),
        )
        self._conn.commit()

    def list_canonicals(self) -> list[str]:
        cur = self._conn.execute("SELECT DISTINCT canonical FROM entities")
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_by_embedding(self, name: str) -> str | None:
        canonicals = self.list_canonicals()
        if not canonicals:
            return None
        assert self._embed_fn is not None
        query_emb = self._embed_fn(name)
        best_canonical: str | None = None
        best_score = 0.0
        for canonical in canonicals:
            score = _cosine_similarity(query_emb, self._embed_fn(canonical))
            if score > best_score:
                best_score = score
                best_canonical = canonical
        if best_score >= _EMBED_THRESHOLD:
            return best_canonical
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
