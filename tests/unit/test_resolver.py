import pytest

from loci.graph.resolver import EntityResolver, _cosine_similarity


@pytest.fixture
def resolver(tmp_path):
    db_path = str(tmp_path / "_system" / "entities.db")
    return EntityResolver(db_path)


@pytest.fixture
def resolver_with_embed(tmp_path):
    """Resolver with a trivial embedding: one-hot by first character."""
    def embed(name: str) -> list[float]:
        vec = [0.0] * 26
        ch = name[0].lower() if name else "a"
        idx = ord(ch) - ord("a")
        if 0 <= idx < 26:
            vec[idx] = 1.0
        return vec

    db_path = str(tmp_path / "_system" / "entities_emb.db")
    return EntityResolver(db_path, embed_fn=embed)


# ── Normalisation ──────────────────────────────────────────────────────────

def test_normalize_strips_and_lowercases():
    assert EntityResolver.normalize("  Alice ") == "alice"
    assert EntityResolver.normalize("АНДРЕЙ") == "андрей"


# ── Case 1: exact match ────────────────────────────────────────────────────

def test_exact_match_returns_canonical(resolver):
    e1 = resolver.resolve("Alice")   # creates canonical "alice"
    e2 = resolver.resolve("alice")   # same normalised name
    assert e1.name == e2.name == "alice"


# ── Case 2: alias lookup ───────────────────────────────────────────────────

def test_alias_lookup(resolver):
    resolver.add_alias("Алиса", "alice")
    entity = resolver.resolve("Алиса")
    assert entity.name == "alice"


# ── Case 3: embedding similarity ──────────────────────────────────────────

def test_embedding_similarity_finds_canonical(resolver_with_embed):
    resolver_with_embed.resolve("alice")   # registers "alice" as canonical
    # "alison" starts with 'a' → same one-hot vector → cosine = 1.0 ≥ 0.85
    entity = resolver_with_embed.resolve("alison")
    assert entity.name == "alice"


def test_embedding_below_threshold_creates_new(tmp_path):
    """Vectors for 'alice' (a) and 'bob' (b) are orthogonal → no match."""
    def embed(name: str) -> list[float]:
        vec = [0.0] * 26
        ch = name[0].lower() if name else "a"
        idx = ord(ch) - ord("a")
        if 0 <= idx < 26:
            vec[idx] = 1.0
        return vec

    from loci.graph.resolver import EntityResolver
    db = str(tmp_path / "e.db")
    r = EntityResolver(db, embed_fn=embed)
    r.resolve("alice")
    entity = r.resolve("bob")
    assert entity.name == "bob"  # new canonical


# ── Case 4: new canonical creation ────────────────────────────────────────

def test_new_entity_creation(resolver):
    entity = resolver.resolve("Completely New Entity")
    assert entity.name == "completely new entity"
    assert "completely new entity" in resolver.list_canonicals()


# ── Helpers ────────────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0
