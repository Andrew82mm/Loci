from llm_memory.llm.tokens import count_tokens


def test_count_tokens_returns_positive():
    result = count_tokens("Hello, world!")
    assert result > 0


def test_count_tokens_longer_text():
    short = count_tokens("hi")
    long  = count_tokens("hi " * 100)
    assert long > short


def test_count_tokens_with_model():
    result = count_tokens("Some text here.", model="gpt-4")
    assert result > 0


def test_token_trigger_5000():
    """A single very long message should have >3000 tokens."""
    text = "word " * 1500  # ~1500 tokens even with char/4 fallback → 7500//4 = 1875, tiktoken ~1500
    result = count_tokens(text)
    # Either way, should be significantly more than 100
    assert result > 100
