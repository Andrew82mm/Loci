def count_tokens(text: str, model: str = "") -> int:
    """Count tokens using tiktoken when available, else approximate via char/4."""
    try:
        import tiktoken
        try:
            enc = tiktoken.encoding_for_model(model) if model else tiktoken.get_encoding("cl100k_base")
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4
