import re
from llm_memory.config import CHUNK_MAX

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def _split_by_paragraphs(text: str, max_size: int) -> list[str]:
    """Split text by double newlines; merge short paragraphs, split large ones."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_size:
            if current:
                chunks.append(current)
                current = ""
            # Sliding window fallback
            start = 0
            while start < len(para):
                chunks.append(para[start : start + max_size])
                start += max_size - 100
        elif current and len(current) + len(para) + 2 > max_size:
            chunks.append(current)
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current:
        chunks.append(current)
    return chunks


def chunk_markdown(text: str, source: str = "") -> list[dict]:
    """
    Split markdown text into semantic chunks.
    Returns list of {content, heading_path, source}.
    """
    if not text.strip():
        return []

    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [{"content": chunk, "heading_path": [], "source": source}
                for chunk in _split_by_paragraphs(text, CHUNK_MAX)]

    chunks: list[dict] = []
    heading_stack: list[tuple[int, str]] = []  # (level, title)

    sections: list[tuple[list[str], str]] = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        # Update heading breadcrumb
        heading_stack = [(l, t) for (l, t) in heading_stack if l < level]
        heading_stack.append((level, title))
        breadcrumb = [t for (_, t) in heading_stack]

        sections.append((breadcrumb, section_text))

    for breadcrumb, content in sections:
        if len(content) <= CHUNK_MAX:
            chunks.append({"content": content, "heading_path": breadcrumb, "source": source})
        else:
            for sub in _split_by_paragraphs(content, CHUNK_MAX):
                chunks.append({"content": sub, "heading_path": breadcrumb, "source": source})

    return chunks
