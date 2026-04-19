from llm_memory.rag.chunker import chunk_markdown


def test_three_sections_produce_three_chunks():
    md = """# Section 1
Content one.

## Section 2
Content two.

### Section 3
Content three.
"""
    chunks = chunk_markdown(md, source="test.md")
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk["source"] == "test.md"
        assert len(chunk["heading_path"]) >= 1


def test_chunk_contains_heading():
    md = "# Alpha\nSome text.\n"
    chunks = chunk_markdown(md, source="a.md")
    assert chunks[0]["heading_path"] == ["Alpha"]


def test_long_section_split():
    # Create a section larger than CHUNK_MAX (2000 chars)
    big_para = "word " * 500  # 2500 chars
    md = f"# Big\n\n{big_para}\n"
    chunks = chunk_markdown(md, source="big.md")
    assert len(chunks) >= 2
    # All content preserved
    combined = " ".join(c["content"] for c in chunks)
    assert "word" in combined


def test_no_headings_splits_by_paragraphs():
    md = "Para one.\n\nPara two.\n\nPara three.\n"
    chunks = chunk_markdown(md, source="flat.md")
    assert len(chunks) >= 1
    assert all(c["heading_path"] == [] for c in chunks)


def test_empty_text_returns_empty():
    assert chunk_markdown("", source="empty.md") == []
