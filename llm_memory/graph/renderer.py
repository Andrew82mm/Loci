from llm_memory.models import Fact


def facts_to_markdown(facts: list[Fact]) -> str:
    """Render a list of facts as Markdown wikilink notation."""
    lines = []
    for fact in facts:
        if fact.object:
            lines.append(f"- [[{fact.subject}]] --({fact.predicate})--> [[{fact.object}]]")
        else:
            lines.append(f"- [[{fact.subject}]]: {fact.predicate}")
    return "\n".join(lines)
