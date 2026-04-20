from loci.colors import log_warn
from loci.config import MODEL_FAST
from loci.llm.client import llm_client
from loci.models import Fact

_JUDGE_SYSTEM = "You are a fact-checking assistant. Answer only 'yes' or 'no'."

_JUDGE_PROMPT = """\
Source text:
{source}

Fact: {raw_text}

Is this fact directly supported by the source text above?
Answer with only "yes" or "no".
"""


class FactJudge:
    """Validates extracted facts against their source chunk using LLM-as-judge.

    Each fact is checked individually. Facts where the judge answers 'no' are
    discarded.  This is intentionally a conservative filter: ambiguous answers
    default to keeping the fact.
    """

    def validate(self, facts: list[Fact], source_chunk: str) -> list[Fact]:
        if not facts:
            return []
        verified: list[Fact] = []
        for fact in facts:
            if self._is_supported(fact, source_chunk):
                verified.append(fact)
            else:
                log_warn(f"[Judge] Rejected fact: {fact.raw_text[:80]}")
        return verified

    def _is_supported(self, fact: Fact, source_chunk: str) -> bool:
        prompt = _JUDGE_PROMPT.format(source=source_chunk, raw_text=fact.raw_text)
        response = llm_client.generate(
            MODEL_FAST,
            _JUDGE_SYSTEM,
            prompt,
            temperature=0.0,
        )
        # Treat errors and ambiguous responses as "keep" (fail-open)
        if response.startswith("Error:"):
            return True
        return not response.strip().lower().startswith("no")
