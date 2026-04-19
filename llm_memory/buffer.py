from datetime import datetime
from llm_memory.models import Message
from llm_memory.llm.tokens import count_tokens


class ConversationBuffer:
    def __init__(self, keep_recent_k: int = 4):
        self._messages: list[Message] = []
        self.keep_recent_k = keep_recent_k

    # ── Persistence helpers ───────────────────────────────────────────────

    @classmethod
    def from_dicts(cls, data: list[dict], keep_recent_k: int = 4) -> "ConversationBuffer":
        buf = cls(keep_recent_k=keep_recent_k)
        for d in data:
            ts = d.get("timestamp")
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts)
                except ValueError:
                    ts = datetime.now()
            elif ts is None:
                ts = datetime.now()
            buf._messages.append(
                Message(role=d["role"], content=d["content"], timestamp=ts)
            )
        return buf

    def to_dicts(self) -> list[dict]:
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()}
            for m in self._messages
        ]

    # ── Core API ──────────────────────────────────────────────────────────

    def add(self, msg: Message) -> None:
        self._messages.append(msg)

    def all(self) -> list[Message]:
        return list(self._messages)

    def recent(self, k: int) -> list[Message]:
        return self._messages[-k:] if k > 0 else []

    def to_summarize(self) -> list[Message]:
        """Returns all messages except the last keep_recent_k."""
        if len(self._messages) <= self.keep_recent_k:
            return []
        return self._messages[: -self.keep_recent_k]

    def clear_summarized(self) -> None:
        """Keep only the last keep_recent_k messages after a successful summarization cycle."""
        self._messages = self._messages[-self.keep_recent_k :]

    def total_tokens(self) -> int:
        return sum(count_tokens(m.content) for m in self._messages)

    def __len__(self) -> int:
        return len(self._messages)
