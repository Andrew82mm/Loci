from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)


class Entity(BaseModel):
    name: str
    aliases: list[str] = []
    file_path: str | None = None


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str | None = None
    raw_text: str
    source_chunk: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    confidence: float = 1.0


class RetrievedChunk(BaseModel):
    content: str
    source: str
    score: float
    reason: Literal["vector", "graph_hop", "pinned", "task", "summary"]


class Snapshot(BaseModel):
    name: str
    timestamp: datetime
    label: str = ""
    path: str
    includes_chroma: bool = False
    parent_snapshot: str | None = None


class CycleResult(BaseModel):
    ok: bool
    clear_buffer: bool
    error: str = ""
