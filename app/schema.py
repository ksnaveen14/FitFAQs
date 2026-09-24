"""
schema.py — The Chunk contract.

WHY THIS EXISTS (interview answer):
  In a multi-file RAG pipeline, chunk.py produces chunks, embed.py consumes
  them, retrieve.py filters them, generate.py cites them. If each file has
  its own ad-hoc dict structure, one rename causes silent bugs everywhere.
  A single dataclass forces every module to speak the same language.
  This is the "schema-first" pattern used in production ML pipelines.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Chunk:
    """
    The atomic unit that flows through the entire RAG pipeline.

    Fields:
        chunk_id   : globally unique identifier  (used as FAISS lookup key)
        text       : the actual text content
        source     : filename it came from       (used for citations in H4)
        page       : page number if available
        chunk_index: position within that document (0-based)
        token_count: approximate token count
        embedding  : populated by embed.py, None before that step
        score      : populated by retrieve.py (relevance score after fusion)
    """
    chunk_id   : str
    text       : str
    source     : str
    page       : int = 0
    chunk_index: int = 0
    token_count: int = 0

    # These are populated later in the pipeline — NOT by chunk.py
    embedding  : Optional[list] = field(default=None, repr=False)
    score      : float = 0.0

    def to_dict(self) -> dict:
        """Serialize for JSON storage. Excludes embedding (too large)."""
        return {
            "chunk_id"   : self.chunk_id,
            "text"       : self.text,
            "source"     : self.source,
            "page"       : self.page,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "score"      : self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        """Deserialize from JSON storage."""
        return cls(
            chunk_id    = d["chunk_id"],
            text        = d["text"],
            source      = d["source"],
            page        = d.get("page", 0),
            chunk_index = d.get("chunk_index", 0),
            token_count = d.get("token_count", 0),
            score       = d.get("score", 0.0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "Chunk":
        return cls.from_dict(json.loads(s))
