"""
chunk.py — Text chunking.

  Takes raw (filename, page, text) tuples and
  splits them into overlapping chunks using a sliding window.
  Returns :  List[Chunk].
"""

import re
import hashlib
import json
import logging
from pathlib import Path
from typing import List
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Chunk:
    """
    Fields:
        chunk_id   : globally unique identifier  (used as FAISS lookup key)
        text       : the actual text content
        source     : filename it came from
        page       : page number if available
        chunk_index: position within that document
        token_count: approximate token count
    """
    chunk_id   : str
    text       : str
    source     : str
    page       : int = 0
    chunk_index: int = 0
    token_count: int = 0
    embedding  : Optional[list] = field(default=None, repr=False)
    score      : float = 0.0

    def to_dict(self) -> dict:
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
    
log = logging.getLogger(__name__)
data_chunks = Path(__file__).parent / "data" / "chunks"
CHUNKS_FILE = data_chunks / "chunks.jsonl"
CHUNK_SIZE    = 512   
CHUNK_OVERLAP = 50  

def _word_tokenise(text: str) -> list[str]:
    """
    Simple whitespace tokenizer. Can be replaced with something more sophisticated if needed.
    """
    return text.split()


def _make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """
    Deterministic ID so re-running produces the same IDs.
    SHA256 of the key fields truncated to 12 chars.
    """
    raw = f"{source}|{page}|{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def chunk_text(
    source: str,
    page: int,
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Sliding-window chunker. Splits text into overlapping word-level windows.

    Args:
        source     : filename (for citation)
        page       : page number (for citation)
        text       : raw text to chunk
        chunk_size : target window size in words
        overlap    : number of words to repeat at the start of next chunk

    Returns:
        List[Chunk] with all fields set except embedding (None) and score (0.0)
    """
    # Normalise whitespace - PDFs often have \n in odd places
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = _word_tokenise(text)
    if not words:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_index = 0

    stride = chunk_size - overlap
    
    # Guard: if stride ≤ 0, we'd loop forever
    if stride <= 0:
        log.error("chunk_size must be greater than overlap. Aborting.")
        return []

    while start < len(words):
        end = min(start + chunk_size, len(words))
        window_words = words[start:end]
        window_text = " ".join(window_words).strip()
 
        if window_text:
            chunk = Chunk(
                chunk_id    = _make_chunk_id(source, page, chunk_index),
                text        = window_text,
                source      = source,
                page        = page,
                chunk_index = chunk_index,
                token_count = len(window_words),
            )
            chunks.append(chunk)
            chunk_index += 1

        # Slide forward by (chunk_size - overlap)
        start += stride

    return chunks


def chunk_documents(
    pages: list[tuple[str, int, str]],
) -> List[Chunk]:
    """
    Chunk all pages returned by ingest.load_documents().

    Args:
        pages: list of (filename, page_number, raw_text)

    Returns:
        flat List[Chunk] across all documents
    """
    all_chunks: List[Chunk] = []

    for source, page, text in pages:
        page_chunks = chunk_text(source, page, text)
        all_chunks.extend(page_chunks)
        log.info(f"{source} page {page}: {len(page_chunks)} chunks")

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: List[Chunk]) -> None:
    """
    Persist chunks to JSONL (one JSON object per line).
    JSONL is append-friendly and readable with standard tools.
    embed.py loads from here if you run hours separately.
    """
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.to_json() + "\n")
    log.info(f"Saved {len(chunks)} chunks → {CHUNKS_FILE}")


def load_chunks() -> List[Chunk]:
    """Load chunks from disk. Used by embed.py and retrieve.py."""
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"No chunks file at {CHUNKS_FILE}. Run chunk_documents() first."
        )
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.from_json(line))
    log.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


if __name__ == "__main__":
    from ingest import load_documents

    pages = load_documents()
    chunks = chunk_documents(pages)
    save_chunks(chunks)

    # Sanity print
    print(f"\n✓ {len(chunks)} chunks created.")
    print("\nSample chunk:")
    c = chunks[0]
    print(f"  ID     : {c.chunk_id}")
    print(f"  Source : {c.source}  Page: {c.page}")
    print(f"  Tokens : {c.token_count}")
    print(f"  Text   : {c.text[:200]}...")
