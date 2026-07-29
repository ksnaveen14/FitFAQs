"""

WHAT IT DOES:
  1. Loads chunks from chunk.py's JSONL output
  2. Encodes each chunk with sentence-transformers
  3. Builds a FAISS flat L2 index (exact search, no approximation)
  4. Saves: the index file + a metadata JSON mapping index position → chunk_id

"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

from schema import Chunk
from chunk import load_chunks

log = logging.getLogger(__name__)

BASE_DIR   = Path(__file__).parent
INDEX_DIR  = BASE_DIR / "indexes"      

EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, good retrieval baseline
EMBED_DIM   = 384

INDEX_FILE    = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "faiss_metadata.json"


# ── Embedding ──────────────────────────────────────────────────────────────

def get_encoder() -> SentenceTransformer:
    """Load (and cache) the sentence transformer model."""
    log.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def encode_chunks(
    chunks: List[Chunk],
    encoder: SentenceTransformer,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode all chunk texts into a float32 matrix of shape (N, EMBED_DIM).

    batch_size=64 is a memory/speed balance. Reduce if you hit OOM.
    show_progress_bar=True is helpful for large corpora.
    """
    texts = [c.text for c in chunks]
    log.info(f"Encoding {len(texts)} chunks (batch={batch_size})…")

    embeddings = encoder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalise → cosine similarity via dot product
    )

    # Ensure float32 - FAISS needs it
    return embeddings.astype(np.float32)


# ── FAISS index ────────────────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS Inner Product (IP) index.

    """
    n, d = embeddings.shape
    assert d == EMBED_DIM, f"Dimension mismatch: got {d}, expected {EMBED_DIM}"

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    log.info(f"FAISS index built: {index.ntotal} vectors, dim={d}")
    return index


def save_index(index: faiss.IndexFlatIP, chunks: List[Chunk]) -> None:
    """
    Persist FAISS index + metadata to disk.

    metadata.json maps integer position (FAISS row) → chunk_id.
    retrieve.py uses this to go from FAISS results back to Chunk objects.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    log.info(f"FAISS index saved → {INDEX_FILE}")

    # metadata: list of chunk dicts in FAISS order
    metadata = [c.to_dict() for c in chunks]
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Metadata saved → {METADATA_FILE} ({len(metadata)} entries)")


def load_index() -> tuple[faiss.IndexFlatIP, List[Chunk]]:
    """
    Load FAISS index + chunk metadata from disk.
    retrieve.py calls this at startup.
    """
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run embed.py first."
        )

    index = faiss.read_index(str(INDEX_FILE))
    log.info(f"FAISS index loaded: {index.ntotal} vectors")

    with open(METADATA_FILE) as f:
        metadata = json.load(f)
    chunks = [Chunk.from_dict(d) for d in metadata]
    log.info(f"Metadata loaded: {len(chunks)} chunks")

    return index, chunks


# ── Pipeline entry point ───────────────────────────────────────────────────

def build_and_save_index() -> None:
    chunks = load_chunks()
    encoder = get_encoder()
    embeddings = encode_chunks(chunks, encoder)
    index = build_faiss_index(embeddings)
    save_index(index, chunks)
    print(f"\n✓ H2 complete. Index has {index.ntotal} vectors.")


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_and_save_index()
