"""
config.py — Single source of truth.
Every other module imports from here. Change once, affects all.
This is how you defend against drift in interviews.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_RAW   = BASE_DIR / "data" / "raw"       # drop your PDFs / TXTs here
DATA_CHUNKS = BASE_DIR / "data" / "chunks"   # chunk.py writes here
INDEX_DIR  = BASE_DIR / "indexes"            # embed.py writes FAISS index here
LOG_DIR    = BASE_DIR / "logs"

for _dir in [DATA_RAW, DATA_CHUNKS, INDEX_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE    = 512   # tokens / words. Interview: "why 512?" → fits 1 idea,
                      # avoids crossing context boundaries, standard in RAG lit.
CHUNK_OVERLAP = 50    # overlap so sentences at boundaries aren't orphaned

# ── Embedding ─────────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # fast, 384-dim, good retrieval baseline
EMBED_DIM   = 384

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K_DENSE  = 5    # FAISS returns top-5 dense hits
TOP_K_SPARSE = 5    # BM25 returns top-5 sparse hits
TOP_K_FINAL  = 5    # after RRF fusion, keep top-5 for LLM context
RRF_K        = 60   # Reciprocal Rank Fusion constant (standard = 60)

# ── LLM ───────────────────────────────────────────────────────────────────
OPENAI_MODEL      = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 512
OPENAI_TEMPERATURE = 0.2   # low = factual, deterministic. Interview: explain tradeoff.

# ── RAGAS Eval ────────────────────────────────────────────────────────────
RAGAS_SAMPLE_SIZE = 10   # number of Q&A pairs to auto-generate for eval

# ── Fallback ──────────────────────────────────────────────────────────────
FALLBACK_MESSAGE = (
    "I couldn't find relevant information in the fitness/nutrition documents. "
    "Please consult a certified nutritionist or trainer for personalised advice."
)

LOW_CONFIDENCE_THRESHOLD = 0.25
# RRF scores are unbounded-threshold-unsafe; gate on rank instead.
RANK_QUALITY_THRESHOLD = 3  # top result must rank in top-3 of dense OR sparse