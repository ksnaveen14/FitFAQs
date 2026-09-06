"""
retrieve.py : Hybrid retrieval (BM25 + Dense) with RRF fusion.

WHAT IT DOES:
  Given a user query, this module:
    1. Runs BM25 (sparse) → keyword-matching score per chunk
    2. Runs FAISS (dense) → semantic similarity per chunk
    3. Fuses rankings with Reciprocal Rank Fusion (RRF)
    4. Returns top-K Chunk objects with .score set
"""

import logging
import numpy as np
from typing import List

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


from chunk import Chunk
from embed import load_index, get_encoder
from embed import EMBED_MODEL, EMBED_DIM

TOP_K_DENSE  = 5    # FAISS returns top-5 dense hits
TOP_K_SPARSE = 5    # BM25 returns top-5 sparse hits
TOP_K_FINAL  = 5    # after RRF fusion, keep top-5 for LLM context
RRF_K        = 60   # Reciprocal Rank Fusion constant (standard = 60)

# Replaces LOW_CONFIDENCE_THRESHOLD = 0.25
# RRF scores are unbounded-threshold-unsafe; gate on rank instead.
RANK_QUALITY_THRESHOLD = 3  # top result must rank in top-3 of dense OR sparse
log = logging.getLogger(__name__)


# ── Retrieval components ───────────────────────────────────────────────────

def build_bm25(chunks: List[Chunk]) -> BM25Okapi:
    """
    Build BM25 index from chunk texts.
    Tokenised as lowercase word lists — matches BM25's expectation.
    BM25Okapi is the standard variant with tf saturation parameter k1.
    """
    tokenised_corpus = [c.text.lower().split() for c in chunks]
    return BM25Okapi(tokenised_corpus)


def dense_search(
    query: str,
    encoder: SentenceTransformer,
    faiss_index,
    chunks: List[Chunk],
    top_k: int = TOP_K_DENSE,
) -> list[tuple[int, float]]:
    """
    Query FAISS index. Returns list of (chunk_index, score) sorted by score desc.
    Score = cosine similarity (0-1 range since embeddings are normalised).
    """
    query_vec = encoder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, indices = faiss_index.search(query_vec, top_k)
    # scores and indices are shape (1, top_k)
    results = [
        (int(idx), float(score))
        for idx, score in zip(indices[0], scores[0])
        if idx >= 0  # FAISS returns -1 for padding if fewer results exist
    ]
    return results  # already sorted desc by FAISS


def sparse_search(
    query: str,
    bm25: BM25Okapi,
    top_k: int = TOP_K_SPARSE,
) -> list[tuple[int, float]]:
    """
    Query BM25 index. Returns list of (chunk_index, score) sorted by score desc.
    """
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    # argsort ascending, take last top_k, reverse
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
    return results


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Combine two ranked lists using RRF.

    Formula: score(doc) = Σ_retriever  1 / (rank + k)
    where rank is 1-based position in that retriever's result list.

    Returns list of (chunk_index, rrf_score) sorted by rrf_score descending.

    WHY RRF OVER SCORE NORMALISATION:
      BM25 scores can be 0–50+; cosine scores are 0–1. Simple average would
      be dominated by BM25. RRF works purely on ranks, eliminating scale bias.
    """
    rrf_scores: dict[int, float] = {}

    for rank, (chunk_idx, _) in enumerate(dense_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + k)

    for rank, (chunk_idx, _) in enumerate(sparse_results, start=1):
        rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (rank + k)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


def retrieve(
    query: str,
    encoder: SentenceTransformer,
    faiss_index,
    chunks: List[Chunk],
    bm25: BM25Okapi,
    top_k: int = TOP_K_FINAL,
) -> List[Chunk]:
    """
    Full hybrid retrieval pipeline.

    Args:
        query       : user's natural language question
        encoder     : sentence-transformer model
        faiss_index : loaded FAISS index
        chunks      : list of all Chunk objects (same order as FAISS index)
        bm25        : built BM25 index
        top_k       : number of chunks to return

    Returns:
        List[Chunk] with .score set to RRF score, sorted desc.
        Returns [] if best score < RANK_QUALITY_THRESHOLD (triggers fallback).
    """
    dense_res  = dense_search(query, encoder, faiss_index, chunks)
    sparse_res = sparse_search(query, bm25)
  
    log.debug(f"Dense hits: {len(dense_res)}, Sparse hits: {len(sparse_res)}")
    if not sparse_res:
            log.debug("Sparse retrieval returned zero hits — hybrid degraded to dense-only for this query.")
    fused = reciprocal_rank_fusion(dense_res, sparse_res)

    if not fused:
        log.warning("No retrieval results found.")
        return []

    top_doc_id = fused[0][0]
    dense_ranks = { idx: rank for rank, (idx, _) in enumerate(dense_res, start=1) }
    sparse_ranks = { idx: rank for rank, (idx, _) in enumerate(sparse_res, start=1) }

    best_dense_rank = dense_ranks.get(top_doc_id, float('inf'))
    best_sparse_rank = sparse_ranks.get(top_doc_id, float('inf'))
    best_individual_rank = min(best_dense_rank, best_sparse_rank)

    if best_individual_rank > RANK_QUALITY_THRESHOLD:
            log.warning(
                f"Top fused doc {top_doc_id} only reached rank {best_individual_rank} "
                f"in its best retriever (threshold: top-{RANK_QUALITY_THRESHOLD}). "
                "Triggering fallback."
            )
            return []
    # Build output: attach scores and return Chunk objects
    result_chunks: List[Chunk] = []
    for chunk_idx, rrf_score in fused[:top_k]:
        if 0 <= chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            chunk.score = rrf_score
            result_chunks.append(chunk)

    return result_chunks


# ── Convenience loader (used by app.py and generate.py) ───────────────────

class Retriever:
    """
    Stateful wrapper that loads all retrieval components once at startup.
    app.py instantiates this once and reuses it across queries.

    """
    def __init__(self):
        log.info("Initialising retriever…")
        self.faiss_index, self.chunks = load_index()
        self.encoder = get_encoder()
        self.bm25 = build_bm25(self.chunks)
        log.info("Retriever ready.")

    def query(self, question: str, top_k: int = TOP_K_FINAL) -> List[Chunk]:
        return retrieve(
            query=question,
            encoder=self.encoder,
            faiss_index=self.faiss_index,
            chunks=self.chunks,
            bm25=self.bm25,
            top_k=top_k,
        )


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) or "How much protein do I need?"
    print(f"\nQuery: {query}")

    r = Retriever()
    results = r.query(query)

    if not results:
        print("\n⚠ No confident results found. Fallback would trigger.")
    else:
        print(f"\nTop {len(results)} chunks:\n")
        for i, chunk in enumerate(results, 1):
            print(f"[{i}] Score={chunk.score:.4f} | {chunk.source} p{chunk.page}")
            print(f"    {chunk.text[:200]}…\n")
