"""
pipeline.py — single entry point for retrieve -> generate, with timing.
"""
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from retrieve import Retriever
from generate import generate_answer


@dataclass
class PipelineResult:
    response: Any                 # whatever generate_answer returns
    chunks: List[Any]
    latency_ms: float             # end to end
    retrieval_ms: float
    generation_ms: float
    top_rrf_score: Optional[float]
    max_sim_score: Optional[float]   # max dense cosine similarity over cited chunks;
                                     # None if no citations (fallback case).
                                     # RRF scores are rank-based and not comparable
                                     # across queries, cosine is.

def run_query(retriever: Retriever, query: str) -> PipelineResult:
    t0 = time.perf_counter()
    chunks = retriever.query(query)
    t1 = time.perf_counter()
    response = generate_answer(query, chunks)
    t2 = time.perf_counter()

    citations = response.citations or []
    rrf_scores = [c.rrf_score for c in citations]
    dense_scores = [c.dense_score for c in citations]

    return PipelineResult(
        response=response,
        chunks=chunks,
        latency_ms=(t2 - t0) * 1000,
        retrieval_ms=(t1 - t0) * 1000,
        generation_ms=(t2 - t1) * 1000,
        top_rrf_score=max(rrf_scores) if rrf_scores else None,
        max_sim_score=max(dense_scores) if dense_scores else None,
    )
