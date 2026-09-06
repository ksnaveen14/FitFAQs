"""
generate.py : LLM generation with citation layer.

WHAT IT DOES:
  1. Takes retrieved chunks (List[Chunk]) from retrieve.py
  2. Formats them into a grounded context prompt
  3. Calls OpenAI Chat Completions API
  4. Returns answer text + structured citations

"""

import os
import logging
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from chunk import Chunk

OPENAI_MODEL      = "gpt-3.5-turbo"
OPENAI_MAX_TOKENS = 512
OPENAI_TEMPERATURE = 0.2   # low = factual, deterministic.
FALLBACK_MESSAGE = (
    "I couldn't find relevant information in the fitness/nutrition documents. "
    "Please consult a certified nutritionist or trainer for personalised advice."
)
log = logging.getLogger(__name__)


# ── Citation schema ────────────────────────────────────────────────────────

@dataclass
class Citation:
    """One cited source passage."""
    source     : str    # filename
    page       : int
    chunk_id   : str
    excerpt    : str    # first 200 chars of the chunk
    rrf_score  : float


@dataclass
class RAGResponse:
    """Full response object returned to app.py."""
    answer     : str
    citations  : List[Citation]
    used_fallback: bool = False
    query      : str = ""


# ── Prompt construction ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable fitness and nutrition assistant.
Your role is to provide accurate, evidence-based information about exercise,
nutrition, meal planning, and healthy lifestyle choices.

STRICT RULES:
1. Answer ONLY using the provided context passages.
2. If the answer is not in the context, say: "I don't have enough information
   in the provided documents to answer this reliably."
3. Do not make up statistics, dosages, or recommendations not present in context.
4. Cite sources inline using [Source: filename, Page N] format.
5. Be concise and practical. Bullet points are welcome for lists."""


def build_user_prompt(query: str, chunks: List[Chunk]) -> str:
    """
    Constructs the user turn of the prompt.

    Format:
        CONTEXT PASSAGES:
        [1] Source: X, Page Y
        <text>
        ...
        QUESTION:
        <query>

    Numbered passages let the LLM reference them easily.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk.source}, Page {chunk.page}\n{chunk.text}"
        )
    context_str = "\n\n".join(context_parts)

    return (
        f"CONTEXT PASSAGES:\n{context_str}\n\n"
        f"QUESTION:\n{query}\n\n"
        f"Answer based only on the context above. "
        f"Cite sources inline as [Source: filename, Page N]."
    )


# ── LLM call ──────────────────────────────────────────────────────────────

def call_openai(system: str, user: str) -> str:
    """
    Thin wrapper around OpenAI Chat Completions.
    API key is read from OPENAI_API_KEY environment variable.
    Raises on API errors — callers should handle.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=OPENAI_MAX_TOKENS,
        temperature=OPENAI_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ── Citation extraction ────────────────────────────────────────────────────

def build_citations(chunks: List[Chunk]) -> List[Citation]:
    """Build Citation objects from retrieved chunks for display in UI."""
    return [
        Citation(
            source    = c.source,
            page      = c.page,
            chunk_id  = c.chunk_id,
            excerpt   = c.text[:200] + ("…" if len(c.text) > 200 else ""),
            rrf_score = c.score,
        )
        for c in chunks
    ]


# ── Main generation function ───────────────────────────────────────────────

def generate_answer(query: str, chunks: List[Chunk]) -> RAGResponse:
    """
    Generate a grounded answer with citations.

    Args:
        query  : the user's question
        chunks : retrieved chunks from retrieve.py (may be empty → fallback)

    Returns:
        RAGResponse with answer text, citations, and fallback flag.
    """
    # ── Fallback gate ──────────────────────────────────────────────────────
    if not chunks:
        log.warning("No chunks provided to generate.py → using fallback message.")
        return RAGResponse(
            answer=FALLBACK_MESSAGE,
            citations=[],
            used_fallback=True,
            query=query,
        )

    # ── Build prompt ───────────────────────────────────────────────────────
    user_prompt = build_user_prompt(query, chunks)

    # ── Call LLM ──────────────────────────────────────────────────────────
    try:
        answer = call_openai(SYSTEM_PROMPT, user_prompt)
        log.info(f"LLM answered ({len(answer)} chars)")
    except Exception as e:
        log.error(f"OpenAI call failed: {e}")
        return RAGResponse(
            answer=f"LLM error: {e}. Please check your OPENAI_API_KEY.",
            citations=[],
            used_fallback=True,
            query=query,
        )

    # ── Build citations ────────────────────────────────────────────────────
    citations = build_citations(chunks)

    return RAGResponse(
        answer=answer,
        citations=citations,
        used_fallback=False,
        query=query,
    )


# ── CLI test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from retrieve import Retriever

    query = " ".join(sys.argv[1:]) or "How much protein should I eat daily?"
    print(f"\nQuery: {query}\n")

    r = Retriever()
    chunks = r.query(query)

    response = generate_answer(query, chunks)

    print("=" * 60)
    print("ANSWER:")
    print(response.answer)
    print("\nCITATIONS:")
    for c in response.citations:
        print(f"  [{c.source}, p{c.page}] score={c.rrf_score:.4f}")
        print(f"    {c.excerpt[:100]}…")
    if response.used_fallback:
        print("\n Fallback was triggered.")
