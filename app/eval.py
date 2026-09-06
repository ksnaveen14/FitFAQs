"""
eval.py: RAGAS evaluation.

WHAT IT DOES:
  1. Auto-generates question-answer pairs from our chunks (no manual labelling)
  2. Runs them through the RAG pipeline (retrieve → generate)
  3. Evaluates with RAGAS metrics:
       - faithfulness      : is the answer supported by the retrieved context?
       - answer_relevancy  : does the answer address the question?
       - context_recall    : did we retrieve the relevant chunks?
       - context_precision : are the retrieved chunks actually useful?

"""

import json
import logging
import os
from pathlib import Path
from typing import List

from openai import OpenAI
from datasets import Dataset


from chunk import Chunk
from chunk import load_chunks
from retrieve import Retriever
from generate import generate_answer

RAGAS_SAMPLE_SIZE = 10   # number of Q&A pairs to auto-generate for eval
LOG_DIR    = Path(__file__).parent / "logs"
OPENAI_MODEL      = "gpt-3.5-turbo"

log = logging.getLogger(__name__)
EVAL_RESULTS_FILE = LOG_DIR / "ragas_results.json"


# -- Step 1: Synthetic Q&A generation ----------------------------------

def generate_question_from_chunk(chunk: Chunk, client: OpenAI) -> str | None:
    """
    Ask the LLM to generate one factual question answerable from this chunk.
    Returns None if generation fails.
    """
    prompt = (
        f"Read the following fitness/nutrition passage and write ONE clear, "
        f"specific question that can be answered directly from the text. "
        f"Only output the question, nothing else.\n\n"
        f"PASSAGE:\n{chunk.text}"
    )
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        q = response.choices[0].message.content.strip()
        return q if q.endswith("?") else q + "?"
    except Exception as e:
        log.warning(f"Q-gen failed for chunk {chunk.chunk_id}: {e}")
        return None


def build_eval_dataset(
    chunks: List[Chunk],
    retriever: Retriever,
    sample_size: int = RAGAS_SAMPLE_SIZE,
) -> list[dict]:
    """
    Build a RAGAS-compatible dataset by:
      1. Sampling chunks
      2. Generating a question per chunk
      3. Running the full RAG pipeline to get answer + contexts
      4. Using the source chunk text as 'ground_truth'

    Returns list of dicts with keys:
      question, answer, contexts, ground_truth
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    import random
    sampled = random.sample(chunks, min(sample_size, len(chunks)))

    rows = []
    for i, chunk in enumerate(sampled):
        log.info(f"Generating eval pair {i+1}/{len(sampled)}…")

        question = generate_question_from_chunk(chunk, client)
        if not question:
            continue

        retrieved = retriever.query(question)
        response  = generate_answer(question, retrieved)

        if response.used_fallback:
            log.info(f"  Skipping (fallback triggered) — q: {question[:50]}")
            continue

        rows.append({
            "question"    : question,
            "answer"      : response.answer,
            "contexts"    : [c.text for c in retrieved],
            "ground_truth": chunk.text,   # the chunk the question was generated from
        })

    log.info(f"Eval dataset built: {len(rows)} valid pairs")
    return rows


# -- Step 2: RAGAS evaluation ---------------------------------------------

def run_ragas_eval(rows: list[dict]) -> dict:
    """
    Run RAGAS metrics on the dataset.

    RAGAS expects a HuggingFace Dataset object with columns:
      question, answer, contexts (List[str]), ground_truth

    Returns a dict of metric_name → float score.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )

    if not rows:
        log.error("No eval rows to evaluate. Check Q-gen step.")
        return {}

    required_cols = {"question", "answer", "contexts", "ground_truth"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        log.error(f"Eval rows missing required columns: {missing}")
        return {}

    dataset = Dataset.from_list(rows)

    log.info("Running RAGAS evaluation…")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )
    df = result.to_pandas()
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    for m in metric_names:
        nan_count = df[m].isna().sum()
        if nan_count:
            log.warning(f"{m}: {nan_count}/{len(df)} rows returned NaN")

    summary = {m: float(df[m].mean(skipna=True)) for m in metric_names}

    return {
        "summary": summary,
        "per_row": df.to_dict(orient="records"),
    }


# -- Step 3: Save + report -----------------------------------------------

def save_results(result: dict, rows: list[dict]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": result["summary"],
        "per_row": result["per_row"],
        "num_pairs": len(rows),
        "sample_pairs": rows[:3],  # save first 3 for inspection
    }
    with open(EVAL_RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved → {EVAL_RESULTS_FILE}")


def print_report(result: dict) -> None:
    summary = result["summary"]

    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in summary.items():
        bar = "█" * int(score * 20)
        status = "✓" if score >= 0.7 else ("~" if score >= 0.5 else "✗")
        print(f"  {status} {metric:<25} {score:.3f}  {bar}")
    print("=" * 50)
    print("  ✓ ≥ 0.7 (good)  ~  0.5-0.7 (tune)  ✗ < 0.5 (fix)")

    nan_metrics = [m for m, s in summary.items() if s != s]  # NaN check
    if nan_metrics:
        print(f"  ⚠ NaN detected in: {', '.join(nan_metrics)} — check logs")

# -- CLI entry point -----------------------------------------------------
if __name__ == "__main__":
    chunks    = load_chunks()
    retriever = Retriever()

    rows   = build_eval_dataset(chunks, retriever)
    result = run_ragas_eval(rows)
    save_results(result, rows)
    print_report(result)
