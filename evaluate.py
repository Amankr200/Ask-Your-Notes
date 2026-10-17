"""
evaluate.py — Stage E of "Ask Your Notes" RAG pipeline
================================================================
PURPOSE:
    Benchmark the retrieval engine by evaluating the Hit Rate @ K.
    - Reads test questions and ground-truth page numbers from eval_questions.json.
    - Executes retrieval across various Top-K settings (e.g. k=1, k=2, k=4).
    - Checks whether the correct ground-truth page was retrieved.
    - Computes and prints the retrieval Hit Rate (accuracy percentage).

WHY EVALUATE RETRIEVAL INDEPENDENTLY?
    In production RAG systems, if the retrieval engine fails to surface
    the relevant document passage, the LLM will inevitably hallucinate
    or fail to answer. Testing retrieval in isolation provides an objective,
    deterministic metric without incurring LLM token costs.

HOW TO RUN:
    python evaluate.py
"""

import json
import os
import sys
from retrieve import retrieve

EVAL_FILE = "eval_questions.json"


def load_eval_questions(filepath: str = EVAL_FILE) -> list[dict]:
    """Load benchmark test cases from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Evaluation dataset '{filepath}' not found. Please create it first."
        )
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(questions: list[dict], k_values: list[int] = [1, 2, 4]) -> dict:
    """
    Compute Hit Rate @ K across the benchmark questions.

    A test case is considered a "Hit" if at least one of the top-k retrieved
    chunks comes from the expected document and expected page.

    Hit Rate @ K = (Total Hits @ K) / (Total Questions)
    """
    total_q = len(questions)
    if total_q == 0:
        return {}

    # Store hit counters for each K
    hits_per_k = {k: 0 for k in k_values}
    detailed_results = []

    print(f"\nEvaluating {total_q} test questions across K = {k_values} ...")
    print("=" * 70)

    for q in questions:
        q_id = q.get("id", "-")
        question_text = q["question"]
        expected_src = q["expected_source"]
        expected_pg = q["expected_page"]

        # Retrieve maximum K required
        max_k = max(k_values)
        retrieved_chunks = retrieve(question_text, top_k=max_k)

        # Check hits at each K threshold
        hit_at = {}
        for k in k_values:
            top_k_chunks = retrieved_chunks[:k]
            # Check if any chunk in top_k matches expected_source and expected_page
            is_hit = any(
                c["source"] == expected_src and c["page"] == expected_pg
                for c in top_k_chunks
            )
            hit_at[k] = is_hit
            if is_hit:
                hits_per_k[k] += 1

        # Determine rank where ground truth first appeared (for MRR tracking)
        first_rank = None
        for rank, c in enumerate(retrieved_chunks, start=1):
            if c["source"] == expected_src and c["page"] == expected_pg:
                first_rank = rank
                break

        detailed_results.append({
            "id": q_id,
            "question": question_text,
            "expected": f"{expected_src} p.{expected_pg}",
            "first_hit_rank": first_rank,
            "hit_at": hit_at
        })

        status_str = f"Rank {first_rank}" if first_rank else "MISS"
        print(f"[{q_id:02d}] Expected: p.{expected_pg} | First Hit: {status_str:<6} | Q: \"{question_text[:45]}...\"")

    print("=" * 70)
    print("\nRETRIEVAL BENCHMARK SUMMARY:")
    print("-" * 40)
    
    summary = {}
    for k in k_values:
        rate = (hits_per_k[k] / total_q) * 100.0
        summary[k] = rate
        print(f"Hit Rate @ {k}: {hits_per_k[k]}/{total_q} ({rate:.1f}%)")

    # Mean Reciprocal Rank (MRR)
    # MRR = average of (1 / first_hit_rank) across all questions
    reciprocal_ranks = [
        (1.0 / res["first_hit_rank"]) if res["first_hit_rank"] else 0.0
        for res in detailed_results
    ]
    mrr = sum(reciprocal_ranks) / total_q
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print("-" * 40)

    return {
        "hit_rates": summary,
        "mrr": round(mrr, 4),
        "total_questions": total_q,
        "details": detailed_results
    }


if __name__ == "__main__":
    try:
        data = load_eval_questions()
        results = evaluate_retrieval(data, k_values=[1, 2, 4])
    except Exception as e:
        print(f"[!] Evaluation failed: {e}")
