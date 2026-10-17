"""
retrieve.py — Stage B of "Ask Your Notes" RAG pipeline
================================================================
PURPOSE:
    Load the saved FAISS index and chunk metadata from index/,
    embed an incoming user question using the same embedding model
    (all-MiniLM-L6-v2), and find the top-K most semantically similar
    chunks based on cosine similarity (L2 on normalized vectors).

HOW TO RUN (CLI test without LLM):
    python retrieve.py "What is gradient descent?"

OUTPUT:
    Prints top-4 chunks with similarity scores, source file, and page numbers.
"""

import os
import pickle
import sys
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Paths & Constants ──────────────────────────────────────────────────────────

INDEX_DIR = "index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 4  # Number of relevant chunks to retrieve


# ── Global Cache for Model & Index ─────────────────────────────────────────────
# In applications (like Streamlit or repeated CLI calls), caching avoids
# reloading the model and index on every query.
_cached_model = None
_cached_index = None
_cached_chunks = None


def load_resources():
    """
    Load the embedding model, FAISS index, and chunks metadata.
    Returns:
        (model, index, chunks)
    """
    global _cached_model, _cached_index, _cached_chunks

    if _cached_model is None:
        _cached_model = SentenceTransformer(MODEL_NAME)

    if _cached_index is None or _cached_chunks is None:
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
            raise FileNotFoundError(
                f"Index files missing in '{INDEX_DIR}/'. Run 'python ingest.py' first!"
            )
        _cached_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            _cached_chunks = pickle.load(f)

    return _cached_model, _cached_index, _cached_chunks


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top_k chunks relevant to the user query.

    Steps:
        1. Embed the query with SentenceTransformer.
        2. Normalize query vector to unit length (L2 norm = 1.0).
        3. Search the FAISS index for top_k nearest neighbors (IndexFlatL2).
        4. Map indices back to chunk text, filename, and page number.

    Returns:
        List of dicts:
        [
            {
                "text": str,
                "source": str,
                "page": int,
                "score": float  # Cosine similarity score between 0 and 1
            },
            ...
        ]
    """
    if not query.strip():
        return []

    model, index, chunks = load_resources()

    # Step 1: Embed query into a 384-dimensional vector
    query_vector = model.encode([query], convert_to_numpy=True)

    # Step 2: Normalize query vector (matches L2-normalization in ingest.py)
    # Cosine similarity = 1 - (L2_distance^2 / 2) for unit-normalized vectors.
    faiss.normalize_L2(query_vector)

    # Step 3: Search FAISS index for k nearest neighbors
    # distances: shape (1, top_k) — squared L2 distances
    # indices: shape (1, top_k) — row indices in the chunk list
    distances, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:  # FAISS returns -1 if fewer vectors exist than top_k
            continue

        chunk_data = chunks[idx].copy()
        
        # Convert squared L2 distance to cosine similarity:
        # Since ||u|| = 1 and ||v|| = 1:
        # L2_dist_sq = ||u - v||^2 = ||u||^2 + ||v||^2 - 2*(u . v) = 2 - 2 * cos_sim
        # => cos_sim = 1 - (L2_dist_sq / 2)
        cos_sim = 1.0 - (float(dist) / 2.0)
        chunk_data["score"] = round(cos_sim, 4)

        results.append(chunk_data)

    return results


# ── CLI Test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_query = " ".join(sys.argv[1:])
    else:
        test_query = "What is the main topic of these notes?"

    print(f"\n[QUERY] \"{test_query}\"")
    print("=" * 60)

    try:
        retrieved_chunks = retrieve(test_query, top_k=TOP_K)
        if not retrieved_chunks:
            print("No matching chunks found.")
        for rank, chunk in enumerate(retrieved_chunks, start=1):
            print(f"\n[Rank {rank}] Similarity: {chunk['score']:.4f}")
            print(f"Source: [{chunk['source']}, p.{chunk['page']}]")
            print(f"Excerpt:\n\"{chunk['text']}\"")
            print("-" * 60)
    except FileNotFoundError as e:
        print(f"\n[!] Error: {e}")
