"""
ingest.py  —  Stage A of "Ask Your Notes" RAG pipeline
================================================================
PURPOSE:
    Read every PDF in the data/ folder, extract text page-by-page,
    split the text into overlapping chunks, embed each chunk with a
    sentence-transformer model, and save the resulting FAISS index
    plus chunk metadata to the index/ folder on disk.

    Run this once (or whenever you add new PDFs).

HOW TO RUN:
    python ingest.py

OUTPUT:
    index/faiss.index   — the vector index (binary, fast nearest-neighbour search)
    index/chunks.pkl    — list of dicts: {text, source, page}
"""

import os          # for file paths and directory creation
import pickle      # to save/load Python objects (our chunk metadata list)

import faiss                          # Facebook AI Similarity Search — vector index
import numpy as np                    # numerical arrays; FAISS speaks numpy
from pypdf import PdfReader           # to read PDF files and extract text
from sentence_transformers import SentenceTransformer  # pre-trained embedder

# ── Constants ──────────────────────────────────────────────────────────────────

# Folder where the user drops their PDFs
DATA_DIR = "data"

# Folder where the FAISS index and metadata will be saved
INDEX_DIR = "index"

# Where the FAISS binary file will live
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")

# Where the list of chunk dicts will be saved (text + source filename + page number)
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")

# The embedding model we'll use.
# all-MiniLM-L6-v2 produces 384-dimensional vectors. It is:
#   • small  (~22 MB)
#   • fast   (can embed thousands of chunks in seconds on CPU)
#   • good   (trained on 1B+ sentence pairs for semantic similarity)
MODEL_NAME = "all-MiniLM-L6-v2"

# Target chunk size in characters.
# Smaller  → more precise retrieval, but less context per chunk.
# Larger   → more context, but the retrieved chunk may contain irrelevant parts.
# ~500 chars ≈ 80–100 words ≈ roughly one paragraph.
CHUNK_SIZE = 500

# How many characters of the previous chunk to repeat at the start of the next one.
# Overlap prevents an important sentence from being cut exactly at a boundary
# and therefore missed during retrieval.
CHUNK_OVERLAP = 80


# ── Helper: chunk a single page's text ────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split `text` into a list of overlapping character-level chunks.

    Example with chunk_size=10, overlap=3 and text="ABCDEFGHIJ...":
        chunk 1: ABCDEFGHIJ
        chunk 2: HIJKLMNOPQ   ← starts 3 chars before the end of chunk 1
        ...

    Why character-level instead of word/sentence?
        Simpler to implement and explain; works well enough at this scale.
    """
    chunks = []
    start = 0  # index of the first character of the current chunk

    while start < len(text):
        end = start + chunk_size          # index of the last character + 1
        chunk = text[start:end]           # slice the text
        chunks.append(chunk.strip())      # strip leading/trailing whitespace

        # Move start forward by (chunk_size - overlap) so the next chunk
        # re-uses the last `overlap` characters of the current chunk.
        start += chunk_size - overlap

    # Filter out any chunks that became empty after stripping
    return [c for c in chunks if c]


# ── Helper: extract all chunks from one PDF ───────────────────────────────────




def ocr_page_image(page) -> str:
    """
    Fallback OCR using Gemini vision when page has no selectable text layer
    (e.g., scanned marksheets, photographed lecture slides).
    """
    if not page.images:
        return ""
    try:
        from dotenv import load_dotenv
        from google import genai
        from google.genai import types
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return ""
        client = genai.Client(api_key=api_key)
        
        extracted_texts = []
        for img in page.images:
            part = types.Part.from_bytes(data=img.data, mime_type="image/jpeg")
            prompt = "Extract all text, numbers, tables, marks, GPA, and details from this document page verbatim. Keep structure clean."
            resp = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[part, prompt]
            )
            if resp.text:
                extracted_texts.append(resp.text.strip())
        return " ".join(extracted_texts)
    except Exception as e:
        print(f"[!] OCR failed: {e}")
        return ""


def extract_chunks_from_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF, iterate over every page, and return a list of chunk dicts.
    If a page has no text (scanned PDF / image), falls back to OCR automatically.
    """
    reader = PdfReader(pdf_path)
    filename = os.path.basename(pdf_path)
    all_chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()

        # If selectable text is empty, check for scanned images and OCR them
        if not raw_text or not raw_text.strip():
            print(f"   [Notice] Page {page_num} of '{filename}' has no text layer. Running OCR on image...")
            raw_text = ocr_page_image(page)

        if not raw_text or not raw_text.strip():
            continue

        clean_text = " ".join(raw_text.split())
        page_chunks = chunk_text(clean_text)

        for chunk in page_chunks:
            all_chunks.append({
                "text":   chunk,
                "source": filename,
                "page":   page_num,
            })

    return all_chunks


# ── Main ingestion pipeline ───────────────────────────────────────────────────

def ingest():
    """
    Full pipeline:
        1. Find all PDFs in DATA_DIR
        2. Extract chunks from every PDF
        3. Embed every chunk with the sentence-transformer model
        4. Build a FAISS index from the embeddings
        5. Save the index and the chunk metadata list to disk
    """

    # ── Step 1: collect PDF paths ──────────────────────────────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)   # create data/ if it doesn't exist
    os.makedirs(INDEX_DIR, exist_ok=True)  # create index/ if it doesn't exist

    pdf_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")      # case-insensitive extension check
    ]

    if not pdf_files:
        print(f"[!] No PDFs found in '{DATA_DIR}/'. Add some PDFs and re-run.")
        return

    print(f"[*] Found {len(pdf_files)} PDF(s): {[os.path.basename(p) for p in pdf_files]}")

    # ── Step 2: extract chunks from every PDF ─────────────────────────────────
    all_chunks = []
    for pdf_path in pdf_files:
        chunks = extract_chunks_from_pdf(pdf_path)
        print(f"   -> {os.path.basename(pdf_path)}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n[*] Total chunks across all PDFs: {len(all_chunks)}")

    if not all_chunks:
        print("[!] No text could be extracted. PDFs might be scanned images.")
        return

    # ── Step 3: embed every chunk ──────────────────────────────────────────────
    print(f"\n[*] Loading embedding model '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)  # downloads model on first run (~22 MB)

    # Extract just the text strings for the model
    texts = [chunk["text"] for chunk in all_chunks]

    print(f"[*] Embedding {len(texts)} chunks ...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # ── Step 4: build FAISS index ──────────────────────────────────────────────
    embedding_dim = embeddings.shape[1]   # 384 for MiniLM-L6-v2
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)

    print(f"\n[+] FAISS index built: {index.ntotal} vectors, dim={embedding_dim}")

    # ── Step 5: save to disk ───────────────────────────────────────────────────
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"[+] FAISS index saved -> {FAISS_INDEX_PATH}")

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"[+] Chunk metadata saved -> {CHUNKS_PATH}")
    print("\n[SUCCESS] Ingestion complete! Run 'python retrieve.py \"your question\"' next.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ingest()
