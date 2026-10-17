"""
app.py — Stage D of "Ask Your Notes" RAG pipeline
================================================================
PURPOSE:
    Streamlit interactive web user interface for "Ask Your Notes".
    - Upload PDFs directly to the data/ folder.
    - Trigger indexing (ingest.py) with a single click.
    - Ask questions about your notes.
    - View grounded Gemini answers with inline citations.
    - Inspect retrieved context chunks, filenames, page numbers,
      and similarity scores inside an interactive expander.

HOW TO RUN:
    streamlit run app.py
"""

import os
import streamlit as st
from ingest import ingest, DATA_DIR, FAISS_INDEX_PATH
from llm import generate_answer

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ask Your Notes - RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# ── Custom Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .answer-box {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-left: 5px solid #2563EB !important;
        padding: 1.2rem !important;
        border-radius: 8px !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    .answer-box * {
        color: #0F172A !important;
    }
    .chunk-card {
        background: #FFFFFF !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        padding: 0.9rem !important;
        margin-bottom: 0.8rem !important;
    }
    .chunk-card * {
        color: #1E293B !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar: Document Management ──────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Manager")
    st.write("Upload lecture notes (PDF) to build your knowledge base.")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_btn = st.button("Save & Ingest Documents", type="primary")
        if save_btn:
            os.makedirs(DATA_DIR, exist_ok=True)
            saved_count = 0
            for uploaded_file in uploaded_files:
                dest_path = os.path.join(DATA_DIR, uploaded_file.name)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_count += 1

            st.success(f"Saved {saved_count} file(s) to `{DATA_DIR}/`")

            # Run ingestion pipeline
            with st.spinner("Parsing PDFs, running OCR on scanned pages, & updating FAISS index..."):
                try:
                    ingest()
                    st.success("✅ Indexing complete! Ready to answer questions.")
                    # Invalidate cache in retrieve module
                    import retrieve
                    retrieve._cached_index = None
                    retrieve._cached_chunks = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")

    st.markdown("---")
    st.subheader("📚 Active Documents")
    if os.path.exists(DATA_DIR):
        existing_pdfs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
        if existing_pdfs:
            for p in existing_pdfs:
                st.caption(f"• {p}")
            if st.button("🗑️ Clear All Documents & Index", help="Delete uploaded PDFs and reset index"):
                for p in existing_pdfs:
                    try:
                        os.remove(os.path.join(DATA_DIR, p))
                    except Exception:
                        pass
                if os.path.exists("index/faiss.index"):
                    try:
                        os.remove("index/faiss.index")
                        os.remove("index/chunks.pkl")
                    except Exception:
                        pass
                import retrieve
                retrieve._cached_index = None
                retrieve._cached_chunks = None
                st.success("All documents cleared.")
                st.rerun()
        else:
            st.caption("No PDFs in library.")
    else:
        st.caption("No PDFs in library.")

    # Status indicator
    st.markdown("---")
    if os.path.exists(FAISS_INDEX_PATH):
        st.success("🟢 FAISS Vector Index: Active")
    else:
        st.warning("🟡 FAISS Vector Index: Missing (Run Ingest)")


# ── Main Content Area ─────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📚 Ask Your Notes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Grounded AI study assistant powered by <b>FAISS dense retrieval</b> and <b>Google Gemini</b>. '
    'Answers strictly adhere to your documents with source & page citations.</div>',
    unsafe_allow_html=True
)

# Search Query Input
user_question = st.text_input(
    "Ask a question about your uploaded lecture notes:",
    placeholder="e.g., How does gradient descent minimize the loss function?"
)

col_ask, col_topk = st.columns([4, 1])
with col_topk:
    top_k_val = st.slider("Retrieved Excerpts (Top-K)", min_value=1, max_value=8, value=4)

if st.button("Get Answer", type="primary", use_container_width=True) or user_question:
    if not user_question.strip():
        st.info("Please enter a question to query your notes.")
    elif not os.path.exists(FAISS_INDEX_PATH):
        st.error("No FAISS index found! Please upload a PDF and click 'Save & Ingest Documents' in the sidebar first.")
    else:
        with st.spinner("Searching document index and consulting Gemini..."):
            result = generate_answer(user_question, top_k=top_k_val)

        st.markdown("### 💡 Answer")
        st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

        # Expander showing retrieved excerpts for verification & auditability
        with st.expander(f"🔍 Inspect Retrieved Excerpts ({len(result['chunks'])} chunks found)"):
            if not result["chunks"]:
                st.write("No excerpts retrieved.")
            else:
                for idx, chunk in enumerate(result["chunks"], start=1):
                    score_pct = int(chunk.get("score", 0.0) * 100)
                    st.markdown(f"""
                    <div class="chunk-card">
                        <b>Excerpt #{idx}</b> — <code>{chunk['source']} (Page {chunk['page']})</code> | <i>Similarity: {chunk.get('score', 0):.4f} (~{score_pct}%)</i>
                        <p style="margin-top: 0.5rem; color: #374151;">{chunk['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
